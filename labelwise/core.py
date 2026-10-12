"""Validated text classification and reproducible active-learning evaluation."""
import csv
import hashlib
import io
import json
import math
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

MAX_BYTES = 5 * 1024 * 1024
MAX_ROWS = 5000


def parse_csv(raw, labelled=True):
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("CSV must contain data and be at most 5 MiB.")
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), strict=True)
        expected = ["id", "text", "label"] if labelled else ["id", "text"]
        if reader.fieldnames != expected:
            raise ValueError("CSV header must be exactly " + ",".join(expected) + ".")
        rows, ids = [], set()
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Row {line}: unexpected columns.")
            item_id, text = row["id"].strip(), row["text"].strip()
            if not item_id or len(item_id) > 100 or item_id in ids:
                raise ValueError(f"Row {line}: IDs must be unique and 1–100 characters.")
            if not text or len(text) > 10000:
                raise ValueError(f"Row {line}: text must contain 1–10,000 characters.")
            item = {"id": item_id, "text": text}
            if labelled:
                label = row["label"].strip()
                if not label or len(label) > 100:
                    raise ValueError(f"Row {line}: label must contain 1–100 characters.")
                item["label"] = label
            ids.add(item_id); rows.append(item)
            if len(rows) > MAX_ROWS:
                raise ValueError("Maximum 5,000 rows per file.")
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("CSV must be well-formed UTF-8.") from exc
    if not rows:
        raise ValueError("CSV has no records.")
    return rows


def validate_splits(seed, pool, test=None, pool_labelled=False):
    groups = [seed, pool] + ([test] if test else [])
    ids = [set(r["id"] for r in group) for group in groups]
    if any(ids[i] & ids[j] for i in range(len(ids)) for j in range(i + 1, len(ids))):
        raise ValueError("IDs must not overlap across seed, pool and test splits.")
    texts = [set(r["text"] for r in group) for group in groups]
    if any(texts[i] & texts[j] for i in range(len(texts)) for j in range(i + 1, len(texts))):
        raise ValueError("Exact text must not overlap across seed, pool and test splits.")
    labels = sorted({r["label"] for r in seed})
    if len(labels) < 2 or any(sum(r["label"] == label for r in seed) < 2 for label in labels):
        raise ValueError("Seed needs at least two examples from every one of at least two classes.")
    if test and set(r["label"] for r in test) - set(labels):
        raise ValueError("Test labels must already be represented in the seed set.")
    if pool_labelled and set(r["label"] for r in pool) - set(labels):
        raise ValueError("Pool labels must already be represented in the seed set.")
    return labels


def _fit(rows):
    labels = [r["label"] for r in rows]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=20000)
    try:
        matrix = vectorizer.fit_transform([r["text"] for r in rows])
    except ValueError as exc:
        raise ValueError("Training text produced an empty vocabulary; include word characters.") from exc
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight="balanced")
    model.fit(matrix, labels)
    return vectorizer, model


def train_and_queue(labelled, pool, queue_size=10):
    if type(queue_size) is not int or not 1 <= queue_size <= 100:
        raise ValueError("queue_size must be an integer from 1 to 100.")
    validate_splits(labelled, pool)
    started = time.perf_counter(); vectorizer, model = _fit(labelled)
    probabilities = model.predict_proba(vectorizer.transform([r["text"] for r in pool]))
    confidence = probabilities.max(axis=1)
    order = sorted(range(len(pool)), key=lambda i: (confidence[i], pool[i]["id"]))[:queue_size]
    queue = []
    for rank, i in enumerate(order, 1):
        predicted = str(model.classes_[int(np.argmax(probabilities[i]))])
        queue.append({**pool[i], "rank": rank, "predicted_label": predicted,
                      "confidence": float(confidence[i]), "uncertainty": float(1 - confidence[i])})
    return {"classes": [str(x) for x in model.classes_], "labelled_count": len(labelled),
            "pool_count": len(pool), "queue": queue,
            "fit_and_queue_ms": (time.perf_counter() - started) * 1000,
            "method": "least confidence: 1 - maximum predicted class probability"}


def _score(train, test):
    vectorizer, model = _fit(train)
    predicted = model.predict(vectorizer.transform([r["text"] for r in test]))
    truth = [r["label"] for r in test]
    return {"accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0))}


def benchmark(seed, pool, test, budgets=(0, 5, 10, 20), random_repeats=20, random_seed=42):
    labels = validate_splits(seed, pool, test, pool_labelled=True)
    if type(random_repeats) is not int or not 1 <= random_repeats <= 100:
        raise ValueError("random_repeats must be an integer from 1 to 100.")
    budgets = list(budgets)
    if not budgets or any(type(x) is not int or x < 0 or x > len(pool) for x in budgets) or budgets != sorted(set(budgets)):
        raise ValueError("Budgets must be unique ascending integers between 0 and pool size.")
    # One vectorizer fit to seed text only per round would change vocabulary as labels arrive.
    # This simulation deliberately refits the full pipeline after each acquired label.
    uncertainty_order, current, remaining = [], list(seed), list(pool)
    while remaining:
        chosen = train_and_queue(current, [{"id": r["id"], "text": r["text"]} for r in remaining], 1)["queue"][0]["id"]
        item = next(r for r in remaining if r["id"] == chosen)
        uncertainty_order.append(item); current.append(item); remaining = [r for r in remaining if r["id"] != chosen]
    uncertainty = []
    for budget in budgets:
        uncertainty.append({"budget": budget, **_score(seed + uncertainty_order[:budget], test)})
    rng = np.random.default_rng(random_seed); random_runs = []
    for repeat in range(random_repeats):
        order = rng.permutation(len(pool))
        random_runs.append([{"budget": budget, **_score(seed + [pool[i] for i in order[:budget]], test)} for budget in budgets])
    random_curve = []
    for j, budget in enumerate(budgets):
        for metric in ["accuracy", "macro_f1"]:
            values = np.array([run[j][metric] for run in random_runs])
            if not np.isfinite(values).all(): raise RuntimeError("Non-finite benchmark metric.")
            target = next((x for x in random_curve if x["budget"] == budget), None)
            if target is None: target = {"budget": budget}; random_curve.append(target)
            target[metric + "_mean"] = float(values.mean())
            target[metric + "_std"] = float(values.std(ddof=0))
    payload = {"schema_version": 1, "classes": labels, "seed_count": len(seed), "pool_count": len(pool),
               "test_count": len(test), "budgets": budgets, "random_repeats": random_repeats,
               "random_seed": random_seed, "uncertainty": uncertainty, "random": random_curve,
               "selection": "least confidence", "classifier": "class-balanced logistic regression",
               "features": "word unigram/bigram TF-IDF, max 20,000 features; refit after each acquisition",
               "protocol": "Pool labels hidden during selection and revealed only after acquisition. Held-out test labels never used for selection or training. Random uses identical budgets.",
               "warnings": ["Synthetic fixture results do not establish performance on real annotation tasks.",
                            "Learning curves are test-set measurements for demonstration; repeatedly consulting them during design would leak evaluation information."]}
    payload["dataset_sha256"] = hashlib.sha256(json.dumps({"seed": seed, "pool": pool, "test": test}, sort_keys=True).encode()).hexdigest()
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload
