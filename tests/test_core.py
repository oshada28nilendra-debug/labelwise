import numpy as np
import pytest
from labelwise.core import parse_csv, validate_splits, train_and_queue, benchmark


def csv(rows, labelled=True):
    header = "id,text,label\n" if labelled else "id,text\n"
    return (header + "\n".join(",".join(r) for r in rows)).encode()


def data():
    seed = [{"id":"s1","text":"refund payment card","label":"billing"},{"id":"s2","text":"invoice charged twice","label":"billing"},
            {"id":"s3","text":"password login account","label":"access"},{"id":"s4","text":"cannot sign in","label":"access"}]
    pool = [{"id":"p1","text":"payment refund missing","label":"billing"},{"id":"p2","text":"login secret rejected","label":"access"},
            {"id":"p3","text":"card charged again","label":"billing"},{"id":"p4","text":"account access denied","label":"access"}]
    test = [{"id":"t1","text":"need invoice refund","label":"billing"},{"id":"t2","text":"sign in password fails","label":"access"}]
    return seed,pool,test


def test_parse_and_validation():
    rows=parse_csv(csv([("1","hello world","x"),("2","good bye","y")]))
    assert rows[0]["id"]=="1"
    assert parse_csv(csv([("1","hello world")],False),False)[0]["text"]=="hello world"

@pytest.mark.parametrize("raw",[b"",b"id,text\n1,x",b"\xff",b"id,text,label\n1,,x",b"id,text,label\n1,x,",b"id,text,label\n1,x,a\n1,y,b",b"id,text,label\n1,x,a,extra"])
def test_bad_csv(raw):
    with pytest.raises(ValueError): parse_csv(raw)

def test_split_leakage_and_seed_coverage():
    seed,pool,test=data();validate_splits(seed,pool,test,True)
    with pytest.raises(ValueError,match="overlap"):validate_splits(seed,[dict(pool[0],id="s1")])
    with pytest.raises(ValueError,match="Exact text"):validate_splits(seed,[dict(pool[0],text=seed[0]["text"])])
    with pytest.raises(ValueError,match="two examples"):validate_splits(seed[:3],pool)
    with pytest.raises(ValueError,match="represented"):validate_splits(seed,pool,[{"id":"z","text":"x","label":"new"}],True)

def test_queue_is_deterministic_and_excludes_labels():
    seed,pool,_=data();unlabelled=[{"id":r["id"],"text":r["text"]} for r in pool]
    a=train_and_queue(seed,unlabelled,3);b=train_and_queue(seed,unlabelled,3)
    assert a["queue"]==b["queue"] and len(a["queue"])==3
    assert all("label" not in r and 0<=r["confidence"]<=1 for r in a["queue"])

def test_benchmark_equal_budgets_and_reproducible():
    seed,pool,test=data();a=benchmark(seed,pool,test,[0,2,4],5,7);b=benchmark(seed,pool,test,[0,2,4],5,7)
    assert a==b and [x["budget"] for x in a["uncertainty"]]==[0,2,4]
    assert [x["budget"] for x in a["random"]]==[0,2,4]
    assert all(0<=x["macro_f1"]<=1 for x in a["uncertainty"])

@pytest.mark.parametrize("kwargs",[{"budgets":[2,1]},{"budgets":[9]},{"random_repeats":0}])
def test_bad_benchmark_settings(kwargs):
    with pytest.raises(ValueError):benchmark(*data(),**kwargs)

def test_empty_vocabulary_actionable():
    seed,pool,_=data();bad=[dict(r,text="!!!") for r in seed]
    with pytest.raises(ValueError,match="vocabulary"):train_and_queue(bad,[{"id":"u","text":"word"}])
