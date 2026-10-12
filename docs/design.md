# LabelWise evaluation contract

Three disjoint ID splits: labelled seed, annotation pool, held-out test. Queue input has no label column. Benchmark pool has labels that are hidden from the selector and revealed only after acquisition. Test labels are used only for scoring. Word unigram/bigram TF-IDF and class-balanced logistic regression are refit after each acquisition. Least-confidence sampling is compared with 20 seeded random permutations at identical budgets. Macro F1 and accuracy are point estimates; random reports population standard deviation, not a confidence interval. No hyperparameter selection from test curves.

CSV parsing is UTF-8, bounded, strict and deterministic. IDs do not cross splits. Every initial class has at least two seed examples. No automatic labelling: predicted class/confidence are prioritisation evidence only. Browser state is ephemeral; annotation happens outside the app by adding reviewed rows to the seed CSV and rerunning.
