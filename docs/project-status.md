# LabelWise delivery record — 2026-10-12

Authorized repository: oshada28nilendra-debug/labelwise. Feature branch: `feat/labelwise-mvp`. No merge or deployment authorized.

Implemented: strict CSV splits, seed-trained TF-IDF/logistic classifier, least-confidence queue, retraining workflow, equal-budget uncertainty/random simulation, held-out macro-F1/accuracy curves, CLI, localhost browser explorer and JSON exports. Bundled support-ticket data is deterministic and synthetic.

Local verification: 18 Python tests passed in a clean Python 3.12 environment. Real Chromium 133 checks passed for benchmark plotting, JSON export, actual queue uploads, confidence evidence, malformed-input reset, 390px layout and JavaScript errors. The reproducible report measures macro F1 0.4773 at budget 0; at budget 10, uncertainty is 1.0000 and the 20-run random mean is 0.7983 ± 0.1104. Synthetic measurements are not evidence of real annotation savings or adoption. Remote CI status is reported on PR #1. See README for protocol and limitations.

Evaluation-leakage audit found exact duplicate synthetic texts across pool/test before publication. Fixture generation was changed to unique text and validation now rejects exact cross-split text overlap. All measurements were regenerated after the fix.
