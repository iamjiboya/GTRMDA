# Baseline integration

Baseline implementations are not vendored because they have separate licenses and
dependency stacks. Run NBFNet, ULTRA, PRODIGY, OFA, AnyBURL, ZeroStem, HGCLAMIR,
and HGTMDA from their official releases on the exact split exported by this
repository, then provide a CSV with:

```text
mirna,disease,label,score
```

Use `scripts/evaluate_baseline.py` to compute the same full-candidate binary,
calibration, and filtered ranking metrics. Supervised HGCLAMIR/HGTMDA results must
remain marked as target-relation upper bounds because they access MDA labels.

