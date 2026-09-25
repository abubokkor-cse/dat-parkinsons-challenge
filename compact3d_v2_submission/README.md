# Compact3D v2 Scored Artifact

This directory contains the exact archive submitted as DrivenData submission
`id-319205`, which scored `0.3309` public log loss and `0.9248` AUROC.

## Contents

- `submission_compact3d_v2_ready_20260814.zip`: exact executable archive.
- `main.py`: extracted root-level entrypoint for inspection.
- `manifest.json`: 75% hemisphere-2D and 25% compact-3D blend configuration.
- `readiness_manifest.json`: packaging, reproducibility, and offline-runtime
  validation record.
- `SHA256SUMS`: artifact integrity checksum.
- `compact3d_domain_balanced_v2_protocol.md`: training and predeclared
  promotion design.
- `public_results.md`: comparative result evidence and interpretation.

## Verify the archive

```bash
shasum -a 256 submission_compact3d_v2_ready_20260814.zip
unzip -t submission_compact3d_v2_ready_20260814.zip
```

The expected SHA-256 is
`e81f088318abc7814c5a2673814972a2ace39c75bd8fc7728cc2fd033144e7ce`.

## Data Governance & Compliance

This submission package contains reproducible inference scripts and trained model checkpoints. In compliance with DrivenData competition rules and medical research data privacy standards, raw competition imaging data, ground-truth patient labels, and individual predictions are strictly excluded.
