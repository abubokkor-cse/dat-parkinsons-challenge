# DaT Parkinson's Challenge: Compact3D v2

This repository preserves the best verified submission from this research
workspace for the DrivenData DaT Parkinson's Challenge.

| Public result | Value |
|---|---:|
| Log loss | `0.3309` |
| AUROC | `0.9248` |
| Submission | `id-319205` |
| Leaderboard rank at scoring | `#59` |

The result is a modest improvement over the earlier `0.3394 / 0.9184` anchor.
It is not a top-ranking solution and does not establish a claim about hidden
test performance beyond this recorded public score.

## Repository layout

```text
compact3d_v2_submission/
  submission_compact3d_v2_ready_20260814.zip  Exact scored submission archive
  main.py                                     Readable inference entrypoint
  manifest.json                               Model blend configuration
  readiness_manifest.json                     Reproducibility and runtime evidence
  SHA256SUMS                                  Integrity checksum
  compact3d_domain_balanced_v2_protocol.md    Fixed training and promotion protocol
  public_results.md                           Submission comparison and limitations
```

## Model

The submission makes an independent prediction for each NIfTI scan and blends:

- five hemisphere-aware 2D projection models, probability weight `0.75`;
- five compact 3D volume models, probability weight `0.25`.

No test-set aggregation, test-time adaptation, or network access is required.
The submission clips final probabilities to `[1e-5, 1 - 1e-5]` and applies no
post-hoc calibration.

## Reproducibility record

The included archive is exactly the scored artifact:

```text
SHA-256: e81f088318abc7814c5a2673814972a2ace39c75bd8fc7728cc2fd033144e7ce
Archive size: 46,552,267 bytes
Archive entries: 15
```

The artifact passed archive integrity checks, checkpoint provenance checks,
30-case OOF reproduction, and offline official-runtime schema checks. Its
20-case smoke-test score (`0.25793` log loss, `0.98901` AUROC) is recorded only
as an execution check; it was not predictive of the full public result.

## Data and publication policy

This repository intentionally excludes all competition NIfTI files, labels,
patient-level predictions, cached tensors, private configuration, service
credentials, and external audit data.

The repository is currently intended to remain private. Before making it
public, review the final competition rules and obtain organizer confirmation
that publishing the trained checkpoint archive is permitted. The source code
is MIT licensed; competition data and third-party software retain their own
terms.

## Local inference

The ZIP is designed for the official DrivenData code-execution runtime. It
expects:

```text
/code_execution/data/submission_format.csv
/code_execution/data/niftis/<uid>.nii.gz
```

At execution, `main.py` writes `submission.csv` in the runtime working
directory. The included archive contains all ten model checkpoints under
`assets/` and requires no internet connection.

## Scope

This repository is an archival record of one scored research artifact. It is
not a clinical diagnostic device and must not be used for patient care.
