# Submitted Domain-Robust vs Compact3D v2

Date: 2026-08-15

## Observed Public Result

`submission_compact3d_v2_ready_20260814.zip` completed successfully and scored:

- Public log loss: `0.3309`
- Public AUROC: `0.9248`
- Current rank after scoring: `#59`

Compared with the submitted domain-robust candidate (`0.3362 / 0.9244`), this
is an absolute log-loss improvement of `0.0053` and an AUROC improvement of
only `0.0004`. Compared with the original `0.3394 / 0.9184` anchor, the gains
are `0.0085` log loss and `0.0064` AUROC.

The experiment therefore succeeded as a modest robustness/confidence gain. It
did not produce the large discrimination improvement required for top three.

## Exact 20-Scan Docker Comparison

Both ZIPs were executed with the same offline runtime, images, labels, and
scoring code.

| Candidate | Log loss | AUROC | Brier | Accuracy |
|---|---:|---:|---:|---:|
| Submitted domain-robust | `0.22955409` | `0.96703297` | `0.07102831` | `0.90` |
| New compact3D v2 | `0.25793224` | `0.98901099` | `0.08055922` | `0.85` |

The old candidate is better on smoke log loss. The new candidate is better on
smoke AUROC. Neither result is a reliable estimate of the hidden public set:
there are only 13 positive and 7 negative smoke cases, so AUROC contains only
91 positive-negative comparisons. One reversed pair changes AUROC by about
`0.01099`.

## Historical Validation

| Protocol | Submitted domain-robust LL/AUROC | New compact3D v2 LL/AUROC |
|---|---:|---:|
| CV-A, 1,362 cases | `0.27114437 / 0.95313939` | `0.28926457 / 0.94761050` |
| CV-B, 1,362 cases | `0.28262856 / 0.94844636` | `0.28939355 / 0.94610638` |
| Public hidden | `0.3362 / 0.9244` | `0.3309 / 0.9248` |

The old ensemble is stronger on reused CV-A/CV-B. Those protocols substantially
overestimated its public transfer and CV-B had been adaptively reused after
many experiments, so they are no longer independent selection evidence.

## Fresh CV-C Evidence for Compact3D

The immutable CV-C split groups all 1,362 scans by exact acquisition signature
and prevents a signature from crossing folds. The compact3D weight was
predeclared before full CV-C evaluation.

| CV-C candidate | Log loss | AUROC |
|---|---:|---:|
| Robust hemisphere seed-73 baseline | `0.32584400` | `0.92989410` |
| Baseline + 25% compact3D v2 | `0.31094831` | `0.93694017` |

The blend gains `0.01489568` log loss and `0.00704607` AUROC, improves all five
folds, improves macro/rare/worst acquisition-signature loss, and has `99.99%`
paired-bootstrap support. It passes every frozen CV-C promotion gate.

Important limitation: the complete old 30-checkpoint ensemble was not retrained
on CV-C. The CV-C comparator is its strongest core robust-hemisphere branch,
not the exact submitted ensemble.

## Why Public AUROC Fell to 0.9244

The old candidate's Platt calibration has positive slope and is monotonic, so
it cannot change AUROC. The AUROC decline is a ranking failure:

- Smoke AUROC to public AUROC: `0.96703 -> 0.9244` (`-0.04263`).
- CV-B AUROC to public AUROC: `0.94845 -> 0.9244` (`-0.02405`).
- CV-B log loss to public log loss: `0.28263 -> 0.3362` (`+0.05357`).

The hidden centers contain scanner/reconstruction/borderline-case variation
not represented by the 20 smoke scans or the coarse CV-B families. The old
ensemble used many checkpoints, but they were closely related 2D projection
models, so their errors remained correlated under the unseen domain shift.

## Result Interpretation

The genuinely different 3D representation closed part of the log-loss transfer
gap, but almost none of the AUROC gap. The smoke-to-public log-loss gap remained
large at `+0.07297`, and smoke AUROC `0.98901` overstated public AUROC by
`0.06421`.

Because AUROC barely changed, the most likely effect is safer probability
magnitudes and correction of some high-loss confidence errors rather than a
materially better ordering of borderline hidden cases. The next candidate must
add scanner-invariant discrimination. More temperature/Platt calibration of
the same rankings cannot solve the remaining problem.
