# Compact 3D Domain-Balanced Protocol v2

Frozen on August 14, 2026 before training.

## Hypothesis

Compact-3D v1 improves pooled CV-C discrimination but is inconsistent across
exact acquisition signatures. The architecture is held fixed while the
training policy directly targets scanner-signature transfer.

## Fixed Controls

- Same `3 x 48 x 48 x 48` cached inputs as compact-3D v1.
- Same immutable CV-C folds and no signature crossing folds.
- Same compact hemisphere encoder and head.
- Same AdamW optimizer, learning rate, label smoothing, batch size, TTA, and
  predeclared blend weights (`0.10`, `0.25`, `0.40`).
- Two controlled seeds: `73` and `211`.

## V2 Training Changes

1. Sampling weight is proportional to acquisition-signature count raised to
   `-0.25`, multiplied by a global class-frequency correction raised to
   `-0.5`. This increases representation of uncommon signatures without making
   singleton signatures dominate training.
2. Scanner-style augmentation uses native spacing from the training row. Fine
   acquisitions are more likely to be downsampled and restored in 3D; coarse
   acquisitions receive milder degradation. Blur, noise, gamma, intensity,
   localization shifts, and left-right flips remain label preserving.
3. Checkpoints are selected with a fixed validation objective:

   `0.5 * pooled_log_loss + 0.5 * macro_large_signature_log_loss`

   A large signature has at least ten scans, matching the frozen promotion
   definition. No public, smoke, historical CV-A, or historical CV-B result is
   used for checkpoint selection.

## Promotion

Each seed and the predeclared 50/50 probability average are evaluated with the
unchanged gates in `validation/domain_holdout_v2/PROMOTION_GATES.md`. A result
that misses any gate is not packaged or submitted.

## Historical Confirmation

After seed 73 passed all CV-C gates, the `0.25` compact-3D blend was fixed as
the conservative candidate before training historical split models. The same
v2 model is trained on the already-defined CV-A and CV-B folds and blended
with the matching robust hemisphere seed-73 OOF predictions.

The candidate must improve log loss on both CV-A and CV-B, must not regress
AUROC by more than `0.002`, and must show at least `97.5%` paired-bootstrap
support for improvement on each split. The `0.40` blend remains diagnostic and
cannot replace the fixed conservative candidate based on these results.
