from __future__ import annotations

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import ndimage


DATA_ROOT = Path(os.environ.get("COMPETITION_DATA_ROOT", "/code_execution/data"))
NIFTI_ROOT = DATA_ROOT / "niftis"
FORMAT_PATH = DATA_ROOT / "submission_format.csv"
OUTPUT_PATH = Path(os.environ.get("SUBMISSION_OUTPUT", "submission.csv"))
ASSET_ROOT = Path(__file__).parent / "assets"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMPACT_WEIGHT = 0.25


class ResidualBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        )
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(value) + self.skip(value))


class HemisphereEncoder2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = 32
        self.encoder = nn.Sequential(
            nn.Conv2d(9, channels, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            ResidualBlock2d(channels, channels),
            ResidualBlock2d(channels, channels * 2, 2),
            ResidualBlock2d(channels * 2, channels * 4, 2),
            ResidualBlock2d(channels * 4, channels * 8, 2),
            nn.AdaptiveAvgPool2d((1, 2)),
            nn.Flatten(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.encoder(value)


class HemisphereNet2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.side_encoder = HemisphereEncoder2d()
        self.head = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, 256),
            nn.SiLU(),
            nn.Dropout(0.35),
            nn.Linear(256, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        midpoint = value.shape[2] // 2
        left = value[:, :, :midpoint, :]
        right = torch.flip(value[:, :, -midpoint:, :], dims=(2,))
        left_feature = self.side_encoder(left)
        right_feature = self.side_encoder(right)
        fused = torch.cat(
            [
                0.5 * (left_feature + right_feature),
                torch.abs(left_feature - right_feature),
                torch.minimum(left_feature, right_feature),
                torch.maximum(left_feature, right_feature),
            ],
            dim=1,
        )
        return self.head(fused).squeeze(1)


class ResidualBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.body = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels, 3, stride=stride, padding=1, bias=False
            ),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False)
        )
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(value) + self.skip(value))


class SideEncoder3d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = 12
        self.encoder = nn.Sequential(
            nn.Conv3d(3, channels, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(6, channels),
            nn.SiLU(),
            ResidualBlock3d(channels, channels),
            ResidualBlock3d(channels, channels * 2, 2),
            ResidualBlock3d(channels * 2, channels * 4, 2),
            ResidualBlock3d(channels * 4, channels * 8, 2),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.encoder(value)


class CompactHemisphere3d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.side_encoder = SideEncoder3d()
        self.head = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 128),
            nn.SiLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        midpoint = value.shape[2] // 2
        left = value[:, :, :midpoint]
        right = torch.flip(value[:, :, -midpoint:], dims=(2,))
        left_feature = self.side_encoder(left)
        right_feature = self.side_encoder(right)
        fused = torch.cat(
            [
                0.5 * (left_feature + right_feature),
                torch.abs(left_feature - right_feature),
                torch.minimum(left_feature, right_feature),
                torch.maximum(left_feature, right_feature),
            ],
            dim=1,
        )
        return self.head(fused).squeeze(1)


def crop_cube(array: np.ndarray, center: np.ndarray, size: int) -> np.ndarray:
    start = np.rint(center).astype(int) - size // 2
    output = np.zeros((size, size, size), dtype=np.float32)
    source_slices = []
    target_slices = []
    for axis in range(3):
        low = max(0, int(start[axis]))
        high = min(array.shape[axis], int(start[axis]) + size)
        source_slices.append(slice(low, high))
        target_start = low - int(start[axis])
        target_slices.append(slice(target_start, target_start + high - low))
    output[tuple(target_slices)] = array[tuple(source_slices)]
    return output


def weighted_center_grid(array: np.ndarray) -> np.ndarray:
    positive = array[array > 0]
    threshold = float(np.quantile(positive, 0.90)) if positive.size else 0.0
    weights = np.clip(array - threshold, 0.0, None)
    if float(weights.sum()) <= 1e-8:
        weights = array
    grids = np.indices(array.shape, dtype=np.float32)
    denominator = max(float(weights.sum()), 1e-8)
    center = np.asarray(
        [(grids[axis] * weights).sum() / denominator for axis in range(3)],
        dtype=np.float32,
    )
    if not np.isfinite(center).all():
        center = (np.asarray(array.shape, dtype=np.float32) - 1.0) / 2.0
    return center


def weighted_center_scipy(array: np.ndarray) -> np.ndarray:
    positive = array[array > 0]
    threshold = float(np.quantile(positive, 0.90)) if positive.size else 0.0
    weights = np.clip(array - threshold, 0.0, None)
    if float(weights.sum()) <= 1e-8:
        weights = array
    center = np.asarray(ndimage.center_of_mass(weights), dtype=np.float32)
    if not np.isfinite(center).all():
        center = (np.asarray(array.shape, dtype=np.float32) - 1.0) / 2.0
    return center


def linear_p999(crop: np.ndarray) -> np.ndarray:
    positive = crop[crop > 0]
    scale = float(np.quantile(positive, 0.999)) if positive.size else 1.0
    return np.clip(crop / max(scale, 1e-6), 0.0, 1.0).astype(np.float32)


def preprocess(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = nib.as_closest_canonical(nib.load(image_path))
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=np.float32)
    array = np.asarray(image.get_fdata(dtype=np.float32))
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = np.clip(array, 0.0, None)
    zoom = spacing / 1.8
    if not np.allclose(zoom, 1.0, atol=0.02):
        array = ndimage.zoom(
            array, zoom=zoom, order=1, mode="constant", prefilter=False
        )

    crop2d = crop_cube(array, weighted_center_grid(array), 96)
    linear2d = linear_p999(crop2d)
    log2d = np.log1p(9.0 * linear2d) / np.log(10.0)
    asymmetry2d = np.abs(linear2d - np.flip(linear2d, axis=0))
    stacked2d = np.stack([linear2d, log2d, asymmetry2d], axis=0)
    projection = np.concatenate(
        [
            np.max(stacked2d, axis=-1),
            np.mean(stacked2d, axis=-1),
            np.mean(stacked2d[..., 32:64], axis=-1),
        ],
        axis=0,
    ).astype(np.float16).astype(np.float32)

    crop3d = crop_cube(array, weighted_center_scipy(array), 72)
    linear3d = linear_p999(crop3d)
    linear3d = ndimage.zoom(
        linear3d,
        zoom=(48 / 72, 48 / 72, 48 / 72),
        order=1,
        mode="constant",
        prefilter=False,
    ).astype(np.float32)
    log3d = np.log1p(9.0 * linear3d) / np.log(10.0)
    asymmetry3d = np.abs(linear3d - np.flip(linear3d, axis=0))
    volume = (
        np.stack([linear3d, log3d, asymmetry3d], axis=0)
        .astype(np.float16)
        .astype(np.float32)
    )

    if projection.shape != (9, 96, 96) or volume.shape != (3, 48, 48, 48):
        raise RuntimeError(
            f"Invalid preprocessed shapes: projection={projection.shape}, volume={volume.shape}"
        )
    if not np.isfinite(projection).all() or not np.isfinite(volume).all():
        raise RuntimeError("Non-finite preprocessed tensor")
    return projection, volume


def load_models() -> tuple[list[nn.Module], list[nn.Module]]:
    models2d = []
    models3d = []
    for fold in range(5):
        model2d = HemisphereNet2d().to(DEVICE)
        model2d.load_state_dict(
            torch.load(
                ASSET_ROOT / "hemisphere2d" / f"fold{fold}.pt",
                map_location=DEVICE,
                weights_only=True,
            )
        )
        model2d.eval()
        models2d.append(model2d)

        model3d = CompactHemisphere3d().to(DEVICE)
        model3d.load_state_dict(
            torch.load(
                ASSET_ROOT / "compact3d" / f"fold{fold}.pt",
                map_location=DEVICE,
                weights_only=True,
            )
        )
        model3d.eval()
        models3d.append(model3d)
    return models2d, models3d


def predict_branch(models: list[nn.Module], tensor: torch.Tensor) -> float:
    probabilities = []
    with torch.inference_mode():
        for model in models:
            logits = 0.5 * (model(tensor) + model(torch.flip(tensor, dims=(2,))))
            probabilities.append(float(torch.sigmoid(logits).float().item()))
    return float(np.mean(probabilities))


def main() -> None:
    torch.set_float32_matmul_precision("high")
    submission = pd.read_csv(FORMAT_PATH)
    if list(submission.columns) != ["uid", "is_pathologic"]:
        raise RuntimeError("submission_format.csv has unexpected columns")
    if submission.uid.duplicated().any():
        raise RuntimeError("submission_format.csv contains duplicate UIDs")
    models2d, models3d = load_models()
    predictions = []
    for uid in submission.uid.astype(str):
        image_path = NIFTI_ROOT / f"{uid}.nii.gz"
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        projection, volume = preprocess(image_path)
        tensor2d = torch.from_numpy(projection[None]).to(DEVICE)
        tensor3d = torch.from_numpy(volume[None]).to(DEVICE)
        probability2d = predict_branch(models2d, tensor2d)
        probability3d = predict_branch(models3d, tensor3d)
        predictions.append(
            (1.0 - COMPACT_WEIGHT) * probability2d
            + COMPACT_WEIGHT * probability3d
        )
    submission["is_pathologic"] = np.clip(predictions, 1e-5, 1 - 1e-5)
    submission.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
