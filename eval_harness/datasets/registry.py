from __future__ import annotations

import hashlib
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task_type: str
    num_classes: int
    input_shape: tuple[int, ...]
    class_names: tuple[str, ...] = ()


class DatasetNotReadyError(FileNotFoundError):
    """Raised when a real dataset loader is requested before data is present."""


class SyntheticGeoDataset:
    """Small deterministic fixture that mirrors each dataset's tensor contract."""

    def __init__(self, spec: DatasetSpec, split: str, size: int = 64, seed: int = 1234):
        self.spec = spec
        self.split = split
        self.size = size
        self.seed = stable_seed(f"{spec.name}:{split}:{seed}")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(stable_seed(f"{self.seed}:{idx}"))
        if self.spec.task_type == "classification":
            y = np.int64(idx % self.spec.num_classes)
            x = self._classification_features(rng, int(y))
        elif self.spec.task_type == "segmentation":
            x, y = self._segmentation_patch(rng)
        else:
            raise ValueError(f"Unsupported task type: {self.spec.task_type}")
        return {"id": f"{self.spec.name}-{self.split}-{idx}", "x": x, "y": y}

    def _classification_features(self, rng: np.random.Generator, label: int) -> np.ndarray:
        time_steps, features = self.spec.input_shape
        t = np.linspace(0, 2 * np.pi, time_steps, dtype="float32")
        phase = label * np.pi / max(1, self.spec.num_classes)
        amplitude = 0.7 + 0.2 * label
        seasonal = amplitude * np.sin(t + phase)
        spectral_offsets = np.linspace(-0.4, 0.4, features, dtype="float32")
        x = seasonal[:, None] + spectral_offsets[None, :]
        x += rng.normal(scale=0.25, size=self.spec.input_shape).astype("float32")
        return x.astype("float32")

    def _segmentation_patch(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        channels, height, width = self.spec.input_shape
        x = rng.normal(scale=0.15, size=self.spec.input_shape).astype("float32")
        y = np.zeros((height, width), dtype="int64")
        n_fields = int(rng.integers(2, 6))
        for _ in range(n_fields):
            h = int(rng.integers(max(6, height // 8), max(7, height // 3)))
            w = int(rng.integers(max(6, width // 8), max(7, width // 3)))
            top = int(rng.integers(0, max(1, height - h)))
            left = int(rng.integers(0, max(1, width - w)))
            y[top : top + h, left : left + w] = 1
        for channel in range(channels):
            background = 0.15 * channel
            foreground = 0.45 + 0.08 * channel
            x[channel] += np.where(y == 1, foreground, background).astype("float32")
        return x, y


class ManifestArrayClassificationDataset:
    """Real field-level classification loader driven by a manifest file."""

    def __init__(self, dataset_cfg: dict, split: str):
        self.cfg = dataset_cfg
        self.root = Path(dataset_cfg["data"]["root"])
        manifest_path = _resolve_path(self.root, dataset_cfg["data"]["manifest"])
        _require_file(manifest_path, dataset_cfg["name"])
        self.manifest_path = manifest_path
        self.split = split
        self.id_column = dataset_cfg["data"].get("id_column", "id")
        self.array_column = dataset_cfg["data"].get("array_column", "array_path")
        self.label_column = dataset_cfg["data"].get("label_column", "label")
        self.split_column = dataset_cfg["data"].get("split_column", "split")
        self.class_names = list(dataset_cfg.get("class_names", []))
        manifest = pd.read_csv(manifest_path)
        _require_columns(
            manifest,
            [self.id_column, self.split_column, self.array_column, self.label_column],
            manifest_path,
        )
        self.df = manifest[manifest[self.split_column].astype(str) == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows for split={split!r} in {manifest_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        x = load_array(_resolve_path(self.root, row[self.array_column])).astype("float32")
        y = self._label_to_int(row[self.label_column])
        return {"id": str(row[self.id_column]), "x": x, "y": np.int64(y)}

    def _label_to_int(self, label: object) -> int:
        if isinstance(label, (int, np.integer)) or str(label).isdigit():
            return int(label)
        if str(label) not in self.class_names:
            raise ValueError(f"Unknown class label {label!r}; expected one of {self.class_names}")
        return self.class_names.index(str(label))


class ManifestSegmentationDataset:
    """Real segmentation loader driven by image/mask paths in a manifest file."""

    def __init__(self, dataset_cfg: dict, split: str):
        self.cfg = dataset_cfg
        self.root = Path(dataset_cfg["data"]["root"])
        manifest_path = _resolve_path(self.root, dataset_cfg["data"]["manifest"])
        _require_file(manifest_path, dataset_cfg["name"])
        self.manifest_path = manifest_path
        self.split = split
        self.id_column = dataset_cfg["data"].get("id_column", "id")
        self.image_column = dataset_cfg["data"].get("image_column", "image_path")
        self.mask_column = dataset_cfg["data"].get("mask_column", "mask_path")
        self.split_column = dataset_cfg["data"].get("split_column", "split")
        manifest = pd.read_csv(manifest_path)
        _require_columns(
            manifest,
            [self.id_column, self.split_column, self.image_column, self.mask_column],
            manifest_path,
        )
        filter_column = dataset_cfg["data"].get("africa_filter_column")
        filter_value = dataset_cfg["data"].get("africa_filter_value")
        if filter_column and filter_value and filter_column in manifest.columns:
            manifest = manifest[manifest[filter_column].astype(str) == str(filter_value)]
        self.df = manifest[manifest[self.split_column].astype(str) == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows for split={split!r} in {manifest_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        x = load_array(_resolve_path(self.root, row[self.image_column])).astype("float32")
        y = load_array(_resolve_path(self.root, row[self.mask_column])).astype("int64")
        return {"id": str(row[self.id_column]), "x": x, "y": y}


class HFSen1Floods11Dataset:
    """Ghana subset of Sen1Floods11 loaded directly from Hugging Face."""

    def __init__(self, dataset_cfg: dict, split: str):
        self.cfg = dataset_cfg
        self.repo_id = dataset_cfg["data"]["hf_repo"]
        self.repo_type = dataset_cfg["data"].get("hf_repo_type", "dataset")
        self.country = dataset_cfg["data"].get("country", "Ghana")
        self.image_group = dataset_cfg["data"].get("image_group", "S2Hand")
        self.label_group = dataset_cfg["data"].get("label_group", "LabelHand")
        self.cache_dir = dataset_cfg["data"].get("cache_dir")
        max_items = dataset_cfg["data"].get("max_items")
        self.split = split
        api = HfApi()
        files = api.list_repo_files(repo_id=self.repo_id, repo_type=self.repo_type)
        labels = sorted(
            file
            for file in files
            if f"/{self.label_group}/" in file and f"/{self.country}_" in file and file.endswith(".tif")
        )
        pairs = []
        for label_path in labels:
            stem = Path(label_path).name.replace("_LabelHand.tif", "")
            image_path = label_path.replace(f"/{self.label_group}/", f"/{self.image_group}/").replace(
                "_LabelHand.tif", "_S2Hand.tif"
            )
            if image_path in files:
                pairs.append({"id": stem, "image_path": image_path, "mask_path": label_path})
        if max_items:
            pairs = pairs[: int(max_items)]
        self.items = _split_items(pairs, dataset_cfg.get("split_fractions", {}), split)
        if not self.items:
            raise ValueError(f"No {self.country} Sen1Floods11 items for split={split!r}.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        image_path = hf_hub_download(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            filename=item["image_path"],
            cache_dir=self.cache_dir,
        )
        mask_path = hf_hub_download(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            filename=item["mask_path"],
            cache_dir=self.cache_dir,
        )
        x = load_array(Path(image_path)).astype("float32")
        y = load_array(Path(mask_path)).astype("int64")
        if y.ndim == 3:
            y = y[0]
        y = (y > 0).astype("int64")
        return {"id": item["id"], "x": x, "y": y}


class HFTarNpyDataset:
    """Load unlabeled .npy arrays from a Hugging Face-hosted tar file."""

    def __init__(self, dataset_cfg: dict, split: str):
        self.cfg = dataset_cfg
        self.repo_id = dataset_cfg["data"]["hf_repo"]
        self.repo_type = dataset_cfg["data"].get("hf_repo_type", "dataset")
        self.cache_dir = dataset_cfg["data"].get("cache_dir")
        self.split = split
        self.tar_files = dataset_cfg["data"]["tar_files"]
        max_items = dataset_cfg["data"].get("max_items")
        all_items = []
        for tar_file in self.tar_files:
            tar_path = hf_hub_download(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                filename=tar_file,
                cache_dir=self.cache_dir,
            )
            with tarfile.open(tar_path) as tar:
                for member in tar.getmembers():
                    if member.isfile() and member.name.endswith(".npy"):
                        all_items.append({"id": Path(member.name).stem, "tar_path": tar_path, "member": member.name})
        all_items = sorted(all_items, key=lambda item: item["id"])
        if max_items:
            all_items = all_items[: int(max_items)]
        self.items = _split_items(all_items, dataset_cfg.get("split_fractions", {}), split)
        if not self.items:
            raise ValueError(f"No tar npy items for split={split!r}.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        with tarfile.open(item["tar_path"]) as tar:
            payload = tar.extractfile(item["member"])
            if payload is None:
                raise FileNotFoundError(item["member"])
            x = np.load(io.BytesIO(payload.read())).astype("float32")
        return {"id": item["id"], "x": x, "y": np.int64(0)}


class HFFTWPlanetDataset:
    """Load FTW-Planet country shards from Hugging Face without extracting tars."""

    IMAGE_SUFFIXES = {
        "window_a": ".window_a.tif",
        "window_b": ".window_b.tif",
        "label": ".label.tif",
        "metadata": ".json",
    }

    def __init__(self, dataset_cfg: dict, split: str):
        self.cfg = dataset_cfg
        self.repo_id = dataset_cfg["data"]["hf_repo"]
        self.repo_type = dataset_cfg["data"].get("hf_repo_type", "dataset")
        self.cache_dir = dataset_cfg["data"].get("cache_dir")
        self.countries = dataset_cfg["data"].get("countries", [])
        self.windows = dataset_cfg["data"].get("windows", ["window_a", "window_b"])
        self.output_size = tuple(dataset_cfg["data"].get("output_size", [512, 512]))
        self.normalize_scale = float(dataset_cfg["data"].get("normalize_scale", 10000.0))
        max_items_per_country = dataset_cfg["data"].get("max_items_per_country")
        self.split = split

        all_items = []
        for country in self.countries:
            tar_file = f"dataset/{country}.tar"
            tar_path = hf_hub_download(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                filename=tar_file,
                cache_dir=self.cache_dir,
            )
            country_items = self._scan_tar(Path(tar_path), country)
            if max_items_per_country:
                country_items = country_items[: int(max_items_per_country)]
            all_items.extend(country_items)
        all_items = sorted(all_items, key=lambda item: (item["country"], item["id"]))
        split_seed = stable_seed(f"{dataset_cfg['name']}:{dataset_cfg.get('data', {}).get('split_seed', 1234)}")
        rng = np.random.default_rng(split_seed)
        all_items = [all_items[idx] for idx in rng.permutation(len(all_items))]
        self.items = _split_items(all_items, dataset_cfg.get("split_fractions", {}), split)
        if not self.items:
            raise ValueError(f"No FTW-Planet items for split={split!r}.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        with tarfile.open(item["tar_path"]) as tar:
            windows = []
            for window in self.windows:
                x = _read_tif_from_tar(tar, item["members"][window]).astype("float32")
                x = _center_crop_or_pad_chw(x, self.output_size) / self.normalize_scale
                windows.append(x)
            y = _read_tif_from_tar(tar, item["members"]["label"]).astype("int64")
        if y.ndim == 3:
            y = y[0]
        y = _center_crop_or_pad_hw(y, self.output_size).astype("int64")
        x = np.concatenate(windows, axis=0) if len(windows) > 1 else windows[0]
        return {
            "id": f"{item['country']}/{item['id']}",
            "x": x,
            "y": y,
            "country": item["country"],
        }

    def _scan_tar(self, tar_path: Path, country: str) -> list[dict]:
        grouped: dict[str, dict] = {}
        with tarfile.open(tar_path) as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                name = Path(member.name).name
                for key, suffix in self.IMAGE_SUFFIXES.items():
                    if name.endswith(suffix):
                        patch_id = name[: -len(suffix)]
                        grouped.setdefault(patch_id, {})[key] = member.name
                        break
        required = set(self.windows) | {"label"}
        items = []
        for patch_id, members in grouped.items():
            if required.issubset(members):
                items.append(
                    {
                        "id": patch_id,
                        "country": country,
                        "tar_path": tar_path,
                        "members": members,
                    }
                )
        return sorted(items, key=lambda item: item["id"])


def stable_seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big")


def make_label_budget_indices(n: int, budget: float, seed: int) -> list[int]:
    if not 0 < budget <= 1:
        raise ValueError("budget must be in the interval (0, 1].")
    rng = np.random.default_rng(seed)
    k = max(1, int(round(n * budget)))
    return sorted(rng.choice(n, size=k, replace=False).tolist())


def make_stratified_label_budget_indices(labels: list[int], budget: float, seed: int) -> list[int]:
    if not 0 < budget <= 1:
        raise ValueError("budget must be in the interval (0, 1].")
    rng = np.random.default_rng(seed)
    labels_array = np.asarray(labels)
    classes = np.unique(labels_array)
    target_n = max(len(classes), int(round(len(labels_array) * budget)))
    selected: list[int] = []
    for label in classes:
        candidates = np.where(labels_array == label)[0]
        selected.append(int(rng.choice(candidates)))
    remaining_n = max(0, target_n - len(selected))
    if remaining_n:
        remaining = np.asarray(sorted(set(range(len(labels_array))) - set(selected)))
        selected.extend(rng.choice(remaining, size=remaining_n, replace=False).astype(int).tolist())
    return sorted(selected)


def build_synthetic_dataset(dataset_cfg: dict, split: str, seed: int = 1234, smoke_size: int = 64) -> SyntheticGeoDataset:
    shape = tuple(dataset_cfg["input"]["shape"])
    spec = DatasetSpec(
        name=dataset_cfg["name"],
        task_type=dataset_cfg["task_type"],
        num_classes=int(dataset_cfg["num_classes"]),
        input_shape=shape,
        class_names=tuple(dataset_cfg.get("class_names", [])),
    )
    return SyntheticGeoDataset(spec=spec, split=split, size=smoke_size, seed=seed)


def build_dataset(
    dataset_cfg: dict,
    split: str,
    seed: int = 1234,
    smoke_size: int = 64,
    allow_synthetic: bool | None = None,
):
    loader = dataset_cfg.get("data", {}).get("loader", "synthetic")
    if loader == "manifest_array":
        return ManifestArrayClassificationDataset(dataset_cfg, split)
    if loader == "manifest_segmentation":
        return ManifestSegmentationDataset(dataset_cfg, split)
    if loader == "hf_sen1floods11":
        return HFSen1Floods11Dataset(dataset_cfg, split)
    if loader == "hf_tar_npy":
        return HFTarNpyDataset(dataset_cfg, split)
    if loader == "hf_ftw_planet":
        return HFFTWPlanetDataset(dataset_cfg, split)
    if loader == "synthetic":
        return build_synthetic_dataset(dataset_cfg, split, seed=seed, smoke_size=smoke_size)
    if allow_synthetic or dataset_cfg.get("data", {}).get("allow_synthetic_fallback", False):
        return build_synthetic_dataset(dataset_cfg, split, seed=seed, smoke_size=smoke_size)
    raise ValueError(f"Unsupported dataset loader {loader!r} for dataset {dataset_cfg['name']}")


def load_array(path: Path) -> np.ndarray:
    _require_file(path, "array")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        data = np.load(path)
        if "x" in data:
            return data["x"]
        if "arr_0" in data:
            return data["arr_0"]
        first_key = next(iter(data.files))
        return data[first_key]
    if suffix == ".csv":
        return pd.read_csv(path).to_numpy()
    if suffix == ".pt":
        tensor = torch.load(path, map_location="cpu")
        if isinstance(tensor, dict):
            for key in ("x", "image", "mask", "array"):
                if key in tensor:
                    tensor = tensor[key]
                    break
        return tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)
    if suffix in {".tif", ".tiff"}:
        try:
            import rasterio
        except ImportError:
            from PIL import Image

            with Image.open(path) as img:
                arr = np.asarray(img)
            if arr.ndim == 2:
                return arr
            if arr.ndim == 3:
                return np.moveaxis(arr, -1, 0)
            return arr
        with rasterio.open(path) as src:
            return src.read()
    raise ValueError(f"Unsupported array file type: {path}")


def _read_tif_from_tar(tar: tarfile.TarFile, member_name: str) -> np.ndarray:
    payload = tar.extractfile(member_name)
    if payload is None:
        raise FileNotFoundError(member_name)
    data = payload.read()
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError("FTW-Planet GeoTIFF loading requires rasterio. Run `uv sync --extra geo`.") from exc
    with rasterio.open(io.BytesIO(data)) as src:
        return src.read()


def _center_crop_or_pad_chw(x: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected [channels, height, width], got shape {x.shape}")
    channels, _, _ = x.shape
    y = np.zeros((channels, output_size[0], output_size[1]), dtype=x.dtype)
    src_h, src_w, dst_h, dst_w = _center_slices(x.shape[-2:], output_size)
    y[:, dst_h, dst_w] = x[:, src_h, src_w]
    return y


def _center_crop_or_pad_hw(x: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected [height, width], got shape {x.shape}")
    y = np.zeros(output_size, dtype=x.dtype)
    src_h, src_w, dst_h, dst_w = _center_slices(x.shape[-2:], output_size)
    y[dst_h, dst_w] = x[src_h, src_w]
    return y


def _center_slices(input_size: tuple[int, int], output_size: tuple[int, int]):
    in_h, in_w = input_size
    out_h, out_w = output_size
    copy_h = min(in_h, out_h)
    copy_w = min(in_w, out_w)
    src_top = max(0, (in_h - copy_h) // 2)
    src_left = max(0, (in_w - copy_w) // 2)
    dst_top = max(0, (out_h - copy_h) // 2)
    dst_left = max(0, (out_w - copy_w) // 2)
    return (
        slice(src_top, src_top + copy_h),
        slice(src_left, src_left + copy_w),
        slice(dst_top, dst_top + copy_h),
        slice(dst_left, dst_left + copy_w),
    )


def _resolve_path(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _require_file(path: Path, dataset_name: str) -> None:
    if not path.exists():
        raise DatasetNotReadyError(
            f"Real dataset file is missing for {dataset_name}: {path}. "
            "Create the manifest/data files described in docs/data_layout.md, or explicitly use build_synthetic_dataset "
            "for toy-only plumbing checks."
        )


def _require_columns(df: pd.DataFrame, columns: list[str], manifest_path: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Manifest {manifest_path} is missing required columns: {missing}")


def _split_items(items: list[dict], fractions: dict, split: str) -> list[dict]:
    train_frac = float(fractions.get("train", 0.7))
    val_frac = float(fractions.get("val", 0.15))
    n = len(items)
    train_end = max(1, int(round(n * train_frac)))
    val_end = max(train_end + 1, int(round(n * (train_frac + val_frac)))) if n > 2 else n
    if split == "train":
        return items[:train_end]
    if split == "val":
        return items[train_end:val_end]
    if split == "test":
        return items[val_end:]
    return [item for item in items if item.get("split") == split]
