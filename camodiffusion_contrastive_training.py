"""Training helpers for the no-Gated-OFM CamoDiffusion contrastive project.

This module deliberately does not import or clone the authors' GitHub source
repository. It trains CamoDiffusionModel from
diffusion_model.py using the project's own dataloader,
noise schedule, contrastive feature loss, and checkpoint utilities.
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from tqdm.auto import tqdm

from diffusion_model import CamoDiffusionModel, bce_iou_loss
from contrastive_loss import contrastive_feature_loss

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PRUNE_DIRS = {"__MACOSX", ".git", ".ipynb_checkpoints", "Imgs", "GT", "Image", "Images", "Edge", "Instance"}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _walk_dirs(root: str = "/kaggle/input", max_depth: int = 9):
    root = str(root)
    if not os.path.isdir(root):
        return
    for dirpath, dirnames, filenames in os.walk(root):
        depth = os.path.relpath(dirpath, root).count(os.sep)
        yield Path(dirpath), list(dirnames), filenames
        if depth >= max_depth:
            dirnames[:] = []
        else:
            dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]


def _index_files(folder: str | Path, exts: Optional[set[str]] = None) -> dict[str, Path]:
    folder = Path(folder)
    out: dict[str, Path] = {}
    if not folder.is_dir():
        return out
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in (exts or IMAGE_EXTS) and not p.name.startswith("."):
            
            out.setdefault(p.stem, p)
    return out


def pair_images_with_gt(image_dir: str | Path, gt_dir: str | Path,
                        image_exts: Optional[set[str]] = None,
                        gt_exts: Optional[set[str]] = None) -> list[tuple[Path, Path]]:
    imgs = _index_files(image_dir, image_exts)
    gts = _index_files(gt_dir, gt_exts or {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    return [(imgs[stem], gts[stem]) for stem in sorted(set(imgs) & set(gts))]


def _find_prepared_layout(search_root: str):
    train_root = None
    test_root = None
    for p, dirs, _ in _walk_dirs(search_root) or []:
        dset = set(dirs)
        if train_root is None and p.name.lower() in {"traindataset", "trainvaldataset"} and {"Imgs", "GT"}.issubset(dset):
            train_root = p
        if test_root is None and p.name.lower() == "testdataset" and any((p / n).is_dir() for n in ("CAMO", "COD10K", "NC4K")):
            test_root = p
    if train_root is None:
        return None
    train_pairs = pair_images_with_gt(train_root / "Imgs", train_root / "GT")
    val_pairs = []
    val_sets = {}
    if test_root is not None:
        for name in ("CAMO", "COD10K", "NC4K"):
            candidate = test_root / name
            if (candidate / "Imgs").is_dir() and (candidate / "GT").is_dir():
                pairs = pair_images_with_gt(candidate / "Imgs", candidate / "GT")
                if pairs:
                    val_sets[name] = pairs
                    val_pairs.extend(pairs)
    if train_pairs:
        return train_pairs, val_pairs, val_sets, f"prepared layout: {train_root}"
    return None


def _find_camo_root(search_root: str):
    for p, dirs, _ in _walk_dirs(search_root) or []:
        lower = {x.lower(): x for x in dirs}
        if "images" in lower and "gt" in lower:
            imdir, gtdir = p / lower["images"], p / lower["gt"]
            
            if any(x in str(p).lower() for x in ("camo", "camoufl")) or any("camourflage" in q.name.lower() for q in imdir.rglob("*" ) if q.is_file()):
                return p, imdir, gtdir
    return None


def _find_cod10k_root(search_root: str):
    for p, dirs, _ in _walk_dirs(search_root) or []:
        lower = {x.lower(): x for x in dirs}
        if p.name.lower().startswith("cod10k-v3") and "train" in lower and "test" in lower:
            train = p / lower["train"]
            test = p / lower["test"]
            def locate(split):
                subs = {x.lower(): x for x in split.iterdir() if x.is_dir()}
                image = next((subs[k] for k in ("image", "images", "imgs") if k in subs), None)
                gt = next((subs[k] for k in ("gt_object", "gt", "masks") if k in subs), None)
                return image, gt
            train_img, train_gt = locate(train)
            test_img, test_gt = locate(test)
            if train_img and train_gt and test_img and test_gt:
                return p, (train_img, train_gt), (test_img, test_gt)
        
        if "train" in lower and "test" in lower:
            try:
                train = p / lower["train"]
                test = p / lower["test"]
                if (train / "Image").is_dir() and (train / "GT_Object").is_dir() and (test / "Image").is_dir() and (test / "GT_Object").is_dir():
                    return p, (train / "Image", train / "GT_Object"), (test / "Image", test / "GT_Object")
            except Exception:
                pass
    return None


def lower_key(mapping: dict[str, str], key: str) -> str:
    return mapping[key]


def discover_dataset_pairs(search_root: str = "/kaggle/input",
                           manual_train_image_dir: Optional[str] = None,
                           manual_train_gt_dir: Optional[str] = None,
                           manual_val_image_dir: Optional[str] = None,
                           manual_val_gt_dir: Optional[str] = None):
    """Return train pairs, validation pairs, and discovered dataset metadata.

    Preferred layout is TrainDataset/Imgs+GT with TestDataset/{CAMO,COD10K,NC4K}/Imgs+GT.
    Otherwise it combines CAMO-train and COD10K-train, validating on their test sets.
    Four explicit manual dirs can override automatic discovery.
    """
    if manual_train_image_dir and manual_train_gt_dir:
        train_pairs = pair_images_with_gt(manual_train_image_dir, manual_train_gt_dir)
        val_pairs = pair_images_with_gt(manual_val_image_dir, manual_val_gt_dir) if manual_val_image_dir and manual_val_gt_dir else []
        if not train_pairs:
            raise RuntimeError("The manually specified training image/mask directories have no matching filename stems.")
        return train_pairs, val_pairs, {"manual_train": len(train_pairs), "manual_val": len(val_pairs)}

    prepared = _find_prepared_layout(search_root)
    if prepared is not None:
        train_pairs, val_pairs, val_sets, source = prepared
        return train_pairs, val_pairs, {"source": source, "train": len(train_pairs), "validation_sets": {k: len(v) for k, v in val_sets.items()}}

    cod = _find_cod10k_root(search_root)
    camo = _find_camo_root(search_root)
    train_pairs: list[tuple[Path, Path]] = []
    val_pairs: list[tuple[Path, Path]] = []
    meta = {"source": "automatic CAMO + COD10K discovery", "train_sets": {}, "validation_sets": {}}

    if cod is not None:
        cod_root, (train_img, train_gt), (test_img, test_gt) = cod
        cod_train = pair_images_with_gt(train_img, train_gt)
        cod_test = pair_images_with_gt(test_img, test_gt)
        
        cam_train = [pair for pair in cod_train if pair[0].stem.lower().startswith("cod10k-cam")]
        cam_test = [pair for pair in cod_test if pair[0].stem.lower().startswith("cod10k-cam")]
        train_pairs.extend(cam_train or cod_train)
        val_pairs.extend(cam_test or cod_test)
        meta["cod10k_root"] = str(cod_root)
        meta["train_sets"]["COD10K"] = len(cam_train or cod_train)
        meta["validation_sets"]["COD10K"] = len(cam_test or cod_test)

    if camo is not None:
        camo_root, img_root, gt_root = camo
        imgs = _index_files(img_root, {".jpg", ".jpeg", ".png"})
        gts = _index_files(gt_root, {".png", ".jpg", ".jpeg"})
        tr, va = [], []
        for stem, img_path in sorted(imgs.items()):
            if not stem.lower().startswith(("camourflage", "camouflage")) or stem not in gts:
                continue
            rel_parts = [x.lower() for x in img_path.relative_to(img_root).parts[:-1]]
            if any("train" in x for x in rel_parts): tr.append((img_path, gts[stem]))
            elif any("test" in x for x in rel_parts): va.append((img_path, gts[stem]))
        
        if not tr and (img_root / "Train").is_dir():
            tr = pair_images_with_gt(img_root / "Train", gt_root / "Train" if (gt_root / "Train").is_dir() else gt_root)
        if not va and (img_root / "Test").is_dir():
            va = pair_images_with_gt(img_root / "Test", gt_root / "Test" if (gt_root / "Test").is_dir() else gt_root)
        train_pairs.extend(tr)
        val_pairs.extend(va)
        meta["camo_root"] = str(camo_root)
        meta["train_sets"]["CAMO"] = len(tr)
        meta["validation_sets"]["CAMO"] = len(va)

    if not train_pairs:
        raise FileNotFoundError(
            f"Could not discover a training dataset under {search_root}. Attach a COD10K-v3 + CAMO dataset, "
            "or set MANUAL_TRAIN_IMAGE_DIR and MANUAL_TRAIN_GT_DIR in the notebook."
        )
    if not val_pairs:
        print("WARNING: no validation dataset was discovered; validation loss will be skipped.")
    return train_pairs, val_pairs, meta


def limited_pairs(pairs: Sequence[tuple[Path, Path]], max_items: Optional[int], seed: int = 1234):
    pairs = list(pairs)
    if max_items and len(pairs) > max_items:
        rng = random.Random(seed)
        pairs = sorted(rng.sample(pairs, max_items), key=lambda x: str(x[0]))
    return pairs


def compute_iou_official(seg: np.ndarray, gt: np.ndarray) -> float:
    intersection = seg * gt
    union = seg + gt
    return (np.count_nonzero(intersection) + 1e-6) / (np.count_nonzero(union) + 1e-6)


def get_random_structure(size: int) -> np.ndarray:
    choice = np.random.randint(1, 5)
    if choice == 1:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    if choice == 2:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    if choice == 3:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, max(1, size // 2)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, size // 2), size))


def random_dilate(seg: np.ndarray, kmin: int = 3, kmax: int = 10) -> np.ndarray:
    size = np.random.randint(kmin, kmax)
    return cv2.dilate(seg, get_random_structure(int(size)), iterations=1)


def random_erode(seg: np.ndarray, kmin: int = 3, kmax: int = 10) -> np.ndarray:
    size = np.random.randint(kmin, kmax)
    return cv2.erode(seg, get_random_structure(int(size)), iterations=1)


def structure_corrupt_official(mask: np.ndarray, max_outer_iterations: int = 50,
                               iou_target: Optional[float] = None) -> np.ndarray:
    """Bounded version of CamoDiffusion's structure-corruption routine."""
    h, w = mask.shape
    seg = (mask.astype(np.uint8) > 0).astype(np.uint8) * 255
    gt = seg.copy()
    if iou_target is None:
        iou_target = float(np.random.uniform(0.8, 1.0))
    if h <= 2 or w <= 2:
        return (seg > 127).astype(np.uint8)
    for _ in range(max(1, max_outer_iterations)):
        for _ in range(4):
            lx, ly = np.random.randint(w), np.random.randint(h)
            lw, lh = np.random.randint(lx + 1, w + 1), np.random.randint(ly + 1, h + 1)
            if np.random.rand() < 0.25:
                cx, cy = int((lx + lw) / 2), int((ly + lh) / 2)
                seg[cy, cx] = np.random.randint(2) * 255
            crop = seg[ly:lh, lx:lw]
            if np.random.rand() < 0.5:
                seg[ly:lh, lx:lw] = random_dilate(crop)
            else:
                seg[ly:lh, lx:lw] = random_erode(crop)
        if compute_iou_official(seg, gt) < iou_target:
            break
    return (seg > 127).astype(np.uint8)


class SNRScheduler:
    """Cosine alpha-bar schedule shifted in log-SNR space, as used by the supplied no-OFM notebook."""
    def __init__(self, T: int = 1000, shift: Optional[float] = None, s: float = 0.008, device: str = "cpu"):
        self.T = int(T)
        self.shift = float(shift if shift is not None else (-2 * np.log(5.5)))
        t = torch.arange(self.T + 1, dtype=torch.float64) / self.T
        f_t = torch.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        alpha_bar_base = (f_t / f_t[0]).clamp(1e-8, 1 - 1e-8)
        log_snr_base = torch.log(alpha_bar_base / (1 - alpha_bar_base))
        log_snr_shifted = log_snr_base + self.shift
        alpha_bar_shifted = torch.sigmoid(log_snr_shifted)
        self.alpha_bars = alpha_bar_shifted[1:].float().to(device)
        alpha_bars_prev = alpha_bar_shifted[:-1].float().to(device)
        self.alphas = (self.alpha_bars / alpha_bars_prev).clamp(1e-8, 1.0)
        self.betas = (1.0 - self.alphas).clamp(1e-8, 0.999)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.log_snr = log_snr_shifted[1:].float().to(device)

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, shape: torch.Size):
        out = arr.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._extract(self.sqrt_alpha_bars, t.long(), x0.shape)
        sqrt_1m_ab = self._extract(self.sqrt_one_minus_alpha_bars, t.long(), x0.shape)
        return sqrt_ab * x0 + sqrt_1m_ab * noise, noise


class CamoDiffusionDataset(Dataset):
    """Paired segmentation dataset producing image, clean mask, noisy mask, and timestep."""
    def __init__(self, pairs, scheduler: SNRScheduler, img_size: int = 384,
                 augment: bool = False, structure_corruption: bool = True,
                 max_outer_iterations: int = 50):
        self.pairs = list(pairs)
        self.scheduler = scheduler
        self.img_size = int(img_size)
        self.augment = augment
        self.use_corruption = structure_corruption
        self.max_outer_iterations = max_outer_iterations
        self.T = scheduler.T

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, gt_path = self.pairs[idx]
        with Image.open(img_path) as im:
            image = im.convert("RGB").resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        with Image.open(gt_path) as gm:
            mask = gm.convert("L").resize((self.img_size, self.img_size), Image.Resampling.NEAREST)
        if self.augment and random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        image_t = TF.normalize(TF.to_tensor(image), mean=[0.5] * 3, std=[0.5] * 3)
        mask_t = (TF.to_tensor(mask) > 0.5).float()
        mask_bin = mask_t[0].numpy().astype(np.uint8)
        corrupted = (structure_corrupt_official(mask_bin, self.max_outer_iterations)
                     if self.use_corruption else mask_bin)
        x0 = mask_t * 2.0 - 1.0
        x0_corrupted = torch.from_numpy(corrupted).float().unsqueeze(0) * 2.0 - 1.0
        t = torch.randint(0, self.T, (1,), dtype=torch.long)
        xt, noise = self.scheduler.q_sample(x0_corrupted.unsqueeze(0), t)
        return {"image": image_t, "x0": x0, "xt": xt.squeeze(0), "noise": noise.squeeze(0), "t": t.squeeze(0)}


def _extract_model_state(payload):
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    return payload


def load_model_weights(model: torch.nn.Module, checkpoint_path: str | Path, strict: bool = True):
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = _extract_model_state(payload)
    result = model.load_state_dict(state, strict=strict)
    print(f"Loaded model weights from: {checkpoint_path}")
    if not strict:
        print("missing keys:", result.missing_keys[:10], "unexpected keys:", result.unexpected_keys[:10])
    return payload


def find_attached_file(filename: str, path_hint: str = "", search_root: str = "/kaggle/input") -> Optional[str]:
    hint = path_hint.lower()
    if not os.path.isdir(search_root):
        return None
    for dirpath, dirnames, filenames in os.walk(search_root):
        dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
        if filename in filenames and (not hint or hint in dirpath.lower()):
            return os.path.join(dirpath, filename)
    return None


def _save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def _write_history(output_dir: Path, history: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(output_dir / "history.json", history)
    if history:
        keys = sorted({key for row in history for key in row.keys()})
        with open(output_dir / "history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot([r["epoch"] for r in history], [r["train_total_loss"] for r in history], label="train total")
            ax.plot([r["epoch"] for r in history], [r.get("val_loss", np.nan) for r in history], label="validation")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.grid(True, alpha=0.3); ax.legend()
            fig.tight_layout(); fig.savefig(output_dir / "training_curves.png", dpi=160); plt.close(fig)
        except Exception as exc:
            print("Could not create training plot:", exc)


def _run_train_epoch(model, loader, diffusion_scheduler, optimizer, scaler, device,
                     contrastive_lambda, delta, w_boundary, grad_accum, fp16,
                     grad_clip, max_batches=None):
    model.train()
    running = {"total": 0.0, "diffusion": 0.0, "contrastive": 0.0, "n": 0}
    optimizer.zero_grad(set_to_none=True)
    n_batches = len(loader) if max_batches is None else min(len(loader), int(max_batches))
    if n_batches == 0:
        raise RuntimeError("Training DataLoader is empty. Check dataset paths and batch size.")
    for step, batch in enumerate(tqdm(loader, total=n_batches, desc="train", leave=False)):
        if max_batches is not None and step >= max_batches:
            break
        image = batch["image"].to(device, non_blocking=True)
        x0 = batch["x0"].to(device, non_blocking=True)
        xt = batch["xt"].to(device, non_blocking=True)
        t = batch["t"].to(device, non_blocking=True).long().view(-1)
        gt01 = ((x0.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        log_snr = diffusion_scheduler.log_snr.to(device).gather(0, t)
        current_window = min(grad_accum, n_batches - (step // grad_accum) * grad_accum)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=bool(fp16 and device.type == "cuda")):
            pred_logits, feats = model.forward_with_features(xt, t, image)
            loss_diff = bce_iou_loss(pred_logits.float(), gt01)
            loss_cont = contrastive_feature_loss(feats, gt01, log_snr, delta=delta, w_boundary=w_boundary)
            loss_total = loss_diff + float(contrastive_lambda) * loss_cont
            scaled_loss = loss_total / current_window
        if not torch.isfinite(loss_total):
            raise FloatingPointError("Non-finite training loss. Try USE_FP16=False or reduce the learning rate.")
        scaler.scale(scaled_loss).backward()
        step_now = ((step + 1) % grad_accum == 0) or ((step + 1) == n_batches)
        if step_now:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        running["total"] += float(loss_total.detach().item())
        running["diffusion"] += float(loss_diff.detach().item())
        running["contrastive"] += float(loss_cont.detach().item())
        running["n"] += 1
    denom = max(1, running.pop("n"))
    return {f"train_{key}_loss": val / denom for key, val in running.items()}


@torch.no_grad()
def _run_val_epoch(model, loader, device, fp16, max_batches=None):
    model.eval()
    losses = []
    n_batches = len(loader) if max_batches is None else min(len(loader), int(max_batches))
    if n_batches == 0:
        return float("nan")
    for step, batch in enumerate(tqdm(loader, total=n_batches, desc="val", leave=False)):
        if max_batches is not None and step >= max_batches:
            break
        image = batch["image"].to(device, non_blocking=True)
        x0 = batch["x0"].to(device, non_blocking=True)
        xt = batch["xt"].to(device, non_blocking=True)
        t = batch["t"].to(device, non_blocking=True).long().view(-1)
        gt01 = ((x0.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=bool(fp16 and device.type == "cuda")):
            logits = model(xt, t, image)
            loss = bce_iou_loss(logits.float(), gt01)
        if torch.isfinite(loss):
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def fit_contrastive_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    diffusion_scheduler: SNRScheduler,
    results_dir: str,
    epochs: int,
    lr: float,
    lr_min: float = 1e-6,
    contrastive_lambda: float = 0.1,
    delta: float = 0.3,
    w_boundary: float = 3.0,
    grad_accum: int = 4,
    fp16: bool = True,
    num_workers: int = 4,
    save_every: int = 5,
    keep_last: int = 1,
    grad_clip: float = 1.0,
    time_limit_hours: Optional[float] = 11.0,
    stop_after_epoch: Optional[int] = None,
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
    resume_checkpoint: Optional[str] = None,
    init_model_checkpoint: Optional[str] = None,
    on_checkpoint=None,
    push_every: int = 5,
    settings: Optional[dict] = None,
):
    """Train/resume this no-OFM model; stage-2 can initialize from stage-1 weights."""
    device = next(model.parameters()).device
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(fp16 and device.type == "cuda"))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=lr_min)
    start_epoch, history, best_val = 0, [], float("inf")
    fit_started = time.time()

    if resume_checkpoint and Path(resume_checkpoint).is_file():
        payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(_extract_model_state(payload), strict=True)
        if isinstance(payload, dict):
            if payload.get("optimizer_state_dict"): optimizer.load_state_dict(payload["optimizer_state_dict"])
            if payload.get("scaler_state_dict"): scaler.load_state_dict(payload["scaler_state_dict"])
            if payload.get("lr_scheduler_state_dict"): lr_scheduler.load_state_dict(payload["lr_scheduler_state_dict"])
            start_epoch = int(payload.get("epoch", 0))
            history = payload.get("history", [])
            best_val = float(payload.get("best_selection_metric", payload.get("best_val_loss", float("inf"))))
        print(f"Resuming same-stage run from epoch {start_epoch} ({resume_checkpoint})")
    elif init_model_checkpoint and Path(init_model_checkpoint).is_file():
        payload = load_model_weights(model, init_model_checkpoint, strict=True)
        print(f"Starting a new stage from pretrained project checkpoint: {init_model_checkpoint}")
        start_epoch, history = 0, []
        best_val = float("inf")

    try:
        for epoch_index in range(start_epoch, epochs):
            epoch_start = time.time()
            train_metrics = _run_train_epoch(
                model, train_loader, diffusion_scheduler, optimizer, scaler, device,
                contrastive_lambda, delta, w_boundary, grad_accum, fp16, grad_clip,
                max_batches=max_train_batches,
            )
            val_loss = float("nan")
            if val_loader is not None:
                val_loss = _run_val_epoch(model, val_loader, device, fp16, max_batches=max_val_batches)
            row = {
                "epoch": epoch_index + 1,
                **train_metrics,
                "val_loss": val_loss,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "epoch_seconds": round(time.time() - epoch_start, 2),
            }
            history.append(row)
            selection_metric = val_loss if np.isfinite(val_loss) else float(train_metrics["train_total_loss"])
            improved = selection_metric < best_val or not (out / "model-best.pt").exists()
            if improved:
                best_val = selection_metric
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch_index + 1,
                    "image_size": getattr(model, "img_size", None),
                    "val_loss": float(val_loss),
                    "selection_metric": float(selection_metric),
                    "settings": settings or {},
                }, out / "model-best.pt")
            lr_scheduler.step()
            payload = {
                "epoch": epoch_index + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "history": history,
                "best_selection_metric": float(best_val),
                "settings": settings or {},
                "image_size": getattr(model, "img_size", None),
            }
            torch.save(payload, out / "checkpoint_last.pt")
            do_save = ((epoch_index + 1) % max(1, save_every) == 0) or (epoch_index + 1 == epochs)
            if do_save:
                numbered = out / f"ckpt_ep{epoch_index + 1:03d}.pt"
                torch.save(payload, numbered)
                all_ckpts = sorted(out.glob("ckpt_ep*.pt"), key=lambda p: int(re.search(r"ep(\d+)", p.name).group(1)))
                for old in all_ckpts[:-max(1, keep_last)]:
                    try: old.unlink()
                    except OSError: pass
            _write_history(out, history)
            print(
                f"Epoch {epoch_index + 1}/{epochs} | total={row['train_total_loss']:.4f} "
                f"diffusion={row['train_diffusion_loss']:.4f} contrastive={row['train_contrastive_loss']:.4f} "
                f"val={val_loss:.4f} lr={row['lr']:.2e}"
            )
            if do_save and on_checkpoint and (((epoch_index + 1) % max(1, push_every) == 0) or (epoch_index + 1 == epochs)):
                try:
                    on_checkpoint(str(out), epoch_index + 1)
                except Exception as exc:
                    print(f"Checkpoint upload failed; local checkpoint is preserved: {type(exc).__name__}: {exc}")
            if stop_after_epoch is not None and epoch_index + 1 >= stop_after_epoch:
                print(f"Stopping at requested epoch {stop_after_epoch}.")
                break
            if time_limit_hours is not None and (time.time() - fit_started) >= time_limit_hours * 3600:
                print(f"Reached time limit of {time_limit_hours} hours; stopping safely after epoch {epoch_index + 1}.")
                break
    finally:
        _write_history(out, history)
    return model, history




def push_checkpoint_dataset(results_dir: str, slug: str, notes: str = "") -> str:
    """Upload the latest checkpoint/log artifacts to a Kaggle dataset.

    Requires Kaggle notebook secrets: KAGGLE_USERNAME and KAGGLE_KEY or
    KAGGLE_API_TOKEN. Failure should be caught by the calling notebook so a
    training run can still continue with local checkpoints.
    """
    import shutil
    import tempfile
    import kagglehub
    from kaggle_secrets import UserSecretsClient

    secrets = UserSecretsClient()
    def secret(name):
        try:
            return secrets.get_secret(name)
        except Exception:
            return None

    user = secret("KAGGLE_USERNAME")
    key = secret("KAGGLE_KEY")
    token = secret("KAGGLE_API_TOKEN")
    if key and key.startswith("KGAT_") and not token:
        token = key
    if token:
        os.environ["KAGGLE_API_TOKEN"] = token
    elif user and key:
        os.environ["KAGGLE_USERNAME"] = user
        os.environ["KAGGLE_KEY"] = key
    if not user:
        raise RuntimeError("Add Kaggle secret KAGGLE_USERNAME to upload checkpoints.")
    if not (token or key):
        raise RuntimeError("Add Kaggle secret KAGGLE_KEY or KAGGLE_API_TOKEN to upload checkpoints.")

    source = Path(results_dir)
    stage = Path(tempfile.gettempdir()) / "camodiffusion_no_ofm_push"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    selected = [
        "checkpoint_last.pt", "model-best.pt", "history.csv", "history.json",
        "training_curves.png", "contrastive_config.json",
    ]
    for name in selected:
        p = source / name
        if p.is_file():
            shutil.copy2(p, stage / name)
    if not (stage / "checkpoint_last.pt").exists():
        raise FileNotFoundError(f"No checkpoint_last.pt found in {source}")
    handle = f"{user}/{slug}"
    kagglehub.dataset_upload(handle, str(stage), version_notes=notes)
    return f"https://www.kaggle.com/datasets/{handle}"
