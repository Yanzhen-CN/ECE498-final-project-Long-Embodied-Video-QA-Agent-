# model/interface.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Any

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


# =========================
# 0) Fixed local HF cache under /model
#    Download once -> reuse forever (as long as this folder persists)
# =========================
_THIS_DIR = Path(__file__).resolve().parent
HF_CACHE_DIR = _THIS_DIR / "hf_cache"   # e.g., <repo_root>/model/hf_cache
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 1) Global cache (load once per process)
# =========================
@dataclass
class _Bundle:
    model: Any
    tokenizer: Any


_BUNDLE: Optional[_Bundle] = None


def is_model_loaded() -> bool:
    return _BUNDLE is not None


def init_model(
    *,
    model_name: str = "OpenGVLab/InternVL3_5-4B",  # or "OpenGVLab/InternVL3_5-8B"
    dtype: torch.dtype = torch.bfloat16,
    use_flash_attn: bool = False,
    device_map: Optional[str] = None,  # set "auto" if you want; otherwise None + .cuda()
    local_files_only: bool = False,    # set True after first download to enforce offline/cache-only
) -> None:
    """
    Load InternVL3.5 model ONCE and cache globally.
    HuggingFace weights will be stored under `model/hf_cache/` (this folder).
    """
    global _BUNDLE
    if _BUNDLE is not None:
        return

    kwargs = dict(
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=use_flash_attn,
        cache_dir=str(HF_CACHE_DIR),         # <-- key: persist under model/
        local_files_only=local_files_only,   # <-- key: force cache-only if True
    )
    if device_map is not None:
        kwargs["device_map"] = device_map

    model = AutoModel.from_pretrained(model_name, **kwargs).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False,
        cache_dir=str(HF_CACHE_DIR),         # <-- also persist tokenizer files
        local_files_only=local_files_only,
    )

    # If not using device_map, move to GPU if available
    if device_map is None and torch.cuda.is_available():
        model = model.cuda()

    _BUNDLE = _Bundle(model=model, tokenizer=tokenizer)


def _get_bundle() -> _Bundle:
    if _BUNDLE is None:
        init_model()
    assert _BUNDLE is not None
    return _BUNDLE


def _model_device(model: Any) -> torch.device:
    # robust: works for both normal and device_map models
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 2) Image preprocessing (tiles -> tensor)
# =========================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(
    aspect_ratio: float, target_ratios: List[Tuple[int, int]], width: int, height: int, image_size: int
) -> Tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for r in target_ratios:
        tr = r[0] / r[1]
        diff = abs(aspect_ratio - tr)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = r
        elif diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * r[0] * r[1]:
                best_ratio = r
    return best_ratio


def _dynamic_preprocess(
    image: Image.Image,
    *,
    image_size: int = 448,
    min_num: int = 1,
    max_num: int = 12,
    use_thumbnail: bool = True,
) -> List[Image.Image]:
    """
    Split one image into multiple 448x448 tiles (<= max_num) + optional thumbnail.
    """
    w, h = image.size
    aspect_ratio = w / h

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(list(target_ratios), key=lambda x: x[0] * x[1])

    grid_w, grid_h = _find_closest_aspect_ratio(aspect_ratio, target_ratios, w, h, image_size)
    target_w, target_h = image_size * grid_w, image_size * grid_h

    resized = image.resize((target_w, target_h))
    tiles: List[Image.Image] = []
    for idx in range(grid_w * grid_h):
        x = (idx % grid_w) * image_size
        y = (idx // grid_w) * image_size
        tiles.append(resized.crop((x, y, x + image_size, y + image_size)))

    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))

    return tiles


def load_image(image_path: str, *, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    """
    Returns: pixel_values [num_tiles, 3, H, W] on CPU (caller moves to device/dtype)
    """
    img = Image.open(image_path).convert("RGB")
    tiles = _dynamic_preprocess(img, image_size=input_size, max_num=max_num, use_thumbnail=True)
    tfm = _build_transform(input_size)
    return torch.stack([tfm(t) for t in tiles], dim=0)


def load_images(image_paths: List[str], *, input_size: int = 448, max_num: int = 12) -> Tuple[torch.Tensor, List[int]]:
    """
    For N images:
      pixel_values: concatenated tiles [sum(num_tiles_i), 3, H, W]
      num_patches_list: [num_tiles_1, ..., num_tiles_N]
    """
    pv_list: List[torch.Tensor] = []
    npl: List[int] = []
    for p in image_paths:
        pv = load_image(p, input_size=input_size, max_num=max_num)
        pv_list.append(pv)
        npl.append(int(pv.size(0)))
    pixel_values = torch.cat(pv_list, dim=0) if pv_list else torch.empty(0)
    return pixel_values, npl


# =========================
# 3) Prompt helpers (multi-image numbering)
# =========================
def _prefix_images(n: int) -> str:
    return "".join([f"Image-{i+1}: <image>\n" for i in range(n)])


# =========================
# 4) Public interface
# =========================
def model_interface(*, image_paths: List[str], prompt: str) -> str:
    """
    Project contract:
      raw_text = model_interface(image_paths=chunk.image_paths, prompt=prompt)
      returns: str (raw model output)
    """
    bundle = _get_bundle()
    model = bundle.model
    tokenizer = bundle.tokenizer

    generation_config = dict(max_new_tokens=512, do_sample=False)

    # ---- pure-text conversation ----
    if not image_paths:
        response, _history = model.chat(tokenizer, None, prompt, generation_config, history=None, return_history=True)
        return str(response)

    # ---- multi-image conversation (separate images) ----
    pixel_values, num_patches_list = load_images(image_paths, input_size=448, max_num=4)

    device = _model_device(model)
    target_dtype = torch.bfloat16  # matches your default init dtype
    if device.type == "cuda":
        pixel_values = pixel_values.to(device=device, dtype=target_dtype)
    else:
        pixel_values = pixel_values.to(device=device)

    question = _prefix_images(len(image_paths)) + prompt.strip()

    with torch.inference_mode():
        response, _history = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=True,
        )
    del pixel_values
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(response)
