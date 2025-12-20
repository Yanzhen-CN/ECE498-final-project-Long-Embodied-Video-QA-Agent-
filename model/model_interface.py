from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


# =========================
# Persistent HF cache under project/model/hf_cache
# =========================
_THIS_DIR = Path(__file__).resolve().parent
HF_CACHE_DIR = _THIS_DIR / "hf_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Global cache (load once per process)
# =========================
@dataclass
class _Bundle:
    model: Any
    tokenizer: Any


_BUNDLE: Optional[_Bundle] = None


def is_model_loaded() -> bool:
    return _BUNDLE is not None


@dataclass(frozen=True)
class InferConfig:
    """
    Inference knobs.
    Keep VRAM low by default.
    """
    max_new_tokens: int = 256
    do_sample: bool = False

    # InternVL dynamic tiling controls
    input_size: int = 448
    max_num: int = 2            # keep small (VRAM)
    use_thumbnail: bool = False # disable extra tile (VRAM)

    # Debug
    debug_tiles: bool = False


def init_model(
    *,
    model_name: str = "OpenGVLab/InternVL3_5-4B",
    dtype: torch.dtype = torch.bfloat16,
    use_flash_attn: bool = False,
    local_files_only: bool = False,
    device_map: Optional[str] = "auto",
) -> None:
    """
    Load model once. Weights/tokenizer cached under model/hf_cache/.

    IMPORTANT:
      - For InternVL (trust_remote_code), pass torch_dtype=..., NOT dtype=...
      - local_files_only=True enforces cache-only (offline).
    """
    global _BUNDLE
    if _BUNDLE is not None:
        return

    kwargs = dict(
        torch_dtype=dtype,              # ✅ DO NOT use "dtype" here
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=use_flash_attn,
        cache_dir=str(HF_CACHE_DIR),
        local_files_only=local_files_only,
    )

    # Keep your existing behavior: accelerate device_map
    if device_map is not None:
        kwargs["device_map"] = device_map

    model = AutoModel.from_pretrained(model_name, **kwargs).eval()

    # tokenizer: try fast then slow (both cached)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
            cache_dir=str(HF_CACHE_DIR),
            local_files_only=local_files_only,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
            cache_dir=str(HF_CACHE_DIR),
            local_files_only=local_files_only,
        )

    _BUNDLE = _Bundle(model=model, tokenizer=tokenizer)


def _get_bundle() -> _Bundle:
    if _BUNDLE is None:
        init_model()
    assert _BUNDLE is not None
    return _BUNDLE


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Image preprocessing (dynamic tiling)
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
    image_size: int,
    min_num: int,
    max_num: int,
    use_thumbnail: bool,
) -> List[Image.Image]:
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


def load_image(
    image_path: str,
    *,
    input_size: int,
    max_num: int,
    use_thumbnail: bool,
) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    tiles = _dynamic_preprocess(
        img,
        image_size=input_size,
        min_num=1,
        max_num=max_num,
        use_thumbnail=use_thumbnail,
    )
    tfm = _build_transform(input_size)
    return torch.stack([tfm(t) for t in tiles], dim=0)


def load_images(
    image_paths: List[str],
    *,
    input_size: int,
    max_num: int,
    use_thumbnail: bool,
) -> Tuple[torch.Tensor, List[int]]:
    pv_list: List[torch.Tensor] = []
    npl: List[int] = []
    for p in image_paths:
        pv = load_image(p, input_size=input_size, max_num=max_num, use_thumbnail=use_thumbnail)
        pv_list.append(pv)
        npl.append(int(pv.size(0)))
    pixel_values = torch.cat(pv_list, dim=0) if pv_list else torch.empty(0)
    return pixel_values, npl


def _prefix_images(n: int) -> str:
    return "".join([f"Image-{i+1}: <image>\n" for i in range(n)])


def model_interface(*, image_paths: List[str], prompt: str, cfg: Optional[InferConfig] = None) -> str:
    """
    Contract:
      raw_text = model_interface(image_paths=[...], prompt="...")
      returns str
    """
    bundle = _get_bundle()
    model = bundle.model
    tokenizer = bundle.tokenizer

    cfg = cfg or InferConfig()
    generation_config = dict(max_new_tokens=cfg.max_new_tokens, do_sample=cfg.do_sample)

    # text-only
    if not image_paths:
        with torch.inference_mode():
            response, _history = model.chat(tokenizer, None, prompt, generation_config, history=None, return_history=True)
        return str(response)

    pixel_values, num_patches_list = load_images(
        image_paths,
        input_size=cfg.input_size,
        max_num=cfg.max_num,
        use_thumbnail=cfg.use_thumbnail,
    )

    if cfg.debug_tiles:
        total_tiles = int(sum(num_patches_list))
        print(f"[Debug] num_images={len(image_paths)} num_patches_list={num_patches_list} total_tiles={total_tiles}")

    device = _model_device(model)
    if device.type == "cuda":
        pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
    else:
        pixel_values = pixel_values.to(device=device)

    question = _prefix_images(len(image_paths)) + prompt.strip()

    try:
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
        return str(response)
    finally:
        # Reduce fragmentation across chunks
        del pixel_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
