from typing import List, Optional, Any, Dict
import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from .settings import settings
from .utils_images import load_images

class InternVLRunner:
    """
    Unified runner:
      input: image_paths (List[str]) + prompt (str)
      output: text (str)  # expected JSON string by prompt
    """

    def __init__(self):
        self.model_id = settings.MODEL_ID
        dtype = torch.float16 if settings.DTYPE == "float16" else torch.bfloat16

        # InternVL3.5 HF checkpoints usually require trust_remote_code
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, use_fast=False)
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        images = load_images(image_paths)

        # Prefer InternVL's chat() if available
        if hasattr(self.model, "chat"):
            # Most InternVL chat APIs accept (tokenizer, images, query, generation_config)
            gen_cfg = {
                "max_new_tokens": max_new_tokens or settings.MAX_NEW_TOKENS,
                "do_sample": False,
            }
            # Some versions expect images as a single image or list; we always pass list.
            return self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=images,   # some implementations use "pixel_values"/"images"
                query=prompt,
                generation_config=gen_cfg,
            )

        # Fallback: transformers-style processor + generate
        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens or settings.MAX_NEW_TOKENS)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

_runner = None

def get_runner() -> InternVLRunner:
    global _runner
    if _runner is None:
        _runner = InternVLRunner()
    return _runner
