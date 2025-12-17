from __future__ import annotations
from pathlib import Path

from agent.data_to_model import read_chunks, build_prompt
from agent.model_to_memory import model_infer, normalize_record

# memory ingest is fully encapsulated inside memory module
from memory.interface import memory_ingest


def run_ingest_pipeline(
    manifest_path: Path,
    *,
    image_token: str,
    use_prev_summary: bool = True,
) -> None:
    chunks = read_chunks(manifest_path)
    prev_summary = ""

    for chunk in chunks:
        prompt = build_prompt(
            chunk=chunk,
            image_token=image_token,
            prev_summary=(prev_summary if use_prev_summary else ""),
        )

        raw_text = model_infer(prompt=prompt, image_paths=chunk.image_paths)
        record = normalize_record(raw_text, chunk)

        # hand off to memory module (it decides how/where to store)
        memory_ingest(record)

        # only needed for the next prompt
        if use_prev_summary and record.get("summary", "").strip():
            prev_summary = record["summary"].strip()

    print("[OK] ingest finished.")


if __name__ == "__main__":
    run_ingest_pipeline(
        manifest_path=Path("agent/data.json"),
        image_token="<IMAGE_TOKEN>",  # later replace with real IMAGE_TOKEN
        use_prev_summary=True,
    )
