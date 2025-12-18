import json
from pathlib import Path

from memory_interface import memory_ingest


def test_single_chunk(tmp_root: Path):
    record = {
        "video_id": "v_test",
        "chunk_id": 0,
        "t_start": 570,
        "t_end": 600,
        "summary": "A person picks up a cup.",
        "entities": ["person", "cup"],
        "events": [{"verb": "pick_up", "obj": "cup", "detail": ""}],
        "state_update": {"holding": "cup"},
        "evidence_frames": ["data/keyframes/v_test/chunk_000/img_00.jpg"],
    }

    result = memory_ingest(record, memory_root=str(tmp_root))
    assert result["status"] == "ok"
    assert result["video_id"] == "v_test"
    assert result["num_chunks"] == 1

    # saved_dir 应该是 tmp_root/v_test
    assert Path(result["saved_dir"]) == tmp_root / "v_test"

    # saved_to 是 list，且只有一个文件
    assert isinstance(result["saved_to"], list) and len(result["saved_to"]) == 1
    out_path = Path(result["saved_to"][0])

    # 路径结构：tmp_root/v_test/0570-0600.jsonl
    assert out_path == tmp_root / "v_test" / "0570-0600.jsonl"
    assert out_path.exists()

    # 文件内容：应只有一行 JSON
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])

    assert obj["video_id"] == "v_test"
    assert obj["chunk_id"] == 0
    assert obj["t_start"] == 570
    assert obj["t_end"] == 600


def test_batch_chunks(tmp_root: Path):
    records = [
        {
            "video_id": "v_batch",
            "chunk_id": 1,
            "t_start": 0,
            "t_end": 30,
            "summary": "chunk 1",
            "entities": ["person"],
            "events": [{"verb": "walk", "obj": "", "detail": ""}],
            "state_update": {},
            "evidence_frames": ["f0.jpg"],
        },
        {
            "video_id": "v_batch",
            "chunk_id": 2,
            "t_start": 30,
            "t_end": 60,
            "summary": "chunk 2",
            "entities": ["cup"],
            "events": [{"verb": "pick_up", "obj": "cup", "detail": ""}],
            "state_update": {"holding": "cup"},
            "evidence_frames": ["f1.jpg"],
        },
    ]

    result = memory_ingest(records, memory_root=str(tmp_root))
    assert result["status"] == "ok"
    assert result["video_id"] == "v_batch"
    assert result["num_chunks"] == 2

    paths = [Path(p) for p in result["saved_to"]]
    assert (tmp_root / "v_batch" / "0000-0030.jsonl") in paths
    assert (tmp_root / "v_batch" / "0030-0060.jsonl") in paths

    for p in paths:
        assert p.exists()
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1  # 每个 chunk 一个文件
        json.loads(lines[0])    # 可解析


def test_schema_validation_raises(tmp_root: Path):
    bad = {
        "video_id": "v_bad",
        "chunk_id": 0,
        # 缺少 t_start/t_end/summary/...
        "entities": [],
        "events": [],
        "state_update": {},
        "evidence_frames": [],
    }

    try:
        memory_ingest(bad, memory_root=str(tmp_root))
        assert False, "Expected ValueError but no error raised"
    except ValueError:
        pass


def test_mixed_video_id_raises(tmp_root: Path):
    records = [
        {
            "video_id": "v1",
            "chunk_id": 0,
            "t_start": 0,
            "t_end": 30,
            "summary": "ok",
            "entities": [],
            "events": [],
            "state_update": {},
            "evidence_frames": [],
        },
        {
            "video_id": "v2",  # 混了 video_id
            "chunk_id": 1,
            "t_start": 30,
            "t_end": 60,
            "summary": "ok",
            "entities": [],
            "events": [],
            "state_update": {},
            "evidence_frames": [],
        },
    ]

    try:
        memory_ingest(records, memory_root=str(tmp_root))
        assert False, "Expected ValueError but no error raised"
    except ValueError:
        pass


def main():
    # 用一个临时目录（在当前目录下创建 .tmp_test_memory）
    tmp_root = Path(".tmp_test_memory")
    if tmp_root.exists():
        # 清理旧测试
        for p in tmp_root.rglob("*"):
            if p.is_file():
                p.unlink()
        for p in sorted(tmp_root.rglob("*"), reverse=True):
            if p.is_dir():
                p.rmdir()
    tmp_root.mkdir(parents=True, exist_ok=True)

    test_single_chunk(tmp_root)
    test_batch_chunks(tmp_root)
    test_schema_validation_raises(tmp_root)
    test_mixed_video_id_raises(tmp_root)

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
