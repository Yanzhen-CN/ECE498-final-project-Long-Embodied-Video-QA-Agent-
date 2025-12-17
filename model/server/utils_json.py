import json

def ensure_json(text: str) -> dict:
    """
    尝试把模型输出变成 dict。
    - 如果模型输出本身就是 JSON：直接 parse
    - 如果输出夹杂多余文本：尽量截取第一个 {...} 或 [...]
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        # naive extraction
        l = text.find("{")
        r = text.rfind("}")
        if l != -1 and r != -1 and r > l:
            return json.loads(text[l:r+1])
        raise ValueError(f"Model output is not valid JSON: {text[:200]}...")
