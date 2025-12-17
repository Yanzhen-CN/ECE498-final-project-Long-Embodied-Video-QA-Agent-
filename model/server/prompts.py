SYSTEM_JSON = """You are a multimodal assistant. Return ONLY valid JSON. No markdown fences."""

# 给 pipeline 的统一协议：模型必须输出一个 JSON 字符串
# 你们后面做 chunk memory、QA 都可以复用这套格式
def build_prompt(user_prompt: str, schema_hint: str | None = None) -> str:
    hint = f"\nSchema:\n{schema_hint}\n" if schema_hint else ""
    return f"{SYSTEM_JSON}\n{hint}\nUser:\n{user_prompt}\nAssistant:\n"
