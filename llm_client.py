"""
DeepSeek LLM Client — OpenAI 兼容接口
配置：在项目根目录创建 .env 文件，写入 DEEPSEEK_API_KEY=sk-xxx
或者设置环境变量 DEEPSEEK_API_KEY
"""
import os
import json
from openai import OpenAI
from typing import Optional

# ── 配置 ─────────────────────────────────────────────────────────────────
_BASE_URL = "https://api.deepseek.com"
_MODEL = "deepseek-chat"  # deepseek-chat 便宜快速；deepseek-reasoner 深度推理

_API_KEY = ""

def _load_api_key() -> str:
    """从环境变量或 .env 文件加载 API Key"""
    global _API_KEY
    if _API_KEY:
        return _API_KEY

    # 优先环境变量
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        _API_KEY = key
        return key

    # 其次从 .env 文件
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("DEEPSEEK_API_KEY="):
                    _API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                    return _API_KEY

    return ""


def configure(api_key: str = None, base_url: str = None, model: str = None):
    """手动配置（优先级最高）"""
    global _API_KEY, _BASE_URL, _MODEL
    if api_key:
        _API_KEY = api_key
    if base_url:
        _BASE_URL = base_url
    if model:
        _MODEL = model


def is_available() -> bool:
    """检查 API Key 是否可用（验证格式）"""
    key = _load_api_key()
    return bool(key and key.startswith("sk-") and len(key) > 10)


def chat(system_prompt: str, user_prompt: str,
         temperature: float = 0.3, json_mode: bool = False) -> Optional[str]:
    """
    调用 DeepSeek Chat API
    :param system_prompt: 系统提示词
    :param user_prompt: 用户提示词
    :param temperature: 0=确定性, 1=创造性
    :param json_mode: 强制返回 JSON 格式
    :return: 回复文本，失败返回 None
    """
    key = _load_api_key()
    if not key:
        return None

    try:
        client = OpenAI(api_key=key, base_url=_BASE_URL)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs = {
            "model": _MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4096,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    except Exception as e:
        print(f"  [LLM ERROR] {e}")
        return None


def chat_json(system_prompt: str, user_prompt: str,
              temperature: float = 0.3) -> Optional[dict]:
    """
    调用 LLM 并解析 JSON 响应
    :return: 解析后的 dict，失败返回 None
    """
    raw = chat(system_prompt, user_prompt, temperature, json_mode=True)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 尝试提取 JSON 块
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
        return None
