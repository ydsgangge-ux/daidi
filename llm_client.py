"""
大模型客户端 — OpenAI 兼容接口
支持品牌：DeepSeek、OpenAI、通义千问、智谱GLM、Kimi、硅基流动等
配置优先级：configure() > llm_config.json > 环境变量 > .env 文件
"""
import os
import json
from openai import OpenAI
from typing import Optional

# ── 预设品牌列表（供配置器使用）───────────────────────────────────────────
PROVIDERS = {
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "models": ["deepseek-reasoner", "deepseek-chat"],
        "model_labels": {
            "deepseek-reasoner": "DeepSeek-R1 推理（深度思考）",
            "deepseek-chat": "DeepSeek-V3 对话（快速便宜）",
        },
        "key_prefix": "sk-",
        "help_url": "https://platform.deepseek.com/api_keys",
    },
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o", "gpt-4o-mini", "o1", "o3-mini"],
        "model_labels": {
            "gpt-4o": "GPT-4o",
            "gpt-4o-mini": "GPT-4o Mini（便宜）",
            "o1": "O1 推理（深度思考）",
            "o3-mini": "O3 Mini（推理）",
        },
        "key_prefix": "sk-",
        "help_url": "https://platform.openai.com/api-keys",
    },
    "qwen": {
        "name": "通义千问（阿里）",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen-max", "qwen-plus", "qwen-turbo"],
        "model_labels": {
            "qwen-max": "通义千问-Max（最强）",
            "qwen-plus": "通义千问-Plus（均衡）",
            "qwen-turbo": "通义千问-Turbo（快速）",
        },
        "key_prefix": "sk-",
        "help_url": "https://dashscope.console.aliyun.com/apiKey",
    },
    "glm": {
        "name": "智谱 GLM",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": ["glm-4-plus", "glm-4-flash"],
        "model_labels": {
            "glm-4-plus": "GLM-4 Plus（推荐）",
            "glm-4-flash": "GLM-4 Flash（免费）",
        },
        "key_prefix": "",
        "help_url": "https://open.bigmodel.cn/usercenter/apikeys",
    },
    "kimi": {
        "name": "Kimi（月之暗面）",
        "base_url": "https://api.moonshot.cn/v1",
        "models": ["moonshot-v1-128k", "moonshot-v1-32k", "moonshot-v1-8k"],
        "model_labels": {
            "moonshot-v1-128k": "Kimi 128K（长文本）",
            "moonshot-v1-32k": "Kimi 32K",
            "moonshot-v1-8k": "Kimi 8K（快速）",
        },
        "key_prefix": "sk-",
        "help_url": "https://platform.moonshot.cn/console/api-keys",
    },
    "siliconflow": {
        "name": "硅基流动",
        "base_url": "https://api.siliconflow.cn/v1",
        "models": ["deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1",
                    "Qwen/Qwen2.5-72B-Instruct", "THUDM/glm-4-9b-chat"],
        "model_labels": {
            "deepseek-ai/DeepSeek-V3": "DeepSeek-V3（硅基）",
            "deepseek-ai/DeepSeek-R1": "DeepSeek-R1（硅基）",
            "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5 72B（硅基）",
            "THUDM/glm-4-9b-chat": "GLM-4 9B（免费）",
        },
        "key_prefix": "sk-",
        "help_url": "https://cloud.siliconflow.cn/apiKeys",
    },
    "custom": {
        "name": "自定义（OpenAI 兼容）",
        "base_url": "",
        "models": [],
        "model_labels": {},
        "key_prefix": "",
        "help_url": "",
    },
}

# ── 配置状态 ─────────────────────────────────────────────────────────────────
_BASE_URL = "https://api.deepseek.com"
_MODEL = "deepseek-reasoner"
_API_KEY = ""
_CONFIG_LOADED = False


def _get_config_path() -> str:
    """获取配置文件路径"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_config.json")


def _load_from_config_file() -> dict:
    """从 llm_config.json 加载配置"""
    global _BASE_URL, _MODEL, _API_KEY, _CONFIG_LOADED
    if _CONFIG_LOADED:
        return {"api_key": _API_KEY, "base_url": _BASE_URL, "model": _MODEL}

    config_path = _get_config_path()
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            _API_KEY = cfg.get("api_key", "")
            _BASE_URL = cfg.get("base_url", _BASE_URL)
            _MODEL = cfg.get("model", _MODEL)
            _CONFIG_LOADED = True
            return cfg
        except (json.JSONDecodeError, IOError):
            pass
    _CONFIG_LOADED = True
    return {"api_key": "", "base_url": _BASE_URL, "model": _MODEL}


def _load_api_key() -> str:
    """加载 API Key（多级回退）"""
    global _API_KEY
    if _API_KEY:
        return _API_KEY

    # 1. 从 llm_config.json 加载
    cfg = _load_from_config_file()
    if cfg.get("api_key"):
        _API_KEY = cfg["api_key"]
        return _API_KEY

    # 2. 从环境变量加载
    key = os.environ.get("LLM_API_KEY", "") or os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        _API_KEY = key
        return key

    # 3. 从 .env 文件加载
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                for prefix in ("LLM_API_KEY=", "DEEPSEEK_API_KEY="):
                    if line.startswith(prefix):
                        _API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                        return _API_KEY

    return ""


def get_config() -> dict:
    """获取当前完整配置"""
    _load_api_key()
    return {"api_key": _API_KEY, "base_url": _BASE_URL, "model": _MODEL}


def save_config(api_key: str = None, base_url: str = None, model: str = None,
                provider: str = None) -> dict:
    """
    保存配置到 llm_config.json
    :return: 保存后的完整配置
    """
    global _API_KEY, _BASE_URL, _MODEL, _CONFIG_LOADED

    if provider and provider in PROVIDERS:
        p = PROVIDERS[provider]
        if base_url is None:
            base_url = p["base_url"]

    cfg = {"api_key": api_key or _API_KEY, "base_url": base_url or _BASE_URL,
           "model": model or _MODEL}
    if provider:
        cfg["provider"] = provider

    config_path = _get_config_path()
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    _API_KEY = cfg["api_key"]
    _BASE_URL = cfg["base_url"]
    _MODEL = cfg["model"]
    _CONFIG_LOADED = True
    return cfg


def configure(api_key: str = None, base_url: str = None, model: str = None):
    """手动配置（内存级，不持久化）"""
    global _API_KEY, _BASE_URL, _MODEL
    if api_key:
        _API_KEY = api_key
    if base_url:
        _BASE_URL = base_url
    if model:
        _MODEL = model


def is_available() -> bool:
    """检查 API Key 是否可用"""
    key = _load_api_key()
    return bool(key and len(key) > 10)


def test_connection() -> dict:
    """
    测试当前配置是否可用
    :return: {"ok": bool, "model": str, "message": str, "response": str}
    """
    key = _load_api_key()
    if not key:
        return {"ok": False, "model": _MODEL, "message": "API Key 未配置", "response": ""}

    try:
        client = OpenAI(api_key=key, base_url=_BASE_URL)
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": "请回复\"连接成功\"四个字"}],
            max_tokens=20,
            temperature=0,
        )
        text = resp.choices[0].message.content or ""
        return {"ok": True, "model": _MODEL, "message": "连接成功",
                "response": text.strip()}
    except Exception as e:
        return {"ok": False, "model": _MODEL, "message": str(e), "response": ""}


def chat(system_prompt: str, user_prompt: str,
         temperature: float = 0.3, json_mode: bool = False) -> Optional[str]:
    """调用大模型 API"""
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
            "max_tokens": 4096,
        }
        # 推理模型不支持 temperature 和 json_mode
        reasoner_models = {"deepseek-reasoner", "o1", "o3-mini"}
        if _MODEL not in reasoner_models:
            kwargs["temperature"] = temperature
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    except Exception as e:
        print(f"  [LLM ERROR] {e}")
        return None


def chat_json(system_prompt: str, user_prompt: str,
              temperature: float = 0.3) -> Optional[dict]:
    """调用 LLM 并解析 JSON 响应"""
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
