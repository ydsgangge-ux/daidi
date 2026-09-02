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
        "models": ["deepseek-v4-pro", "deepseek-v4-flash"],
        "model_labels": {
            "deepseek-v4-pro": "DeepSeek-V4-Pro 旗舰（最强推理）",
            "deepseek-v4-flash": "DeepSeek-V4-Flash 快速（性价比高）",
        },
        "key_prefix": "sk-",
        "help_url": "https://platform.deepseek.com/api_keys",
    },
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-5.4", "gpt-5.2", "gpt-4.1", "gpt-4.1-mini", "o3", "o4-mini"],
        "model_labels": {
            "gpt-5.4": "GPT-5.4 旗舰（最强推理）",
            "gpt-5.2": "GPT-5.2 高级（推荐）",
            "gpt-4.1": "GPT-4.1 通用（长上下文）",
            "gpt-4.1-mini": "GPT-4.1 Mini（快速便宜）",
            "o3": "O3 推理（深度思考）",
            "o4-mini": "O4 Mini（推理/性价比）",
        },
        "key_prefix": "sk-",
        "help_url": "https://platform.openai.com/api-keys",
    },
    "qwen": {
        "name": "通义千问（阿里）",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen3.8-max", "qwen3.7-plus", "qwen3.7-flash"],
        "model_labels": {
            "qwen3.8-max": "通义千问-Max 3.8（旗舰最强）",
            "qwen3.7-plus": "通义千问-Plus 3.7（均衡）",
            "qwen3.7-flash": "通义千问-Flash 3.7（快速便宜）",
        },
        "key_prefix": "sk-",
        "help_url": "https://dashscope.console.aliyun.com/apiKey",
    },
    "glm": {
        "name": "智谱 GLM",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": ["glm-5.3", "glm-5.2", "glm-4.7-flash"],
        "model_labels": {
            "glm-5.3": "GLM-5.3 旗舰（最强编程）",
            "glm-5.2": "GLM-5.2 高级（推荐）",
            "glm-4.7-flash": "GLM-4.7 Flash（免费）",
        },
        "key_prefix": "",
        "help_url": "https://open.bigmodel.cn/usercenter/apikeys",
    },
    "kimi": {
        "name": "Kimi（月之暗面）",
        "base_url": "https://api.moonshot.cn/v1",
        "models": ["kimi-k3", "kimi-k2.7-code", "kimi-k2.6"],
        "model_labels": {
            "kimi-k3": "Kimi K3 旗舰（2.8T参数）",
            "kimi-k2.7-code": "Kimi K2.7 Code（编程专用）",
            "kimi-k2.6": "Kimi K2.6 通用（多模态）",
        },
        "key_prefix": "sk-",
        "help_url": "https://platform.moonshot.cn/console/api-keys",
    },
    "siliconflow": {
        "name": "硅基流动",
        "base_url": "https://api.siliconflow.cn/v1",
        "models": ["deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Pro",
                    "moonshotai/Kimi-K2.7-Code", "zai-org/GLM-5.2"],
        "model_labels": {
            "deepseek-ai/DeepSeek-V4-Flash": "DeepSeek-V4-Flash（硅基）",
            "deepseek-ai/DeepSeek-V4-Pro": "DeepSeek-V4-Pro（硅基）",
            "moonshotai/Kimi-K2.7-Code": "Kimi K2.7 Code（硅基）",
            "zai-org/GLM-5.2": "GLM-5.2（硅基）",
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
_MODEL = "deepseek-v4-flash"
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
    """调用大模型 API（带当日磁盘缓存，避免同一天重复发起相同请求→省token/费用）"""
    key = _load_api_key()
    if not key:
        return None

    # 当日缓存：相同 model+prompt 直接复用，结果保存供后续复用
    try:
        from data_cache import load_json as _load_llm_cache, save_json as _save_llm_cache
        import hashlib
        cache_key = hashlib.md5(
            f"{_MODEL}|{temperature}|{system_prompt}|{user_prompt}".encode()
        ).hexdigest()
        hit = _load_llm_cache("llm", cache_key)
        if hit is not None:
            return hit
    except Exception:
        pass

    try:
        for attempt in range(3):  # 失败重试，容忍 LLM 偶发空返回/瞬时错误
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
                reasoner_models = {"o3", "o4-mini"}
                if _MODEL not in reasoner_models:
                    kwargs["temperature"] = temperature
                    if json_mode:
                        kwargs["response_format"] = {"type": "json_object"}

                response = client.chat.completions.create(**kwargs)
                text = (response.choices[0].message.content or "").strip()
                if text:  # 非空才算成功，空串视为失败触发重试
                    try:
                        _save_llm_cache("llm", cache_key, text)
                    except Exception:
                        pass
                    return text
            except Exception as e:
                if attempt < 2:
                    print(f"  [LLM 重试{attempt+2}/3] {e}")
                else:
                    print(f"  [LLM ERROR] {e}")
        return None

    except Exception as e:
        print(f"  [LLM ERROR] {e}")
        return None


def chat_json(system_prompt: str, user_prompt: str,
              temperature: float = 0.3) -> Optional[dict]:
    """调用 LLM 并解析 JSON 响应（解析失败自动重试，容忍非JSON/空返回抖动）"""
    for attempt in range(3):
        raw = chat(system_prompt, user_prompt, temperature, json_mode=True)
        if raw is None:
            if attempt < 2:
                continue
            return None
        # 尝试完整解析
        if raw.strip().startswith("{"):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
        # 尝试提取 JSON 块
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
        if attempt < 2:
            continue  # 解析失败视为无效返回，重试
    return None
