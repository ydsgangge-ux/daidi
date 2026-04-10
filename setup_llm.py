"""
大模型配置器 — 一键配置各种品牌的大模型
用法：python setup_llm.py
会自动打开浏览器，进入可视化配置页面
"""
import os
import sys
import json
import webbrowser
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# 项目根目录
ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(ROOT, "web")
CONFIG_PATH = os.path.join(ROOT, "llm_config.json")


class ConfigHandler(SimpleHTTPRequestHandler):
    """处理配置页面请求和 API"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        # API: 获取当前配置
        if parsed.path == "/api/config":
            cfg = {}
            if os.path.exists(CONFIG_PATH):
                try:
                    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                except Exception:
                    pass
            # 隐藏 API Key 中间部分
            key = cfg.get("api_key", "")
            if key:
                cfg["api_key_masked"] = key[:6] + "****" + key[-4:] if len(key) > 10 else "****"
            else:
                cfg["api_key_masked"] = ""
            self._json_response(cfg)
            return

        # API: 获取品牌列表
        if parsed.path == "/api/providers":
            from llm_client import PROVIDERS
            self._json_response(PROVIDERS)
            return

        # 配置页面
        if parsed.path == "/config" or parsed.path == "/":
            self.path = "/config.html"
            return super().do_GET()

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._json_response({"ok": False, "message": "无效的请求数据"}, 400)
            return

        # API: 保存配置
        if parsed.path == "/api/config":
            api_key = data.get("api_key", "").strip()
            base_url = data.get("base_url", "").strip()
            model = data.get("model", "").strip()
            provider = data.get("provider", "").strip()

            if not api_key:
                self._json_response({"ok": False, "message": "请输入 API Key"}, 400)
                return
            if not base_url:
                self._json_response({"ok": False, "message": "请填写接口地址"}, 400)
                return
            if not model:
                self._json_response({"ok": False, "message": "请选择模型"}, 400)
                return

            cfg = {"api_key": api_key, "base_url": base_url, "model": model}
            if provider:
                cfg["provider"] = provider

            try:
                with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, ensure_ascii=False, indent=2)
                print(f"  [配置已保存] 品牌={provider} 模型={model}")
                self._json_response({"ok": True, "message": "配置已保存"})
            except Exception as e:
                self._json_response({"ok": False, "message": f"保存失败: {e}"}, 500)
            return

        # API: 测试连接
        if parsed.path == "/api/test":
            api_key = data.get("api_key", "").strip()
            base_url = data.get("base_url", "").strip()
            model = data.get("model", "").strip()

            if not api_key or not base_url or not model:
                self._json_response({"ok": False, "message": "请先填写完整配置"}, 400)
                return

            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key, base_url=base_url)
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "请回复\"连接成功\""}],
                    max_tokens=20,
                    temperature=0,
                )
                text = resp.choices[0].message.content or ""
                self._json_response({"ok": True, "message": "连接成功",
                                     "response": text.strip(), "model": model})
            except Exception as e:
                err = str(e)
                # 常见错误翻译
                if "401" in err or "auth" in err.lower():
                    err = "API Key 无效或已过期，请检查后重新输入"
                elif "404" in err:
                    err = "模型名称不正确，请检查后重新选择"
                elif "timeout" in err.lower():
                    err = "连接超时，请检查网络或接口地址"
                self._json_response({"ok": False, "message": err})
            return

        self._json_response({"ok": False, "message": "未知接口"}, 404)

    def _json_response(self, data, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        pass  # 静默日志


def main():
    # Windows 编码兼容
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    port = 18765
    server = HTTPServer(("127.0.0.1", port), ConfigHandler)

    url = f"http://127.0.0.1:{port}/config"
    print()
    print("=" * 50)
    print("   大模型配置器")
    print("=" * 50)
    print(f"   浏览器已打开：{url}")
    print("   配置完成后关闭此窗口即可")
    print("=" * 50)
    print()

    # 延迟打开浏览器（等服务器启动）
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n配置器已关闭。")
        server.server_close()


if __name__ == "__main__":
    main()
