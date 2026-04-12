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

PORT = 18765


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
                print(f"  [OK] 配置已保存: {provider} / {model}")
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


def kill_port(port):
    """尝试关闭占用指定端口的进程"""
    import subprocess
    try:
        if sys.platform == "win32":
            # 用 netstat 找 PID，再 taskkill
            r = subprocess.run(
                f'netstat -ano | findstr ":{port}" | findstr "LISTENING"',
                shell=True, capture_output=True, text=True)
            for line in r.stdout.strip().splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    subprocess.run(f"taskkill /F /PID {pid}", shell=True,
                                   capture_output=True)
        else:
            subprocess.run(f"lsof -ti :{port} | xargs kill -9 2>/dev/null",
                           shell=True, capture_output=True)
    except Exception:
        pass


def main():
    # Windows 编码兼容
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    print()
    print("=" * 50)
    print("   大模型配置器 — 正在启动...")
    print("=" * 50)
    print()

    # 检查端口占用
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", PORT))
        sock.close()
    except OSError:
        sock.close()
        print(f"  [!] 端口 {PORT} 被占用，正在清理...")
        kill_port(PORT)
        import time
        time.sleep(1)
        # 再次检查
        sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock2.bind(("127.0.0.1", PORT))
            sock2.close()
            print(f"  [OK] 端口已释放")
        except OSError:
            sock2.close()
            print(f"\n  [X] 端口 {PORT} 仍被占用，无法启动")
            print(f"  请手动关闭占用端口的程序后重试")
            input("\n  按回车键退出...")
            return

    # 启动服务器
    try:
        server = HTTPServer(("127.0.0.1", PORT), ConfigHandler)
    except Exception as e:
        print(f"\n  [X] 启动失败: {e}")
        input("\n  按回车键退出...")
        return

    url = f"http://127.0.0.1:{PORT}/config"
    print(f"  [OK] 服务器已启动")
    print(f"  [OK] 浏览器即将打开: {url}")
    print()
    print("  配置完成后关闭此窗口即可")
    print()

    # 延迟打开浏览器
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  配置器已关闭。")
        server.server_close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n  [X] 程序异常: {e}")
        input("\n  按回车键退出...")
