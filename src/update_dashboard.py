import time
import re
import json
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import socketserver
from pathlib import Path

TASKS_DIR = Path(r"C:\Users\AlgoRubik\.gemini\antigravity\brain\b399ee3c-3a92-4776-abf1-20e9c7e4d7da\.system_generated\tasks")
CSV_PATH = Path(r"D:\Downloads\concrete_ddpm\experiments\exp6_level_c_spatial_physics\logs\training_loss.csv")
BRAIN_DIR = Path(r"C:\Users\AlgoRubik\.gemini\antigravity\brain\b399ee3c-3a92-4776-abf1-20e9c7e4d7da")
DATA_STREAM_HTML = BRAIN_DIR / "data_stream.html"
STATUS_JSON = BRAIN_DIR / "status.json"
STATUS_JS = BRAIN_DIR / "status.js"

PATTERN = re.compile(r'Epoch (\d+)/(\d+):\s+(\d+)%\|.*?\|\s+(\d+)/(\d+)\s+\[([\d:]+)<([\d:]+),\s+([\d\.]+)it/s,\s+loss=([0-9\.]+)\]')

def get_latest_train_log():
    if not TASKS_DIR.exists():
        return None
    logs = sorted(TASKS_DIR.glob("task-*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for l in logs:
        try:
            with open(l, "r", encoding="utf-8", errors="ignore") as f:
                head = f.read(4096)
                if "train_exp6_spatial_physics_256" in head or "train_exp5_mechanisms_256" in head or "Epoch " in head:
                    return l
        except Exception:
            pass
    return logs[0] if logs else None

CURRENT_DATA = {
    "epoch": 9,
    "total_epochs": 50,
    "pct": 60,
    "step": 1323,
    "total_steps": 2190,
    "elapsed": "05:55",
    "remaining_epoch": "04:06",
    "speed": 3.52,
    "loss": 0.00872,
    "smooth_loss": 0.015,
    "recent_losses": [],
    "completed_epochs": [],
    "updated_at": "--:--:--"
}

class TelemetryHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Quiet

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        try:
            self.send_response(200)
            self.end_headers()
        except Exception:
            pass

    def do_GET(self):
        try:
            path = self.path.split("?")[0]
            if path == "/status":
                payload = json.dumps(CURRENT_DATA).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            elif path == "/status.js":
                js = f"window.__TRAINING_DATA__ = {json.dumps(CURRENT_DATA)};\nif (window.__ON_TRAINING_UPDATE__) {{ window.__ON_TRAINING_UPDATE__(window.__TRAINING_DATA__); }}\n"
                payload = js.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/javascript")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            elif path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                while True:
                    payload = json.dumps(CURRENT_DATA)
                    msg = f"data: {payload}\n\n".encode("utf-8")
                    self.wfile.write(msg)
                    self.wfile.flush()
                    time.sleep(1.5)
            else:
                self.send_response(404)
                self.end_headers()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass
        except Exception:
            pass

def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    while True:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", 49152), TelemetryHandler)
            print("ThreadingHTTPServer listening on http://127.0.0.1:49152")
            server.serve_forever()
        except Exception as e:
            print(f"Server restart due to: {e}")
            time.sleep(2)

def get_completed_epochs():
    completed = []
    if CSV_PATH.exists():
        try:
            with open(CSV_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[1:]:
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        completed.append({
                            "epoch": int(parts[0]),
                            "mean_loss": float(parts[1]),
                            "epoch_time_sec": float(parts[2])
                        })
        except Exception:
            pass
    return completed

def main():
    global CURRENT_DATA
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    last_losses = []
    while True:
        try:
            log_path = get_latest_train_log()
            if log_path and log_path.exists():
                with open(log_path, "rb") as f:
                    f.seek(0, 2)
                    size = f.tell()
                    read_size = min(size, 65536)
                    f.seek(size - read_size)
                    content = f.read().decode("utf-8", errors="ignore")

                lines = content.splitlines()
                latest_match = None
                recent_losses = []
                for line in lines:
                    m = PATTERN.search(line)
                    if m:
                        latest_match = m
                        recent_losses.append(float(m.group(9)))

                if latest_match:
                    epoch_cur, epoch_tot, pct, step_cur, step_tot, elap, rem, speed, cur_loss = latest_match.groups()
                    completed = get_completed_epochs()

                    if recent_losses:
                        last_losses = recent_losses[-45:]

                    smooth = sum(last_losses[-12:]) / len(last_losses[-12:]) if last_losses else float(cur_loss)

                    CURRENT_DATA = {
                        "epoch": int(epoch_cur),
                        "total_epochs": int(epoch_tot),
                        "pct": int(pct),
                        "step": int(step_cur),
                        "total_steps": int(step_tot),
                        "elapsed": elap,
                        "remaining_epoch": rem,
                        "speed": float(speed),
                        "loss": float(cur_loss),
                        "smooth_loss": round(smooth, 4),
                        "recent_losses": [round(x, 5) for x in last_losses],
                        "completed_epochs": completed,
                        "updated_at": time.strftime("%H:%M:%S")
                    }

                    try:
                        raw_json = json.dumps(CURRENT_DATA)
                        iframe_html = f"""<!DOCTYPE html><html><body><script>
window.parent.postMessage({raw_json}, '*');
setTimeout(() => {{ window.location.reload(); }}, 1500);
</script></body></html>"""
                        with open(DATA_STREAM_HTML, "w", encoding="utf-8") as df:
                            df.write(iframe_html)
                    except Exception:
                        pass

                    try:
                        with open(STATUS_JSON, "w", encoding="utf-8") as jf:
                            json.dump(CURRENT_DATA, jf)
                        with open(STATUS_JS, "w", encoding="utf-8") as jsf:
                            jsf.write(f"window.__TRAINING_DATA__ = {raw_json};\nif (window.__ON_TRAINING_UPDATE__) {{ window.__ON_TRAINING_UPDATE__(window.__TRAINING_DATA__); }}\n")
                    except Exception:
                        pass

        except Exception:
            pass

        time.sleep(1.5)

if __name__ == "__main__":
    main()
