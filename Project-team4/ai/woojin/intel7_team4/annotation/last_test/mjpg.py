# mjpg.py
import subprocess
import time

def is_port_listening(port: int) -> bool:
    try:
        out = subprocess.check_output(["ss", "-ltnp"]).decode("utf-8", "ignore")
        return f":{port} " in out or f":{port}\n" in out
    except Exception:
        return False

def start_mjpg_streamer_if_needed(mjpg_bin: str, input_plugin: str, output_plugin: str, dev_node: str, port:int, width=640, height=480, fps=15):
    if is_port_listening(port):
        print(f"[MJPG] port {port} already listening -> assume running")
        return None
    cmd = f'{mjpg_bin} -i "{input_plugin} -d {dev_node} -r {width}x{height} -f {fps}" -o "{output_plugin} -p {port} -w ./www"'
    print("[MJPG] starting mjpg-streamer:", cmd)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=None)
    # wait a bit for listener
    for _ in range(25):
        time.sleep(0.2)
        if is_port_listening(port):
            print("[MJPG] started and listening on port", port)
            return p
    print("[MJPG] failed to start (no listen)")
    return p

def kill_process_group(p):
    import os, signal
    if not p: return
    try:
        pgid = os.getpgid(p.pid)
        print(f"[MJPG] killing process group {pgid}")
        import signal, os
        os.killpg(pgid, signal.SIGTERM)
    except Exception as e:
        print("[MJPG] kill exception:", e)
