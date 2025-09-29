# iot_client.py
import socket
import threading
import time
from typing import Optional

class IoTClient:
    def __init__(self, host: str, port: int, username: str, password: str, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self._rx_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # -------------------- connection --------------------
    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.host, self.port))

        # (1) 로그인은 개행 없이!
        self.sock.sendall(f"{self.username}:{self.password}".encode())

        # (2) 첫 응답을 잠깐 동기적으로 확인해서 에러 핸들링
        self.sock.settimeout(3.0)
        try:
            resp = self.sock.recv(4096).decode(errors="ignore")
            if resp:
                print("[SERVER]", resp.strip())
            if "Authentication Error" in resp:
                raise RuntimeError("Auth failed from server")
            if "Already logged!" in resp:
                raise RuntimeError("Already logged on server")
        finally:
            # 이후에는 수신 스레드로 전환
            self.sock.settimeout(self.timeout)

        # (3) 수신 스레드 시작
        self._stop.clear()
        self._rx_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._rx_thread.start()


    def close(self):
        self._stop.set()
        try:
            if self.sock:
                self.sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            if self.sock:
                self.sock.close()
        finally:
            self.sock = None

    def _recv_loop(self):
        buf = b""
        while not self._stop.is_set():
            try:
                data = self.sock.recv(4096)
                if not data:
                    print("[CLIENT] Server closed connection.")
                    break
                buf += data
                # 서버는 줄바꿈 단위로 응답을 보내므로 \n 기준으로 출력
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    print("[SERVER]", line.decode(errors="ignore"))
            except (socket.timeout, BlockingIOError):
                continue
            except Exception as e:
                print("[CLIENT] recv error:", e)
                break

    # -------------------- helpers: send one line --------------------
    def _send_line(self, line: str):
        if not self.sock:
            raise RuntimeError("Not connected")
        if not line.endswith("\n"):
            line += "\n"
        self.sock.sendall(line.encode())

    # # -------------------- protocol methods --------------------
    # # 졸음 플래그: AI:<name>:SLP:ON|OFF
    # # (서버가 UART로 "SERVER:<name>:SLP:ON/OFF"를 씀) 
    # def send_sleep(self, on: bool):
    #     self._send_line(f"AI:{self.username}:SLP:{'ON' if on else 'OFF'}")

    # # 자리 상태: AI:<name>:FACE:OK|NO
    # # (서버는 OK/NO 누적시간을 DB에 업데이트)
    # def send_face_present(self, present: bool):
    #     self._send_line(f"AI:{self.username}:FACE:{'OK' if present else 'NO'}")

    # # 출입 이벤트: AI:<name>:FACE:IN|OUT  (서랍 LOCK/UNLOCK 제어 라우팅)
    # def send_face_inout(self, inside: bool):
    #     self._send_line(f"AI:{self.username}:FACE:{'IN' if inside else 'OUT'}")

    # # 자세 상태: 
    # #  - OK:  AI:<name>:POSTURE:OK
    # #  - BAD: AI:<name>:POSTURE:BAD:<detail>   (detail 예: back, leg, back_leg)
    # def send_posture_ok(self):
    #     self._send_line(f"AI:{self.username}:POSTURE:OK")

    # def send_posture_bad(self, detail: str):
    #     # detail 은 한 단어 권장. 서버는 다섯 토큰(i==5)일 때 BAD를 처리함.
    #     self._send_line(f"AI:{self.username}:POSTURE:BAD:{detail}")

# --------------- example ---------------
if __name__ == "__main__":
    # 환경에 맞게 바꾸세요
    HOST = "192.168.0.158"
    PORT = 5000          # 서버 실행 시 넘긴 포트 번호
    USER = "AI2"    # 서버 DB(users.username)에 존재해야 인증됨
    PW   = "PASSWD"    # 해당 사용자의 비밀번호

    cli = IoTClient(HOST, PORT, USER, PW)
    cli.connect()
    print("[CLIENT] connected, logged in.")

     
    # cli.send_face_present(True)  # AI:<name>:FACE:OK (착석)
    # time.sleep(1)

    # 간단 데모: 상태 몇 개 전송
    # cli.send_sleep(False)        # AI:<name>:SLP:OFF
    # cli.send_posture_bad("back") # AI:<name>:POSTURE:BAD:back  (굽은등)
    # time.sleep(1)
    # cli.send_posture_ok()        # AI:<name>:POSTURE:OK
    # time.sleep(1)
    # cli.send_face_inout(False)   # AI:<name>:FACE:OUT (퇴실/자리비움)

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass
    finally:
        cli.close()