import socket
import io

# 서버 설정
server_address = "192.168.0.158"  # 서버의 실제 IP 주소 또는 도메인 이름
server_port = 5000         # 서버 포트 번호

# 서버에 연결
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_address, server_port))

request = f"{name}"
client_socket.send(request.encode("utf-8"))
response = client_socket.recv(1024).decode("utf-8")
print(f"서버 : {response}\n")


try:
    while True:
        # 입력 받기
        print("입력 예시: AI KMS FACE OK")
        line = input("입력: ")
        if not line:  # 엔터만 치면 종료 (원하면 빼도 됨)
            break
        # 값 4개 분리
        try:
            src, user, typ, stat = line.split()
        except ValueError:
            print("입력 오류! 값 4개를 공백으로 입력하세요.")
            continue

        request = f"{src}:{user}:{typ}:{stat}"
        client_socket.send(request.encode("utf-8"))

        # 서버 응답 받기
        response = client_socket.recv(1024)
        if not response:
            print("서버가 연결을 종료했습니다.")
            break
        print("서버 응답:", response.decode("utf-8"))

finally:
    client_socket.close()