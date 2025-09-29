import requests
import os

# AI 분석 서버의 주소
SERVER_URL = "http://192.168.0.40:5001/upload"

# 전송할 이미지 파일의 경로
IMAGE_PATH = "test1.jpg"

def send_image_for_analysis(image_path):
    """지정된 이미지를 AI 분석 서버로 전송하고 응답을 출력합니다."""
    
    if not os.path.exists(image_path):
        print(f"에러: 이미지 파일을 찾을 수 없습니다: {image_path}")
        return

    try:
        # 파일을 바이너리 모드로 열기
        with open(image_path, "rb") as image_file:
            # 'multipart/form-data' 형식으로 파일을 전송
            files = {
                "image": (os.path.basename(image_path), image_file, "image/jpeg")
            }
            
            print(f"'{image_path}' 이미지를 서버({SERVER_URL})로 전송합니다...")
            
            # POST 요청 보내기
            response = requests.post(SERVER_URL, files=files, timeout=60) # 타임아웃을 넉넉하게 60초로 설정
            
            # 응답 상태 코드 확인
            response.raise_for_status()
            
            print("\n[성공] 서버로부터 응답을 받았습니다.")
            print(f"상태 코드: {response.status_code}")
            
            # JSON 응답 출력
            print("응답 내용:")
            print(response.json())

    except requests.exceptions.RequestException as e:
        print(f"\n[에러] 서버에 연결할 수 없습니다: {e}")
    except Exception as e:
        print(f"\n[에러] 예기치 못한 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    send_image_for_analysis(IMAGE_PATH)
