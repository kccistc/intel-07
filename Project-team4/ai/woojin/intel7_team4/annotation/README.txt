################################################################################
eye 디렉터리에 가상환경 test_venv가 있습니다.

# dataset 디렉터리
    - 웹캠의 프레임을 필요시 jpg로 저장.


# eye 디렉터리
    - eye_EAR_save_export_json.py # 안경이 있을때와 없을때를 구분하여 30초간 눈 크기 정보를 계산해서 json파일로 저장
    - sleep_eye.py # 저장된 json파일을 불러와서 졸음을 감지하는 코드, json 파일이 없으면 디폴트값을 사용

# face 디렉터리
    - face.py # 얼굴 인증에 필요한 정보 계산 예시 코드

# neck_eye 디렉터리
    - neck_eye_final.py # 마지막으로 테스트된 코드
    < 기능 >
        1. 서버 연결, 로그인
        2. CAM 연결
        3. 서버 데이터 베이스에 저장된 face 인증 상태 값을 받아옴.
            1. 인증 상태가 1이라면 `ATTENDANCE:OK` 명령 대기.
            2. 인증 상태가 0이라면 `AUTH:VALUE` 명령으로 face 인증에 필요한 값을 받아옴.
               인증에 성공 하면 서버에서 `ATTENDANCE:OK` 명령 대기
        4. `ATTENDANCE:OK` 명령을 받으면 졸음감지, 거북목을 인식해서 서버에 사용자의 상태를 보내기 진행.

# turtle_neck 디렉터리
    - annotation.py # 이미지 데이터셋에 관절을 표시해서 dataset에 저장하는 코드
    - turtle_neck.py 
    < 기능 >
        1. 웹캠에서
        2. 관절표시
        3. 거북목 판단 ( 왼쪽 귀, 오른쪽 귀, 코의 좌표를 잇는 중간지점과 목을 이어서 거북목을 판단)
    - neck_face_body_annotation.py # 데이터셋에 관절을 추가해서 저장하는 코드
    
