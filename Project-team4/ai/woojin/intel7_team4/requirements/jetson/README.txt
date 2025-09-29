# 젯슨 가상 환경 구성
- python 3.8.5
- mediapipe==0.10.9
- opencv-contrib-python==4.12.0.88


# 파이토치는 가상환경에 설치 안하고 젯슨에 설치함.
# mediapipe 프리트레이닝된 모델을 사용하였기 때문에 torch는 필요없어서 가상환경에 구성 안했습니다.
- torch                                1.13.0a0+git7c98e70
- torchvision                          0.14.0a0+5ce4506


# 현재 작업 디렉터리의 젯슨 가상환경 경로
$ source ../../annotation/eye/test_venv/bin/activate


# 우분투에서 젯슨으로 옮기기 전에 테스트한 가상환경
$ source ../../woojin_ubuntu_venv/bin/activate