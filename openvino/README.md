
# Jetson + OpenVINO Object & Color Detection

이 프로젝트는 **ARM CPU가 탑재된 Jetson Nano**에서 **OpenVINO**와 **OpenCV**를 활용하여  
이미지에서 객체 탐지와 색상 분석을 수행하는 예제입니다.  
<img width="784" height="374" alt="스크린샷 2025-09-11 163936" src="https://github.com/user-attachments/assets/cc7e2e1f-5f30-4de4-a2f1-f601c02e8a22" />

탐지된 결과는 지정된 폴더에 저장되며, 원본 이미지 위에 **바운딩 박스, 클래스명, 색상 정보**가 표시됩니다.

---

## 1. 환경 준비

### 1.1 가상환경 생성 및 활성화
```bash
python3 -m venv onnx_venv
source onnx_venv/bin/activate
````

### 1.2 필수 패키지 설치

```bash
pip install --upgrade pip
pip install openvino opencv-python numpy
```

---

## 2. 프로젝트 구조

```
패키지 구성
openvino/
├── LICENSE
├── README.md
├── deployment/
│   ├── Classification/        # 분류 모델 관련 파일
│   ├── Detection/             # 객체 탐지 모델 관련 파일
│   ├── sample_image/          # 입력 이미지 폴더
│   ├── result_image/          # 추론 결과 저장 폴더
│   ├── opencv_with_test.py    # 객체 탐지 + 색상 분석 메인 코드
│   ├── chain_test.py          # 분류 테스트 스크립트
│   ├── debug.py               # 디버깅용 코드
│   └── project.json           # 프로젝트 메타데이터
└── example_code/
    ├── demo.py
    ├── demo_notebook.ipynb
    ├── requirements.txt
    └── requirements-notebook.txt

```
LICENSE — 라이선스 파일

README.md — 설명 문서

deployment/ — 모델, 샘플 이미지, 결과 폴더, 데모 및 테스트 스크립트 포함

example_code/ — 데모 실행 스크립트 및 Jupyter 노트북

**<span style="color:red; font-weight:bold;">opencv_with_test.py</span> — 객체 탐지 + 색상 분석 메인 코드**

---

## 3. 실행 방법

### 3.1 입력 이미지 준비

`deployed/deployment/sample_image/` 폴더에 `.jpg` 파일을 넣습니다.

### 3.2 코드 실행

```bash
cd ~/openvino/deployed/deployment
python3 opencv_with_test.py
```

---

## 4. 결과 확인

탐지된 이미지는 `../result_image/` 폴더에 저장됩니다.

* 원본 위에 **바운딩 박스, 클래스명, 색상 정보**가 표시됩니다.

---

## 5. 주요 기능

* OpenVINO를 활용한 **객체 탐지**
* 상위 2~3개 결과만 **바운딩 박스**로 표시
* 박스 중앙에 **클래스명 + 색상명 출력**
* 탐지 영역의 **평균 색상 계산 및 표시**
* 결과 이미지를 지정 폴더에 저장

---

## 6. OpenVINO가 Jetson에서 동작하는 이유

* OpenVINO는 Intel 전용이 아님 → **ARM 아키텍처용 빌드 제공**
* Jetson 환경에서는 **ARM Compute Library**를 활용해 연산 최적화
* Intel CPU 전용 기능은 빠지지만, 기본적인 **추론 기능은 동일하게 지원**
* 따라서 **Intel 하드웨어 없이도 Jetson(ARM 보드)에서 실행 가능**

---

## 7. 예시

* **입력** :
![20250904_004449](https://github.com/user-attachments/assets/8bb5332d-9d95-4496-aa85-0a1c8a1636da)
![Jesus-Is-West-Hoodies-for-Men-433x516](https://github.com/user-attachments/assets/16c9c778-793b-45c7-9f31-a08136a271f8)

  
* **출력** :
![20250904_004449](https://github.com/user-attachments/assets/086d0f6f-2b9e-42bc-808a-511881c8c725)
![Jesus-Is-West-Hoodies-for-Men-433x516](https://github.com/user-attachments/assets/f5fe9d0e-8428-467d-975c-b09ed19381e1)


---

