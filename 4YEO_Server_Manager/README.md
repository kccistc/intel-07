# 4YourEyesOnly - IoT 이미지 모니터링 Qt 클라이언트

Qt 기반의 실시간 이미지 수신 및 채팅 클라이언트입니다. 서버로부터 AI 분석된 이미지와 메시지를 실시간으로 받아 표시합니다.

![KakaoTalk_20250929_143444588](https://github.com/user-attachments/assets/ee8ee0ae-9806-4f48-8f97-80c870dcc0e6)


## 주요 기능

- **실시간 이미지 수신**: 서버에서 브로드캐스트하는 이미지를 실시간으로 수신 및 표시
- **채팅 기능**: 다른 클라이언트와 텍스트 메시지 송수신
- **AI 분석 결과 수신**: 서버의 AI 추론 결과를 텍스트 로그로 표시
- **자동 로그인**: 연결 시 자동으로 인증 정보 전송

## 시스템 요구사항

- Ubuntu 18.04 이상 (또는 다른 Linux 배포판)
- Qt 5.14.2
- GCC 7.0 이상
- 최소 2GB RAM

## 설치 방법

### 1. Qt 설치

```bash
# Qt 온라인 설치 프로그램 다운로드
wget https://download.qt.io/official_releases/online_installers/qt-unified-linux-x64-online.run

# 실행 권한 부여
chmod +x qt-unified-linux-x64-online.run

# 설치 프로그램 실행
./qt-unified-linux-x64-online.run
```

설치 시 다음 구성요소를 선택하세요:
- Qt 5.14.2
- Desktop gcc 64-bit
- Qt Creator

### 2. 프로젝트 클론

```bash
git clone https://github.com/wjin08/4_Your_Eyez_Only.git
cd 4_Your_Eyez_Only
```

### 3. 빌드

#### 방법 1: Qt Creator 사용 (권장)

1. Qt Creator 실행
2. `File` → `Open File or Project` 선택
3. `4YourEyesOnly.pro` 파일 열기
4. `Configure Project` 클릭
5. 왼쪽 하단의 빌드 모드를 `Debug` 또는 `Release`로 선택
6. `Build` → `Build Project` (Ctrl+B)

#### 방법 2: 명령줄 사용

```bash
# Qt 환경 변수 설정 (경로는 실제 설치 경로에 맞게 수정)
export PATH=/home/your_username/Qt/6.8.3/gcc_64/bin:$PATH

# qmake 실행
qmake 4YourEyesOnly.pro

# 빌드
make

# 실행 파일 확인
ls -la 4YourEyesOnly
```

## 사용 방법

### 실행

```bash
# 빌드된 실행 파일 실행
./4YourEyesOnly

# 또는 Qt Creator에서 실행 버튼(Ctrl+R) 클릭
```

### 서버 연결

1. 프로그램 실행 후 "서버 연결" 버튼 클릭
2. 서버 IP 주소 입력 (기본값: 192.168.0.57)
3. 자동으로 로그인 정보 `[11:PASSWD]` 전송됨
4. 연결 성공 시 버튼이 "서버 해제"로 변경됨

### 주요 UI 구성

<img width="699" height="542" alt="KakaoTalk_20250929_144513713" src="https://github.com/user-attachments/assets/bfef7101-51c2-4859-bd06-8ce92152488e" />


### 메시지 전송

- **전체 메시지**: 수신자 ID를 비워두고 메시지 입력 → 송신
- **개인 메시지**: 수신자 ID 입력 + 메시지 입력 → 송신

## 프로젝트 구조

```
4YourEyesOnly/
├── 4YourEyesOnly.pro        # Qt 프로젝트 파일
├── main.cpp                 # 메인 엔트리 포인트
├── widget.h/cpp/ui          # 메인 윈도우
├── tab1.h/cpp/ui            # 이미지 수신 및 채팅 탭
├── socketclient.h/cpp       # TCP 소켓 통신 클래스
└── build/                   # 빌드 출력 디렉토리
```

## 코드 설명

### SocketClient 클래스

TCP 소켓 통신을 담당하며 다음 기능을 제공합니다:

```cpp
// 주요 기능
- 자동 로그인: connect 시그널에 람다 연결로 자동 인증
- 이미지 수신: IMAGE:filename:filesize 프로토콜 파싱
- 실시간 표시: 500바이트 이상 수신 시 부분 이미지 표시
- 텍스트 처리: 서버 메시지 실시간 수신 및 시그널 발송
```

### 통신 프로토콜

**로그인**:
```
[ID:PASSWORD]
예: [11:PASSWD]
```

**이미지 전송**:
```
IMAGE:filename:filesize\n
[binary image data]
예: IMAGE:latest.jpg:6528\n[JPEG데이터]
```

**텍스트 메시지**:
```
[발신자]: 메시지내용\n
예: [AI_Inference]: 검정색 T-Shirts입니다.\n
```

## 설정 변경

### 서버 정보 수정

`socketclient.h` 파일에서:

```cpp
QString SERVERIP = "192.168.0.57";  // 서버 IP
int SERVERPORT = 5000;               // 서버 포트
QString LOGID = "11";                // 로그인 ID
QString LOGPW = "PASSWD";            // 로그인 비밀번호
```

### 이미지 수신 임계값 조정

`socketclient.cpp`의 `processImageData()` 함수에서:

```cpp
if (m_imageBuffer.size() >= 500) {  // 500 바이트 → 원하는 값으로 변경
    emit socketRecvImageSig(m_imageBuffer);
}
```

## 문제 해결

### 빌드 오류

**오류**: `error: undefined reference to 'socketConnectServerSlot()'`
```bash
# Clean 후 재빌드
make clean
qmake
make
```

**오류**: Qt 라이브러리를 찾을 수 없음
```bash
# Qt 경로 확인 및 환경변수 설정
export PATH=/path/to/Qt/6.8.3/gcc_64/bin:$PATH
export LD_LIBRARY_PATH=/path/to/Qt/6.8.3/gcc_64/lib:$LD_LIBRARY_PATH
```

### 실행 오류

**문제**: 프로그램이 시작되지 않음
```bash
# 의존성 확인
ldd ./4YourEyesOnly

# 누락된 Qt 라이브러리가 있다면 LD_LIBRARY_PATH 설정
```

**문제**: 서버 연결 실패
- 서버 IP 주소 확인
- 방화벽 설정 확인
- 서버가 실행 중인지 확인

**문제**: 이미지는 뜨는데 로그가 안 뜸
- 콘솔 출력 확인: `*** 텍스트 메시지 수신 ***` 로그 확인
- 시그널 연결 확인: `텍스트 시그널 연결 완료` 확인

## 디버깅

### 콘솔 로그 활성화

프로그램은 기본적으로 상세한 디버그 로그를 출력합니다:

```bash
# 터미널에서 실행하여 로그 확인
./4YourEyesOnly

# 로그 파일로 저장
./4YourEyesOnly 2>&1 | tee debug.log
```

### 주요 디버그 메시지

- `socketReadDataSlot 호출됨`: 데이터 수신 시작
- `*** 새로운 이미지 헤더 감지됨 ***`: 이미지 시작
- `*** 텍스트 메시지 수신 ***`: 텍스트 메시지 수신
- `이미지 수신 완료`: 이미지 완전 수신
- `텍스트 시그널 발송 완료`: UI로 텍스트 전달

## 서버 요구사항

이 클라이언트는 다음 형식의 서버와 호환됩니다:

1. **로그인 처리**: `[ID:PW]` 형식 인증
2. **이미지 브로드캐스트**: `IMAGE:filename:size\n` 헤더 + 바이너리 데이터
3. **텍스트 메시지**: `[발신자]: 내용\n` 형식


---

