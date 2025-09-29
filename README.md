# 상공회의소 서울기술교육센터 인텔교육 7기

## Clone code 

```shell
git clone --recurse-submodules https://github.com/kccistc/intel-03.git
```

* `--recurse-submodules` option 없이 clone 한 경우, 아래를 통해 submodule update

```shell
git submodule update --init --recursive
```

## Preparation

### Git LFS(Large File System)

* 크기가 큰 바이너리 파일들은 LFS로 관리됩니다.

* git-lfs 설치 전

```shell
# Note bin size is 132 bytes before LFS pull

$ find ./ -iname *.bin|xargs ls -l
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
-rwxrwxr-x 1 <ID> <GROUP> 132 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
```

* git-lfs 설치 후, 다음의 명령어로 전체를 가져 올 수 있습니다.

```shell
$ sudo apt install git-lfs

$ git lfs pull
$ find ./ -iname *.bin|xargs ls -l
-rw-rw-r-- 1 <ID> <GROUP> 3358630 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 3358630 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 8955146 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
-rwxrwxr-x 1 <ID> <GROUP> 8955146 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
```

### 환경설정

* [Ubuntu](./doc/environment/ubuntu.md)
* [OpenVINO](./doc/environment/openvino.md)
* [OTX](./doc/environment/otx.md)

## Team projects

### 제출방법

1. 팀구성 및 프로젝트 세부 논의 후, 각 팀은 프로젝트 진행을 위한 Github repository 생성

2. [doc/project/README.md](./doc/project/README.md)을 각 팀이 생성한 repository의 main README.md로 복사 후 팀 프로젝트에 맞게 수정 활용

3. 과제 제출시 `인텔교육 7기 Github repository`에 `New Issue` 생성. 생성된 Issue에 하기 내용 포함되어야 함.

    * Team name : Project Name
    * Project 소개
    * 팀원 및 팀원 역활
    * Project Github repository
    * Project 발표자료 업로드

4. 강사가 생성한 `Milestone`에 생성된 Issue에 추가 

### 평가방법

* [assessment-criteria.pdf](./doc/project/assessment-criteria.pdf) 참고

### 제출현황

### Team: 4 Your Eyes Only
<프로젝트 요약>
시각 장애인도 일반 사람들처럼 위험을 인식할 수 있고, 자신이 입은 옷이 어떤지, 어떤 옷이 좋을지 판단할 수 있어야하지 않을까에서 시작한 프로젝트입니다.
* Members

| 이름 | 역할 |
| :---: | :--- |
| 조성범 | PM, 안드로이드 앱 개발 |
| 이민재 | 모델 제작 |
| 윤종석 | 시스템 구성 및 코드 종합 |
| 이진우 | 하드웨어 구성 |
| 김준형 | 데이터 수집 및 모델 최적화 |
| 방진혁 | 데이터 수집 및 Qt 제작 |

* Project Github : https://github.com/CSeongBeom/INTEL7_TEAM1
* 발표자료 : [스마트글래스(최종).pdf](https://github.com/CSeongBeom/INTEL7_TEAM1/blob/d2074473ce124f6a5eeafa87983fc5108673e077/%EC%8A%A4%EB%A7%88%ED%8A%B8%EA%B8%80%EB%9E%98%EC%8A%A4(%EC%B5%9C%EC%A2%85).pdf)

