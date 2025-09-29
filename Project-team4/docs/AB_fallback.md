# 라즈베리파이4(bcm2711) 부트로더 U-Boot A/B

## 0. 경로 변수로 고정
```bash
# 내 워크스페이스
$ BASE=~/embedded_linux/bootloader
$ UBOOT_DIR=$BASE/u-boot

# SD가 자동 마운트될 경로 (lsblk 상 이미 확인됨)
$ BOOT=/media/ubuntu03/bootfs   # 부트(FAT32)
$ ROOT=/media/ubuntu03/rootfs   # 루트(ext4)
```
## 1. 패키지 설치 및 U-Boot 32-bit 빌드(ubuntu)
```bash
$ sudo apt update
$ sudo apt install -y git build-essential gcc-arm-linux-gnueabihf \
                        bison flex swig python3-dev libssl-dev \
                        device-tree-compiler u-boot-tools \
                        pkg-config libgnutls28-dev minicom
```
```bash
$ git clone git://git.denx.de/u-boot.git
$ cd "$UBOOT_DIR"
$ make rpi_4_32b_defconfig
$ vi enable_boot_ab_fit.sh
```
enable_boot_ab_fit.sh에 작성
```bash
#!/usr/bin/env bash
# U-Boot 옵션 일괄 설정 스크립트 (ENV 백엔드 = FAT 파일로 통일)
# 사용: chmod +x enable_boot_ab_fit.sh && ./enable_boot_ab_fit.sh
set -Eeuo pipefail

# 실패 지점 표시
trap 'echo "ERROR: \"$BASH_COMMAND\" 실행 중 실패했습니다." >&2' ERR

# 실행 위치 확인
if [[ ! -x ./scripts/config ]]; then
  echo "현재 디렉터리에 ./scripts/config 가 없습니다. U-Boot 소스 루트에서 실행하세요." >&2
  exit 1
fi

echo "[1/3] Kconfig 플래그 적용 중..."

# ──────────────────────────────────────────────────────────────
# 표준 부트 플로우 & 파일시스템/스토리지
# ──────────────────────────────────────────────────────────────
./scripts/config --enable CONFIG_BOOTSTD                 # 표준 부트 플로우 엔진
./scripts/config --enable CONFIG_CMD_BOOTFLOW            # 저장소에서 부팅 항목 자동 탐색
./scripts/config --enable CONFIG_BOOTMETH_EXTLINUX       # extlinux.conf 방식 지원
./scripts/config --enable CONFIG_CMD_SYSBOOT             # 명시적 sysboot 호출
./scripts/config --enable CONFIG_DISTRO_DEFAULTS         # 주소 변수/기본 명령 세트 자동화
./scripts/config --enable CONFIG_CMD_FS_GENERIC          # 통합 FS 커맨드
./scripts/config --enable CONFIG_FS_FAT                  # FAT 부트 파티션
./scripts/config --enable CONFIG_CMD_FAT
./scripts/config --enable CONFIG_FS_EXT4                 # extlinux가 ext4에 있어도 OK
./scripts/config --enable CONFIG_CMD_EXT4
./scripts/config --enable CONFIG_MMC                     # SD/eMMC 접근
./scripts/config --enable CONFIG_CMD_PART                # 파티션 정보 확인
./scripts/config --enable CONFIG_CMD_BOOTMENU            # (선택) U-Boot 부트 메뉴 UI

# ──────────────────────────────────────────────────────────────
# A/B 롤백 핵심: bootcount/bootlimit + 영구 저장(env)
# ──────────────────────────────────────────────────────────────
./scripts/config --enable CONFIG_BOOTCOUNT_LIMIT         # bootcount/bootlimit 코어
./scripts/config --enable CONFIG_BOOTCOUNT_ENV           # bootcount를 환경에 저장
./scripts/config --enable CONFIG_BOOTCOUNT_SAVE_ON_INIT  # 이른 단계에서도 카운트 증가(전원 차단 내성 ↑)
./scripts/config --enable CONFIG_CMD_BOOTCOUNT           # (편의) 프롬프트에서 조회/테스트
./scripts/config --enable CONFIG_CMD_SAVEENV             # env 저장 명령
./scripts/config --enable CONFIG_CMD_ENV                 # env 편집 명령

# ──────────────────────────────────────────────────────────────
# ✅ 환경 저장소: FAT 파일 백엔드 (uboot.env) — Linux fw_env.config와 일치
#   * 기존 RAW MMC 백엔드 비활성화
# ──────────────────────────────────────────────────────────────
./scripts/config --disable CONFIG_ENV_IS_IN_MMC
./scripts/config --enable  CONFIG_ENV_IS_IN_FAT
./scripts/config --set-str CONFIG_ENV_FAT_INTERFACE "mmc"
./scripts/config --set-str CONFIG_ENV_FAT_DEVICE_AND_PART "0:1"
./scripts/config --set-str CONFIG_ENV_FAT_FILE "uboot.env"
./scripts/config --enable  CONFIG_FAT_WRITE              # FAT 쓰기 허용(fatwrite 등)
# (선택) mmc 장치/파티션 명시
./scripts/config --set-val CONFIG_SYS_MMC_ENV_DEV 0
./scripts/config --set-val CONFIG_SYS_MMC_ENV_PART 0

# ──────────────────────────────────────────────────────────────
# 무결성/보안: FIT 이미지 + RSA 서명 검증
# ──────────────────────────────────────────────────────────────
./scripts/config --enable CONFIG_FIT                      # FIT 이미지 포맷
./scripts/config --enable CONFIG_FIT_VERBOSE             # 검증 로그 자세히
./scripts/config --enable CONFIG_FIT_SIGNATURE           # FIT 서명 검증
./scripts/config --enable CONFIG_RSA                     # RSA 공개키 검증
./scripts/config --enable CONFIG_HASH                    # 해시 프레임워크
./scripts/config --enable CONFIG_SHA256                  # SHA-256 해시
./scripts/config --enable CONFIG_CMD_HASH                # (편의) 해시 계산/검증 커맨드
./scripts/config --enable CONFIG_CMD_MD5SUM              # (선택) 레거시 MD5 도구
./scripts/config --enable CONFIG_MD5SUM_VERIFY

# ──────────────────────────────────────────────────────────────
# [D] 디버깅/운영 편의(선택)
# ──────────────────────────────────────────────────────────────
./scripts/config --enable CONFIG_CMD_BDI                  # 보드 정보
./scripts/config --enable CONFIG_CMD_MEMINFO              # 메모리 정보

echo "[2/3] .config 재생성(olddefconfig)..."
make olddefconfig

echo "[3/3] 완료! .config에 설정이 반영되었습니다."
echo "필요 시 저장: make savedefconfig  (defconfig 갱신)"
```
```bash
# 설정 반영 및 빌드
$ make -j$(nproc) CROSS_COMPILE=arm-linux-gnueabihf-

# 산출물 아키 확인 (반드시 ELF32/ARM)
$ readelf -h u-boot | egrep 'Class|Machine'
# 기대: Class: ELF32 / Machine: ARM
```
## 2. SD 부트 파티션에 U-Boot 배치 + config.txt 수정(ubuntu)
```bash
# u-boot.bin 복사
$ sudo cp "$UBOOT_DIR/u-boot.bin" "$BOOT/"

# config.txt 백업
$ sudo cp "$BOOT/config.txt" "$BOOT/config.txt.bak.$(date +%F-%H%M%S)"

# 중복 키 제거 후 필요한 항목 추가
$ sudo sed -i -E '/^arm_64bit=/d; /^enable_uart=/d; /^kernel=/d; /^dtoverlay=disable-bt/d' "$BOOT/config.txt"
$ cat <<'EOF' | sudo tee -a "$BOOT/config.txt" >/dev/null
arm_64bit=0
enable_uart=1
dtoverlay=disable-bt
kernel=u-boot.bin
EOF

# 적용 확인
$ grep -E '^(arm_64bit|enable_uart|dtoverlay|kernel)=' "$BOOT/config.txt"

# 끝나면 언마운트 → SD 분리
$ sync
$ sudo umount "$BOOT"
$ sudo umount "$ROOT"
```
## 3. 시리얼 콘솔 준비(ubuntu)
```bash
$ sudo minicom -o -D /dev/ttyUSB0 -b 115200
# 종료 메뉴로 나가기: Ctrl + A → X → “Leave without reset?”에서 Yes 선택
# 확인 없이 바로 종료: Ctrl + A → Q (보통 “quit without reset”)
```
- minicom 설정: Ctrl-A → O → Serial port setup
    - Bps/Par/Bits = 115200 8N1
    - Hardware Flow Control = No ⟵ 꼭!
    - Software Flow Control = No
- 배선(3.3V TTL)
    - RPi Pin 8 = GPIO14 = TXD0 → 어댑터 RX
    - RPi Pin10 = GPIO15 = RXD0 → 어댑터 TX
    - GND–GND 공통

[RPI] 전원 ON → minicom 화면에 Hit any key to stop autoboot:가 뜨면 스페이스 연타 → U-Boot 프롬프트 진입.

## 4. 커널을 시리얼로 확실히 부팅(ubuntu)
```bash
# SD카드가 꽂힌 컨트롤러를 현재 부팅 장치로 선택.
U-Boot> mmc dev 0 

# 부트 파티션(FAT, 1번 파티션) 내용을 확인. kernel7l.img, bcm2711-rpi-4-b.dtb가 있어야 함.
U-Boot> fatls mmc 0:1

# 커널 커멘드 라인 설정
U-Boot> setenv bootargs 'console=tty1 console=ttyAMA0,115200 root=/dev/mmcblk0p2 rw rootwait fsck.repair=yes loglevel=7 earlycon=pl011,0xFE201000 keep_bootcon'

# 커널(zImage)과 DTB(Device Tree) 파일을 RAM으로 로드.
U-Boot> fatload mmc 0:1 ${kernel_addr_r} kernel7l.img
U-Boot> fatload mmc 0:1 ${fdt_addr_r} bcm2711-rpi-4-b.dtb

# zImage(32비트)를 부팅(그래서 bootz 사용; 64비트 Image면 booti).
U-Boot> bootz ${kernel_addr_r} - ${fdt_addr_r}
```
```bash
# 라즈베리파이 부팅 후
$ sudo systemctl enable --now serial-getty@ttyAMA0.service
```

## 5. U-Boot extlinux 메뉴(3초 대기)
```bash
$ cd /boot/firmware
$ sudo mkdir -p extlinux
$ sudo tee extlinux/extlinux.conf >/dev/null <<'EOF'
DEFAULT rpios
TIMEOUT 8
MENU TITLE U-Boot menu

LABEL rpios
  MENU LABEL Boot Raspberry Pi OS (serial console)
  LINUX /kernel7l.img
  FDT /bcm2711-rpi-4-b.dtb
  APPEND console=tty1 console=ttyAMA0,115200 root=/dev/mmcblk0p2 rw rootwait loglevel=7 earlycon=pl011,0xFE201000 keep_bootcon
EOF
$ sync
$ sudo reboot

# U-Boot에서 자동 부팅 재개
U-Boot> run bootcmd
```

## 6. A/B 슬롯 구성 + 실패 시나리오 데모
```bash
# 부트 파티션이 쓰기 가능인지 확인 (읽기 전용이면 rw로 다시 마운트)
$ mount | grep "/boot/firmware" || true
$ sudo mount -o remount,rw /boot/firmware 2>/dev/null || true

# 2) A/B용 커널 파일 준비(없으면 생성)
$ cd /boot/firmware
$ sudo cp -n kernel7l.img kernel7l_A.img
$ sudo cp -n kernel7l.img kernel7l_B.img
$ ls -l kernel7l*.img

# 3) extlinux.conf 작성
$ sudo tee /boot/firmware/extlinux/extlinux.conf >/dev/null <<'EOF'
DEFAULT A
TIMEOUT 30
MENU TITLE U-Boot A/B demo

LABEL A
  MENU LABEL Slot A
  LINUX /kernel7l_A.img
  FDT /bcm2711-rpi-4-b.dtb
  APPEND console=tty1 console=ttyAMA0,115200 root=/dev/mmcblk0p2 rw rootwait loglevel=7 earlycon=pl011,0xFE201000 keep_bootcon slot=A

LABEL B
  MENU LABEL Slot B
  LINUX /kernel7l_B.img
  FDT /bcm2711-rpi-4-b.dtb
  APPEND console=tty1 console=ttyAMA0,115200 root=/dev/mmcblk0p2 rw rootwait loglevel=7 earlycon=pl011,0xFE201000 keep_bootcon slot=B
EOF
```
영구적으로 extlinux 사용으로 전환
```bash
U-Boot> setenv bootcmd 'run AB_boot'
U-Boot> setenv AB_boot 'sysboot mmc 0:1 any ${scriptaddr} /extlinux/extlinux.conf'
U-Boot> setenv bootdelay 5
U-Boot> saveenv

# 부팅 후 확인
## 커널 이미지(slot) 확인
$ cat /proc/cmdline | tr ' ' '\n' | grep '^slot='
## DTB 확인
$ tr -d '\0' </proc/device-tree/model
## 부트 파라미터(bootargs) 확인
$ tr -d '\0' </proc/device-tree/chosen/bootargs
```

## 7. autoboot.txt를 이용한 부팅
DTB/오버레이 적용 주체를 펌웨어(start4.elf)로 되돌려서 config.txt/tryboot.txt의 dtoverlay=...가 항상 반영되게 한다.

1. 사전 준비 & 백업
```bash
# 부트 파티션 마운트 지점(예: RPi OS는 /boot/firmware, 어떤 OS는 /boot)
$ BOOT=/boot/firmware   # 환경에 맞게 조정

# extlinux와 config.txt 백업
$ sudo cp -a $BOOT/extlinux/extlinux.conf $BOOT/extlinux/extlinux.conf.bak 2>/dev/null || true
$ sudo cp -a $BOOT/config.txt $BOOT/config.txt.bak.$(date +%F-%H%M%S)
```

2. extlinux의 FDT/FDTOVERLAYS 라인을 모두 제거.

3. autoboot.txt 생성
`autoboot.txt`는 “어떤 설정 파일을 읽을지”를 정하는 스위치다. 기본은 `config.txt`, 1회 시험은 `tryboot.txt`.
```bash
$ sudo tee $BOOT/autoboot.txt >/dev/null <<'EOF'
[all]
config=config.txt

[tryboot]
config=tryboot.txt
EOF
```

4. tryboot 생성
```bash
$ sudo cp config.txt tryboot.txt
$ sudo vi tryboot.txt
# tryboot.txt에 Shift + G 후 추가
dtoverlay=uart3
```

5. “다음 부팅 1회만 시험”
```bash
$ sudo vcmailbox 0x00038064 4 0 1   # tryboot 플래그 set(= 다음 부팅 1회 시험)
$ sudo reboot
# 또는
$ sudo reboot '0 tryboot'           # 커널을 통해 같은 효과

# 해제/취소
$ sudo vcmailbox 0x00038064 4 0 0   # tryboot 플래그 clear(= 다음 부팅도 안정)
```
- … 1 : 다음 부팅 1회만 [tryboot]/tryboot.txt를 사용. 부팅 후 자동으로 플래그 해제(원샷).
- … 0 : 플래그 해제. 이미 걸어둔 tryboot를 취소.

5. 시험 절차
tryboot 트리거 → 재부팅
```bash
$ pinctrl get 4,5
# 기대 출력
# 4: a4    pn | hi // GPIO4 = TXD3
# 5: a4    pu | hi // GPIO5 = RXD3
```

## 8. extlinux.conf로 즉시 B→A 폴백
이후 실습을 위한 autoboot.txt와 tryboot.txt 삭제
```bash
# 부트 파티션에서 제거
$ sudo rm /boot/firmware/autoboot.txt
$ sudo rm /boot/firmware/tryboot.txt

# extlinux.conf 확인
$ ls /boot/firmware/extlinux/
```

`extlinux.conf` file에 FDTOVERLAY로 dtbo 파일을 적어놔서 dtb 파일에 덧입히려 했으나 실패하여 방식 변경
```bash
$ sudo mkdir dtb-stable
$ sudo mkdir dtb-test
$ sudo cp bcm2711-rpi-4-b.dtb dtb-stable/
# uart.dtbo 파일을 부팅하는 기본dtb 파일에 merge
$ sudo fdtoverlay -v \
  -i /boot/firmware/bcm2711-rpi-4-b.dtb \
  -o /boot/firmware/dtb-test/bcm2711-rpi-4-b+uart3.dtb \
  /boot/firmware/overlays/uart3.dtbo
```

`extlinux.conf` A/B slot으로
```bash
$ sudo tee /boot/firmware/extlinux/extlinux-A.conf >/dev/null <<'EOF'
DEFAULT A
TIMEOUT 5
MENU TITLE Slot=A first

LABEL A
  MENU LABEL Slot A (default)
  LINUX /kernel7l_A.img
  FDT /dtb-stable/bcm2711-rpi-4-b.dtb
  APPEND console=tty1 console=ttyAMA0,115200 root=/dev/mmcblk0p2 rw rootwait loglevel=7 earlycon=pl011,0xFE201000 keep_bootcon slot=A

LABEL B
  MENU LABEL Slot B (fallback)
  LINUX /kernel7l_B.img
  FDT /dtb-test/bcm2711-rpi-4-b+uart3.dtb
  APPEND console=tty1 console=ttyAMA0,115200 root=/dev/mmcblk0p2 rw rootwait loglevel=7 earlycon=pl011,0xFE201000 keep_bootcon slot=B
EOF
$ sudo tee /boot/firmware/extlinux/extlinux-B.conf >/dev/null <<'EOF'
DEFAULT B
TIMEOUT 5
MENU TITLE Slot=B first

LABEL B
  MENU LABEL Slot B (default)
  LINUX /kernel7l_B.img
  FDT /dtb-test/bcm2711-rpi-4-b+uart3.dtb
  APPEND console=tty1 console=ttyAMA0,115200 root=/dev/mmcblk0p2 rw rootwait loglevel=7 earlycon=pl011,0xFE201000 keep_bootcon slot=B

LABEL A
  MENU LABEL Slot A (fallback)
  LINUX /kernel7l_A.img
  FDT /dtb-stable/bcm2711-rpi-4-b.dtb
  APPEND console=tty1 console=ttyAMA0,115200 root=/dev/mmcblk0p2 rw rootwait loglevel=7 earlycon=pl011,0xFE201000 keep_bootcon slot=A
EOF

$ sync
$ sudo reboot
```
U-Boot 진입
```bash
U-Boot> setenv AB_boot 'sysboot mmc 0:1 any ${scriptaddr} /extlinux/extlinux-"${slot}".conf'
U-Boot> setenv slot B
U-Boot> saveenv
U-Boot> run bootcmd
```

## 9. U-Boot A/B 자동 롤백 (bootcount/bootlimit 기반)
U-Boot 환경 스니펫
```bash
# (전제) extlinux-A/B.conf는 기존 그대로 사용
U-Boot> setenv AB_boot 'sysboot mmc 0:1 any ${scriptaddr} /extlinux/extlinux-"${slot}".conf'

# 실패 한계
U-Boot> setenv bootlimit 2

# 현재 슬롯(원하는 기본값)
U-Boot> setenv slot A

# 정상 부팅 마커 파일(부트 파티션 FAT 루트)
U-Boot> setenv success_mark /uboot-good

# 1) bootcount 수동 증가 + 저장
U-Boot> setenv inc_bootcount '\
if env exists bootcount; then setexpr bootcount ${bootcount} + 1; else setenv bootcount 1; fi; \
saveenv'

# 2) bootlimit 초과 시 처리: 슬롯 토글 → 카운터 리셋 → 저장 → 새 슬롯로 부팅
U-Boot> setenv altbootcmd '\
echo "*** bootlimit reached: switching slot ***"; \
if test "${slot}" = "A"; then setenv slot B; else setenv slot A; fi; \
setenv bootcount 0; saveenv; run AB_boot'

# 3) preboot 순서:
#    (a) 성공 마커 있으면 카운터 0으로 만들고 마커 삭제(원샷)
#    (b) 아니면 카운터 1 증가
#    (c) 상태 출력 후, 임계 넘었으면 즉시 altbootcmd 실행
U-Boot> setenv preboot '\
if test -e mmc 0:1 ${success_mark}; then \
  echo "Found success mark -> reset bootcount"; \
  setenv bootcount 0; saveenv; \
  fatrm mmc 0:1 ${success_mark} || true; \
else \
  run inc_bootcount; \
fi; \
echo "slot=${slot} bootcount=${bootcount} bootlimit=${bootlimit}"; \
if env exists bootlimit && env exists bootcount; then \
  test ${bootcount} -gt ${bootlimit} && run altbootcmd; \
fi'

# 4) 정상 부팅 경로
U-Boot> setenv bootcmd 'run AB_boot'

# 5) 부팅 지연과 저장
U-Boot> setenv bootdelay 5
U-Boot> saveenv
```
```bash
$ sudo tee /etc/systemd/system/ab-mark-good.service >/dev/null <<'EOF'
[Unit]
Description=Write U-Boot GOOD marker (no fw_setenv)
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'touch /boot/firmware/uboot-good || touch /boot/uboot-good'

[Install]
WantedBy=multi-user.target
EOF

$ sudo systemctl daemon-reload
$ sudo systemctl enable --now ab-mark-good.service
```

## ?. 환경변수 읽어오기
```bash
# 1) u-boot-tools가 없다면
$ sudo apt update
$ sudo apt install -y u-boot-tools

# 2) fw_env.config 생성 (파일 백엔드: 경로, 오프셋, 크기)
$ echo '/boot/firmware/uboot.env 0x0 0x20000' | sudo tee /etc/fw_env.config
$ sudo chmod 644 /etc/fw_env.config

$ sudo fw_printenv
```