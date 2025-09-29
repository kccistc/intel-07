#!/usr/bin/env bash
# U-Boot 옵션 일괄 설정 스크립트 (주석 유지)
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
./scripts/config --enable CONFIG_CMD_SYSBOOT             # 명시적 sysboot 호출 가능
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

# ── 환경 저장소: RAW MMC 오프셋 백엔드(권장: fw_setenv 연동 쉬움, 이중화 가능)
./scripts/config --enable CONFIG_ENV_IS_IN_MMC
./scripts/config --set-val CONFIG_ENV_OFFSET 0x00800000      # ★파티션과 겹치지 않는 RAW 오프셋(예시: 8MiB)
./scripts/config --set-val CONFIG_ENV_SIZE   0x00020000      # 128KiB
./scripts/config --set-val CONFIG_ENV_OFFSET_REDUND 0x00840000
./scripts/config --enable CONFIG_SYS_REDUNDAND_ENVIRONMENT   # 환경 이중화로 전원 차단 내성
./scripts/config --enable CONFIG_CMD_SAVEENV                  # env 저장 명령
./scripts/config --enable CONFIG_CMD_ENV                      # env 편집 명령

# ──────────────────────────────────────────────────────────────
# 무결성/보안: FIT 이미지 + RSA 서명 검증
# ──────────────────────────────────────────────────────────────
./scripts/config --enable CONFIG_FIT                      # FIT 이미지 포맷
./scripts/config --enable CONFIG_FIT_VERBOSE             # 검증 로그 자세히
./scripts/config --enable CONFIG_FIT_SIGNATURE           # FIT 서명 검증 활성화
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
./scripts/config --enable CONFIG_CMD_MEMINFO

echo "[2/3] .config 재생성(olddefconfig)..."
make olddefconfig

echo "[3/3] 완료! .config에 설정이 반영되었습니다."
echo "필요 시 저장: make savedefconfig  (defconfig 갱신)"

