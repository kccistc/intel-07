#pragma once


// 공통 상수/설정
#define BUF_SIZE 100
#define MAX_CLNT 32
#define ID_SIZE 50
#define ARR_CNT 6


#define UART_TXQ_CAP 4096
#define UART_RXQ_CAP 4096


// 환경에 맞게 수정
#define UART_DEV "/dev/uart3_raw"


// DB 접속 정보 (실서비스면 환경변수/설정파일로 이동 권장)
#define DB_HOST "localhost"
#define DB_USER "iot"
#define DB_PASS "pwiot"
#define DB_NAME "memberdb"