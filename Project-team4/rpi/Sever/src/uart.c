#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <mysql/mysql.h>
#include "config.h"
#include "types.h"
#include "state.h"
#include "server.h" // send_text_to_id()

int uart_open_nonblock(const char *dev)
{
    int fd = open(dev, O_RDWR | O_NONBLOCK);
    if (fd < 0)
        perror("open uart");
    return fd;
}

void uart_queues_init(void)
{
    memset(&uart_txq, 0, sizeof(uart_txq));
    memset(&uart_rxq, 0, sizeof(uart_rxq));
    pthread_mutex_init(&uart_txq.lock, NULL);
    pthread_mutex_init(&uart_rxq.lock, NULL);
}

// 송신 큐에 데이터 push
void uart_send(const char *msg, size_t len)
{
    pthread_mutex_lock(&uart_txq.lock);
    for (size_t i = 0; i < len; i++)
    {
        if (uart_txq.sz < UART_TXQ_CAP)
        {
            uart_txq.buf[uart_txq.head] = (unsigned char)msg[i];
            uart_txq.head = (uart_txq.head + 1) % UART_TXQ_CAP;
            uart_txq.sz++;
        }
    }
    pthread_mutex_unlock(&uart_txq.lock);
}

// === poll 기반 UART 스레드 ===
// - 짧은 타임아웃(20ms)으로 주기 깨어나 TX 큐 확인
// - RX는 원문 보존(orig) + 복사본(parsebuf)으로 안전 파싱
// - 프로토콜 처리: STM32:...:RFID:<key>
void *poll_uart_thread(void *arg)
{
    unsigned char rxbuf[512] = {0};  // 라인 경계 보존용 누적 버퍼
    unsigned char rrxbuf[256] = {0}; // 1회 read 버퍼
    size_t rxbuf_pos = 0;
    int fd = uart_fd;

    for (;;)
    {
        struct pollfd pfd = {
            .fd = fd,
            .events = POLLIN | POLLOUT,
            .revents = 0};

        // 짧은 타임아웃으로 주기 깨어남 (TX 큐 확인용)
        int pr = poll(&pfd, 1, 20);
        if (pr < 0)
        {
            if (errno == EINTR)
                continue;
            perror("poll");
            break;
        }

        // fd 오류/끊김
        if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))
        {
            fprintf(stderr, "[UART] fd error/hup/nval\n");
            break;
        }

        // =========================
        // RX: STM32 -> 서버
        // =========================
        if (pfd.revents & POLLIN)
        {
            for (;;)
            {
                ssize_t n = read(fd, rrxbuf, sizeof(rrxbuf));
                if (n > 0)
                {
                    // 누적 버퍼에 합치기 (초과 시 앞부분 리셋)
                    if (rxbuf_pos + (size_t)n >= sizeof(rxbuf))
                        rxbuf_pos = 0;

                    memcpy(rxbuf + rxbuf_pos, rrxbuf, (size_t)n);
                    rxbuf_pos += (size_t)n;

                    // ---- 여러 줄 분리 처리 ----
                    size_t start = 0;
                    for (size_t i = 0; i < rxbuf_pos; ++i)
                    {
                        if (rxbuf[i] == '\n')
                        {
                            size_t linelen = i - start + 1; // '\n' 포함
                            char orig[512];                 // 원문 백업
                            if (linelen >= sizeof(orig))
                                linelen = sizeof(orig) - 1;
                            memcpy(orig, rxbuf + start, linelen);
                            orig[linelen] = '\0';

                            // 가시성: 수신 로그
                            fprintf(stdout, "[UART RX] %s", orig);
                            fflush(stdout);

                            // 파싱은 복사본에서 (원문 보호)
                            char parsebuf[512];
                            memcpy(parsebuf, orig, linelen + 1);
                            char *saveptr = NULL;
                            char *pArray[8] = {0};
                            int k = 0;
                            for (char *tok = strtok_r(parsebuf, ":\r\n", &saveptr);
                                 tok && k < 8;
                                 tok = strtok_r(NULL, ":\r\n", &saveptr))
                            {
                                pArray[k++] = tok;
                            }

                            // --- 프로토콜 처리: STM32:...:RFID:<key> ---
                            if (k >= 4 && strcmp(pArray[0], "STM32") == 0 && strcmp(pArray[2], "RFID") == 0)
                            {
                                const char *rfid_key = pArray[3];

                                MYSQL *conn = mysql_init(NULL);
                                if (mysql_real_connect(conn, DB_HOST, DB_USER, DB_PASS, DB_NAME, 0, NULL, 0))
                                {
                                    char sql[256];
                                    MYSQL_RES *res = NULL;
                                    MYSQL_ROW row;
                                    char name[ID_SIZE] = {0};

                                    // (1) RFID 키로 사원 이름 조회
                                    snprintf(sql, sizeof(sql),
                                             "SELECT name FROM employees WHERE rfid_key='%s'",
                                             rfid_key);
                                    if (!mysql_query(conn, sql))
                                    {
                                        res = mysql_store_result(conn);
                                        if ((row = mysql_fetch_row(res)) && row[0])
                                        {
                                            strncpy(name, row[0], ID_SIZE - 1);

                                            // (2) 오늘 출석 row 존재 여부
                                            int today_exists = 0;
                                            snprintf(sql, sizeof(sql),
                                                     "SELECT COUNT(*) FROM attendance WHERE name='%s' AND date=CURDATE();",
                                                     name);
                                            if (!mysql_query(conn, sql))
                                            {
                                                MYSQL_RES *cres = mysql_store_result(conn);
                                                MYSQL_ROW crow;
                                                if (cres)
                                                {
                                                    if ((crow = mysql_fetch_row(cres)) && crow[0])
                                                        today_exists = atoi(crow[0]) > 0;
                                                    mysql_free_result(cres);
                                                }
                                            }

                                            if (today_exists)
                                            {
                                                // (3) IN/OUT 토글 + checkin/checkout 처리
                                                extern int rfid_toggle_and_update(const char *name);
                                                extern int flag_check(const char *name);
                                                int rc = rfid_toggle_and_update(name);
                                                if (rc == 1)
                                                {
                                                    // IN 전환: face=1이면 checkin_time 찍혔을 수 있음
                                                    if (flag_check(name))
                                                    {
                                                        // 모든 인증 OK → 알림
                                                        send_text_to_id("AI", name, "ATTENDANCE:OK");
                                                    }
                                                }
                                                else if (rc == 2)
                                                {
                                                    // OUT 전환 + checkout_time
                                                    send_text_to_id("AI", name, "ATTENDANCE:CHECKOUT");
                                                }
                                            }
                                        }
                                        if (res)
                                            mysql_free_result(res);
                                    }
                                    mysql_close(conn);
                                }
                            }

                            // (선택) 원문을 내부 RX 큐에도 저장 (다른 스레드 소비용)
                            pthread_mutex_lock(&uart_rxq.lock);
                            for (size_t j = 0; j < linelen; j++)
                            {
                                uart_rxq.buf[uart_rxq.head] = (unsigned char)orig[j];
                                uart_rxq.head = (uart_rxq.head + 1) % UART_RXQ_CAP;
                                if (uart_rxq.sz < UART_RXQ_CAP)
                                    uart_rxq.sz++;
                            }
                            pthread_mutex_unlock(&uart_rxq.lock);

                            start = i + 1; // 다음 줄 시작
                        }
                    }

                    // 남은 조각을 버퍼 앞으로 땡기기
                    if (start < rxbuf_pos)
                    {
                        memmove(rxbuf, rxbuf + start, rxbuf_pos - start);
                        rxbuf_pos -= start;
                    }
                    else
                    {
                        rxbuf_pos = 0;
                    }
                }
                else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
                {
                    break; // 더 읽을 것 없음
                }
                else if (n < 0 && errno == EINTR)
                {
                    continue; // 다시 시도
                }
                else
                {
                    // n == 0 (EOF) or 기타
                    break;
                }
            }
        }

        // =========================
        // TX: 서버 -> STM32
        // =========================
        if (uart_txq.sz > 0)
        {
            pthread_mutex_lock(&uart_txq.lock);
            size_t can = uart_txq.sz > 256 ? 256 : uart_txq.sz;
            unsigned char chunk[256];
            for (size_t i = 0; i < can; i++)
                chunk[i] = uart_txq.buf[(uart_txq.tail + i) % UART_TXQ_CAP];

            ssize_t w = write(fd, chunk, can);
            if (w > 0)
            {
                uart_txq.tail = (uart_txq.tail + (size_t)w) % UART_TXQ_CAP;
                uart_txq.sz -= (size_t)w;
            }
            pthread_mutex_unlock(&uart_txq.lock);
        }
    }

    return NULL;
}