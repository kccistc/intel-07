#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <arpa/inet.h>
#include <mysql/mysql.h>
#include "config.h"
#include "types.h"
#include "state.h"
#include "util.h"
#include "uart.h"
#include "db.h"

void send_msg(MSG_INFO *msg_info, CLIENT_INFO *first_client_info)
{
    int i = 0;
    printf("msg to : %s, %s\n", msg_info->to, msg_info->msg);
    for (i = 0; i < MAX_CLNT; i++)
        if ((first_client_info + i)->fd != -1)
            if (!strcmp(msg_info->to, (first_client_info + i)->id))
                write((first_client_info + i)->fd, msg_info->msg, msg_info->len);
}

// SERVER:<from>:<payload>\n 형태로 보냄
void send_text_to_id(const char *to, const char *from, const char *fmt, ...)
{
    if (!g_clients || !to)
        return;

    char payload[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(payload, sizeof(payload), fmt, ap);
    va_end(ap);

    char line[600];
    if (!strcmp(to, "QT"))
        snprintf(line, sizeof(line), "SERVER|%s|%s\n", from, payload);
    else
        snprintf(line, sizeof(line), "SERVER:%s:%s\n", from, payload);

    MSG_INFO mi = {
        .fd = -1,
        .from = (char *)"SERVER",
        .to = (char *)to,
        .msg = line,
        .len = (int)strlen(line),
    };
    send_msg(&mi, g_clients);
}

// 클라이언트 스레드 (원본 주석 일부 보존)
void *clnt_connection(void *arg)
{
    static int today_absent_sec = 0; // 오늘 비운 시간 누적 (초)
    CLIENT_INFO *client_info = (CLIENT_INFO *)arg;
    int str_len = 0;
    int index = client_info->index;
    char msg[BUF_SIZE];
    char to_msg[MAX_CLNT * ID_SIZE + 1];
    char strBuff[BUF_SIZE * 2] = {0};
    MSG_INFO msg_info;
    CLIENT_INFO *first_client_info;

    first_client_info = (CLIENT_INFO *)((void *)client_info - (void *)(sizeof(CLIENT_INFO) * index));

    while (1)
    {
        memset(msg, 0x0, sizeof(msg));
        str_len = read(client_info->fd, msg, sizeof(msg) - 1);
        if (str_len <= 0)
            break;

        msg[str_len] = '\0';

        char *line = strtok(msg, "\n");
        char raw_line[BUF_SIZE * 2];
        strncpy(raw_line, line, sizeof(raw_line) - 1);
        raw_line[sizeof(raw_line) - 1] = '\0';
        while (line != NULL)
        {
            // ARR_CNT 5: [AI:KMS:FACE:OK] 등 파싱
            char *pArray[ARR_CNT] = {0};
            int i = 0;
            char *pToken = strtok(line, ":");
            while (pToken != NULL && i < ARR_CNT)
            {
                pArray[i++] = pToken;
                pToken = strtok(NULL, ":");
            }

            // ===== [AI:KMS:SLP:ON/OFF] 졸음 감지 UART 전송 (항상 맨 위!) =====
            if (i == 4 && strcmp(pArray[0], "AI") == 0 && strcmp(pArray[2], "SLP") == 0)
            {
                char uart_msg[64] = {0};
                if (strcmp(pArray[3], "ON") == 0)
                {
                    snprintf(uart_msg, sizeof(uart_msg), "SERVER:%s:SLP:ON\n", pArray[1]);
                    printf("[UART] Sent: %s\n", uart_msg);
                    // slp_flag = 0;
                }
                else if (strcmp(pArray[3], "OFF") == 0)
                {
                    snprintf(uart_msg, sizeof(uart_msg), "SERVER:%s:SLP:OFF\n", pArray[1]);
                    printf("[UART] Sent: %s\n", uart_msg);
                    // slp_flag = 0;
                }
                uart_send(uart_msg, strlen(uart_msg));
            }
            // ===== AI가 얼굴데이터 요청 시 DB에서 값 전달 =====
            if (i >= 4 && strcmp(pArray[0], "AI") == 0 && strcmp(pArray[2], "AUTH") == 0)
            {
                char *name = pArray[1];
                char *req_type = pArray[3];

                if (strcmp(req_type, "VALUE") == 0) // 얼굴값, ear 값 반환
                {
                    MYSQL *conn_face;
                    MYSQL_RES *res_face;
                    MYSQL_ROW row_face;
                    char sql_cmd[256];
                    int face_id1 = 0, face_id2 = 0, face_id3 = 0;
                    float ear = 0.0f;

                    conn_face = mysql_init(NULL);
                    if (mysql_real_connect(conn_face, DB_HOST, DB_USER, DB_PASS, DB_NAME, 0, NULL, 0))
                    {
                        snprintf(sql_cmd, sizeof(sql_cmd),
                                 "SELECT face_id1, face_id2, face_id3, ear FROM employees WHERE name='%s'",
                                 name);
                        if (!mysql_query(conn_face, sql_cmd))
                        {
                            res_face = mysql_store_result(conn_face);
                            if ((row_face = mysql_fetch_row(res_face)) && row_face[0])
                            {
                                face_id1 = row_face[0] ? atoi(row_face[0]) : 0;
                                face_id2 = row_face[1] ? atoi(row_face[1]) : 0;
                                face_id3 = row_face[2] ? atoi(row_face[2]) : 0;
                                ear = row_face[3] ? (float)atof(row_face[3]) : 0.0f;
                            }
                            mysql_free_result(res_face);
                        }
                        mysql_close(conn_face);
                    }
                    char ai_msg[128];
                    snprintf(ai_msg, sizeof(ai_msg), "SERVER:VALUE:%d:%d:%d:%.2f\n", face_id1, face_id2, face_id3, ear);
                    write(client_info->fd, ai_msg, strlen(ai_msg));
                    printf("[AI] 얼굴인식 데이터 전송: %s", ai_msg);
                    if (flag_check(name))
                        send_text_to_id("AI", name, "ATTENDANCE:OK");
                }
                else if (strcmp(req_type, "FLAG") == 0) // 출석 플래그 반환
                {
                    MYSQL *conn_flag = mysql_init(NULL);
                    MYSQL_RES *res_flag = NULL;
                    MYSQL_ROW row_flag = NULL;
                    char sql_cmd[512];
                    int face_verified = 0, rfid_verified = 0;

                    // name 정리(공백/CRLF 제거)
                    char name_buf[256];
                    snprintf(name_buf, sizeof(name_buf), "%s", name);
                    {
                        char *p = name_buf, *q = name_buf + strlen(name_buf);
                        while (p < q && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n'))
                            p++;
                        while (q > p && (q[-1] == ' ' || q[-1] == '\t' || q[-1] == '\r' || q[-1] == '\n'))
                            q--;
                        memmove(name_buf, p, (size_t)(q - p));
                        name_buf[q - p] = '\0';
                    }

                    if (!mysql_real_connect(conn_flag, DB_HOST, DB_USER, DB_PASS, DB_NAME, 0, NULL, 0))
                    {
                        fprintf(stderr, "[FLAG] connect err(%d): %s\n", mysql_errno(conn_flag), mysql_error(conn_flag));
                    }
                    else
                    {
                        char esc_name[512];
                        mysql_real_escape_string(conn_flag, esc_name, name_buf, strlen(name_buf));

                        snprintf(sql_cmd, sizeof(sql_cmd),
                                 "SELECT face_verified, rfid_verified "
                                 "FROM attendance "
                                 "WHERE name='%s' AND (`date`=CURDATE() OR DATE(checkin_time)=CURDATE()) "
                                 "ORDER BY id DESC LIMIT 1",
                                 esc_name);

                        if (mysql_query(conn_flag, sql_cmd))
                        {
                            fprintf(stderr, "[FLAG] query err(%d): %s\n", mysql_errno(conn_flag), mysql_error(conn_flag));
                        }
                        else if (!(res_flag = mysql_store_result(conn_flag)))
                        {
                            fprintf(stderr, "[FLAG] store_result err(%d): %s\n", mysql_errno(conn_flag), mysql_error(conn_flag));
                        }
                        else
                        {
                            unsigned long rows = mysql_num_rows(res_flag);
                            printf("[FLAG] rows=%lu for name='%s'\n", rows, esc_name);

                            if (rows > 0 && (row_flag = mysql_fetch_row(res_flag)))
                            {
                                // 문자열을 정수로 변환해서 사용
                                face_verified = row_flag[0] ? atoi(row_flag[0]) : 0;
                                rfid_verified = row_flag[1] ? atoi(row_flag[1]) : 0;
                                printf("[FLAG] fetched face=%d, rfid=%d\n", face_verified, rfid_verified);
                            }
                            else
                            {
                                fprintf(stderr, "[FLAG] no row fetched\n");
                            }
                            mysql_free_result(res_flag);
                        }
                        mysql_close(conn_flag);
                    }

                    // ※ 여기서 flag_check를 사용하고 싶다면, DB 결과 기반 후처리로만 사용
                    if (face_verified && rfid_verified)
                    {
                        printf("[FLAG] sent ATTENDANCE:OK (both=1)\n");
                    }

                    char ai_msg[64];
                    snprintf(ai_msg, sizeof(ai_msg), "SERVER:%s:FLAG:%d\n", name_buf, face_verified);
                    write(client_info->fd, ai_msg, strlen(ai_msg));
                }
            }

            // face_verified=1, checkin_time 기록 등 DB 처리
            if (i >= 4 && strcmp(pArray[0], "AI") == 0 && strcmp(pArray[2], "FACE") == 0 && strcmp(pArray[3], "OK") == 0)
            {
                int user_id = -1;
                MYSQL *conn3;
                MYSQL_RES *res3;
                MYSQL_ROW row3;
                char sql_cmd[256];

                // name -> user_id 변환
                conn3 = mysql_init(NULL);
                if (mysql_real_connect(conn3, DB_HOST, DB_USER, DB_PASS, DB_NAME, 0, NULL, 0))
                {
                    snprintf(sql_cmd, sizeof(sql_cmd), "SELECT emp_id FROM employees WHERE name='%s'", pArray[1]);
                    if (!mysql_query(conn3, sql_cmd))
                    {
                        res3 = mysql_store_result(conn3);
                        if ((row3 = mysql_fetch_row(res3)) && row3[0])
                            user_id = atoi(row3[0]);
                        mysql_free_result(res3);
                    }
                    // 오늘 출석 row 있는지 체크
                    snprintf(sql_cmd, sizeof(sql_cmd), "SELECT COUNT(*) FROM attendance WHERE name=\"%s\" AND date=CURDATE();;", pArray[1]);
                    int already_checked = 0;
                    if (!mysql_query(conn3, sql_cmd))
                    {
                        res3 = mysql_store_result(conn3);
                        if ((row3 = mysql_fetch_row(res3)) && atoi(row3[0]) > 0)
                            already_checked = 1;
                        mysql_free_result(res3);
                    }
                    if (!already_checked && user_id != -1)
                    {
                        // 없으면 insert (face_verified=1)
                        snprintf(sql_cmd, sizeof(sql_cmd),
                                 "INSERT INTO attendance (user_id, name, `date`, checkin_time, face_verified) "
                                 "VALUES (%d, \"%s\", CURDATE(), NOW(), 1);",
                                 user_id, pArray[1]);
                        if (mysql_query(conn3, sql_cmd))
                            fprintf(stderr, "INSERT error: %s\n", mysql_error(conn3));
                        else
                            printf("[DB] Attendance INSERT(face) for name=%s, user_id=%d\n", pArray[1], user_id);
                    }
                    else
                    {
                        // 이미 row 있으면 update만!
                        snprintf(sql_cmd, sizeof(sql_cmd),
                                 "UPDATE attendance SET face_verified=1 WHERE name=\"%s\" AND date=CURDATE();",
                                 pArray[1]);
                        if (mysql_query(conn3, sql_cmd))
                            fprintf(stderr, "UPDATE error: %s\n", mysql_error(conn3));
                        else
                            printf("[DB] face_verified=1 UPDATE for name=%s\n", pArray[1]);
                    }
                    // face_verified=1 UPDATE까지 끝난 뒤에
                    set_checkin_if_both_1(pArray[1]);
                    if (flag_check(pArray[1]))
                    {
                        send_text_to_id("AI", pArray[1], "ATTENDANCE:OK");
                    }
                    mysql_close(conn3);
                }
            }

            // ===== [AI:KMS:FACE:IN/OUT] 얼굴 인식 → 서랍 제어(UNLOCK/LOCK) =====
            else if (slp_flag == 0 && i >= 4 &&
                     strcmp(pArray[0], "AI") == 0 &&
                     strcmp(pArray[2], "FACE") == 0)
            {
                char uart_msg[64] = {0};
                time_t now = time(NULL);

                if (strcmp(pArray[3], "IN") == 0)
                {
                    // === IN을 OK처럼 동작 (방금 자리로 복귀)
                    if (face_state == 2 && no_time > 0)
                    {
                        double absence_sec = difftime(now, no_time);

                        // ----- [누적 absent_time DB 업데이트] -----
                        MYSQL *conn2 = mysql_init(NULL);
                        if (mysql_real_connect(conn2, DB_HOST, DB_USER, DB_PASS, DB_NAME, 0, NULL, 0))
                        {
                            MYSQL_RES *res2 = NULL;
                            MYSQL_ROW row2;
                            char sql_cmd[256], time_str[16];
                            char esc_name[256];

                            // name 이스케이프
                            mysql_real_escape_string(conn2, esc_name, pArray[1], strlen(pArray[1]));

                            // 오늘 absent_time 읽기 (없으면 0)
                            snprintf(sql_cmd, sizeof(sql_cmd),
                                     "SELECT absent_time FROM attendance "
                                     "WHERE name='%s' AND DATE(checkin_time)=CURDATE() "
                                     "ORDER BY id DESC LIMIT 1;",
                                     esc_name);

                            int db_sec = 0;
                            if (!mysql_query(conn2, sql_cmd))
                            {
                                res2 = mysql_store_result(conn2);
                                if (res2 && (row2 = mysql_fetch_row(res2)) && row2[0])
                                {
                                    int h, m, s;
                                    if (sscanf(row2[0], "%d:%d:%d", &h, &m, &s) == 3)
                                        db_sec = h * 3600 + m * 60 + s;
                                }
                                if (res2)
                                    mysql_free_result(res2);
                            }

                            int total_sec = db_sec + (int)absence_sec;
                            seconds_to_time_str(total_sec, time_str); // "HH:MM:SS"

                            // 오늘자 absent_time 누적 반영 (여러 행일 경우 전부 업데이트: 기존 로직 유지)
                            snprintf(sql_cmd, sizeof(sql_cmd),
                                     "UPDATE attendance SET absent_time='%s' "
                                     "WHERE name='%s' AND DATE(checkin_time)=CURDATE();",
                                     time_str, esc_name);
                            mysql_query(conn2, sql_cmd);
                            mysql_close(conn2);

                            int h = (int)(absence_sec / 3600);
                            int m = (int)((absence_sec - h * 3600) / 60);
                            int s = (int)(absence_sec - h * 3600 - m * 60);
                            printf("[DB] %s: absent_time 누적: %s (이번 비움 %d:%d:%d, 총초:%d)\n",
                                   pArray[1], time_str, h, m, s, total_sec);
                        }
                        // ----- [누적 absent_time DB 업데이트 끝] -----
                    }

                    ok_time = now;
                    face_state = 1;

                    // 서랍 해제
                    snprintf(uart_msg, sizeof(uart_msg), "SERVER:%s:DRAWER:UNLOCK\n", pArray[1]);
                    printf("서랍 해제 (UNLOCK)\n");
                }
                else if (strcmp(pArray[3], "OUT") == 0)
                {
                    // === OUT을 NO처럼 동작 (방금 자리에서 떠남)
                    if (face_state == 1 && ok_time > 0)
                    {
                        double present_sec = difftime(now, ok_time);
                        int h = (int)(present_sec / 3600);
                        int m = (int)((present_sec - h * 3600) / 60);
                        int s = (int)(present_sec - h * 3600 - m * 60);
                        printf("%s: %d시간 %d분 %d초 자리에 앉아있었음\n", pArray[1], h, m, s);
                        fflush(stdout);
                    }

                    no_time = now;
                    face_state = 2;

                    // 서랍 잠금
                    snprintf(uart_msg, sizeof(uart_msg), "SERVER:%s:DRAWER:LOCK\n", pArray[1]);
                    printf("서랍 잠금 (LOCK)\n");
                }

                if (uart_msg[0])
                {
                    uart_send(uart_msg, strlen(uart_msg));
                    printf("[UART] Sent: %s\n", uart_msg);
                }
            }

            // ===== [AI:KMS:POSTURE:BAD:neck] 등 자세감지 UART 전송 =====
            else if (slp_flag == 0 && i >= 4 && strcmp(pArray[0], "AI") == 0 && strcmp(pArray[2], "POSTURE") == 0)
            {
                char uart_msg[64] = {0};
                if (strcmp(pArray[3], "OK") == 0)
                {
                    snprintf(uart_msg, sizeof(uart_msg), "SERVER:%s:POSTURE:OK\n", pArray[1]);
                    printf("[UART] Sent: %s\n", uart_msg);
                    send_text_to_id("QT", pArray[1], "POSTURE|OK");
                }
                else if (strcmp(pArray[3], "BAD") == 0 && i == 5)
                {
                    snprintf(uart_msg, sizeof(uart_msg), "SERVER:%s:POSTURE:BAD:%s\n", pArray[1], pArray[4]);
                    printf("[UART] Sent: %s\n", uart_msg);
                    send_text_to_id("QT", pArray[1], "POSTURE|BAD|%s", pArray[4]);
                }
                if (uart_msg[0])
                {
                    uart_send(uart_msg, strlen(uart_msg));
                }
            }
            if (i >= 4 && strcmp(pArray[0], "QT") == 0 && strcmp(pArray[2], "ATT") == 0 && strcmp(pArray[3], "LIST") == 0)
            {
                const char *name = pArray[1];
                const char *from_ymd = (i >= 5) ? pArray[4] : NULL;
                const char *to_ymd = (i >= 6) ? pArray[5] : NULL;
                int limit_rows = (i >= 7) ? atoi(pArray[6]) : 200;

                notify_att_times_all("QT", name, from_ymd, to_ymd, limit_rows);
            }

            // ===== 메시지 브로드캐스트/로깅 (기존 로직) =====
            char *colon = strchr(raw_line, ':');
            if (colon)
            {
                *colon = '\0';
                const char *to = raw_line;       // 예: "AI"
                const char *payload = colon + 1; // 예: "seol:ATT:OK"

                char out[BUF_SIZE * 2];
                // ✨ 수신자에게 보이는 형태: SERVER:seol:ATT:OK
                snprintf(out, sizeof(out), "SERVER:%s\n", payload);

                MSG_INFO mi = {
                    .fd = -1,
                    .from = "SERVER",
                    .to = (char *)to,
                    .msg = out,
                    .len = (int)strlen(out),
                };
                send_msg(&mi, first_client_info);
            }

            line = strtok(NULL, "\n");
        }
    }

    close(client_info->fd);

    sprintf(strBuff, "Disconnect ID:%s (ip:%s,fd:%d,sockcnt:%d)\n", client_info->id, client_info->ip, client_info->fd, clnt_cnt - 1);
    log_file(strBuff);

    pthread_mutex_lock(&mutx);
    clnt_cnt--;
    client_info->fd = -1;
    pthread_mutex_unlock(&mutx);
    return NULL;
}