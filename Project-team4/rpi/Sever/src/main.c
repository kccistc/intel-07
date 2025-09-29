#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <pthread.h>
#include <mysql/mysql.h>
#include "config.h"
#include "types.h"
#include "state.h"
#include "util.h"
#include "db.h"
#include "uart.h"
#include "server.h"

int clnt_cnt = 0;
int slp_flag = 0;
int face_state = 0;
time_t ok_time = 0;
time_t no_time = 0;
CLIENT_INFO *g_clients = NULL;
pthread_mutex_t mutx;

int uart_fd = -1;
uart_txq_t uart_txq;
uart_rxq_t uart_rxq;

int main(int argc, char *argv[])
{
    int serv_sock, clnt_sock;
    struct sockaddr_in serv_adr, clnt_adr;
    socklen_t clnt_adr_sz;
    int sock_option = 1;
    pthread_t t_id[MAX_CLNT] = {0};
    int str_len = 0;
    int i = 0;
    char idpasswd[(ID_SIZE * 2) + 3];
    char *pToken;
    char *pArray[ARR_CNT] = {0};
    char msg[BUF_SIZE];

    if (argc != 2)
    {
        printf("Usage : %s <port>\n", argv[0]);
        exit(1);
    }

    fputs("IoT Server Start!!\n", stdout);

    if (pthread_mutex_init(&mutx, NULL))
        error_handling("mutex init error");

    // 오늘 날짜로 서버를 실행시 오늘 날짜로 출석 row를 자동 생성
    insert_attendance_today();

    // ------------------ DB에서 username/password 불러오기 ------------------
    CLIENT_INFO *client_info = (CLIENT_INFO *)calloc(MAX_CLNT, sizeof(CLIENT_INFO));
    if (client_info == NULL)
    {
        perror("calloc()");
        exit(1);
    }
    g_clients = client_info; // 전역 포인터 연결

    MYSQL *conn;
    MYSQL_RES *res;
    MYSQL_ROW row;

    conn = mysql_init(NULL);
    if (!mysql_real_connect(conn, DB_HOST, DB_USER, DB_PASS, DB_NAME, 0, NULL, 0))
    {
        fprintf(stderr, "DB Connection failed: %s\n", mysql_error(conn));
        exit(1);
    }
    if (mysql_query(conn, "SELECT username, password FROM users"))
    {
        fprintf(stderr, "SELECT error: %s\n", mysql_error(conn));
        exit(1);
    }
    res = mysql_store_result(conn);
    int idx = 0;
    while ((row = mysql_fetch_row(res)) && idx < MAX_CLNT)
    {
        client_info[idx].fd = -1;
        strncpy(client_info[idx].id, row[0], ID_SIZE - 1);
        client_info[idx].id[ID_SIZE - 1] = 0;
        strncpy(client_info[idx].pw, row[1], ID_SIZE - 1);
        client_info[idx].pw[ID_SIZE - 1] = 0;
        idx++;
    }
    mysql_free_result(res);
    mysql_close(conn);
    // -----------------------------------------------------------------------

    // uart open
    uart_fd = uart_open_nonblock(UART_DEV);
    if (uart_fd < 0)
        exit(1);
    printf("UART device opened: %s (fd=%d)\n", UART_DEV, uart_fd);

    // === poll 기반 큐/락 초기화 (한번만!) ===
    uart_queues_init();

    // poll_uart_thread 스레드 시작
    pthread_t uart_poll_tid;
    pthread_create(&uart_poll_tid, NULL, poll_uart_thread, NULL);

    // --- 기존 socket()~bind()~listen() 코드 ---
    serv_sock = socket(PF_INET, SOCK_STREAM, 0);

    memset(&serv_adr, 0, sizeof(serv_adr));
    serv_adr.sin_family = AF_INET;
    serv_adr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_adr.sin_port = htons(atoi(argv[1]));

    setsockopt(serv_sock, SOL_SOCKET, SO_REUSEADDR, (void *)&sock_option, sizeof(sock_option));
    if (bind(serv_sock, (struct sockaddr *)&serv_adr, sizeof(serv_adr)) == -1)
        error_handling("bind() error");
    if (listen(serv_sock, 5) == -1)
        error_handling("listen() error");

    while (1)
    {
        clnt_adr_sz = sizeof(clnt_adr);
        clnt_sock = accept(serv_sock, (struct sockaddr *)&clnt_adr, &clnt_adr_sz);
        if (clnt_cnt >= MAX_CLNT)
        {
            printf("socket full\n");
            shutdown(clnt_sock, SHUT_WR);
            continue;
        }
        else if (clnt_sock < 0)
        {
            perror("accept()");
            continue;
        }

        str_len = read(clnt_sock, idpasswd, sizeof(idpasswd));
        idpasswd[str_len] = '\0';

        if (str_len > 0)
        {
            i = 0;
            pToken = strtok(idpasswd, "[:]");
            while (pToken != NULL)
            {
                pArray[i] = pToken;
                if (i++ >= ARR_CNT)
                    break;
                pToken = strtok(NULL, "[:]");
            }
            for (i = 0; i < MAX_CLNT; i++)
            {
                if (!strcmp(client_info[i].id, pArray[0]))
                {
                    if (client_info[i].fd != -1)
                    {
                        sprintf(msg, "[%s] Already logged!\n", pArray[0]);
                        write(clnt_sock, msg, strlen(msg));
                        log_file(msg);
                        shutdown(clnt_sock, SHUT_WR);
#if 1 // for MCU
                        client_info[i].fd = -1;
#endif
                        break;
                    }
                    if (!strcmp(client_info[i].pw, pArray[1]))
                    {
                        strcpy(client_info[i].ip, inet_ntoa(clnt_adr.sin_addr));
                        pthread_mutex_lock(&mutx);
                        client_info[i].index = i;
                        client_info[i].fd = clnt_sock;
                        clnt_cnt++;
                        pthread_mutex_unlock(&mutx);
                        sprintf(msg, "[%s] New connected! (ip:%s,fd:%d,sockcnt:%d)\n",
                                pArray[0], inet_ntoa(clnt_adr.sin_addr), clnt_sock, clnt_cnt);
                        log_file(msg);
                        write(clnt_sock, msg, strlen(msg));

                        pthread_create(t_id + i, NULL, clnt_connection, (void *)(client_info + i));
                        pthread_detach(t_id[i]);
                        break;
                    }
                }
            }
            if (i == MAX_CLNT)
            {
                sprintf(msg, "[%s] Authentication Error!\n", pArray[0]);
                write(clnt_sock, msg, strlen(msg));
                log_file(msg);
                shutdown(clnt_sock, SHUT_WR);
            }
        }
        else
            shutdown(clnt_sock, SHUT_WR);
    }

    return 0;
}