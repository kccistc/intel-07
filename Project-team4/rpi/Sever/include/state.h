#pragma once
#include <time.h>
#include <pthread.h>
#include "types.h"


// 여러 .c에서 공유할 전역 상태 extern (정의는 main.c)
extern int clnt_cnt;
extern int slp_flag;
extern int face_state;
extern time_t ok_time;
extern time_t no_time;


extern CLIENT_INFO *g_clients;
extern pthread_mutex_t mutx;


extern int uart_fd;
extern uart_txq_t uart_txq;
extern uart_rxq_t uart_rxq;