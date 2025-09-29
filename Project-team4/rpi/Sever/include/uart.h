#pragma once
#include <stddef.h>


int uart_open_nonblock(const char *dev); // fd 반환
void uart_queues_init(void); // tx/rx 큐 초기화
void *poll_uart_thread(void *arg); // pthread 전용
void uart_send(const char *msg, size_t len); // 송신 큐에 push