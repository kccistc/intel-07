#pragma once
#include <stddef.h>
#include <pthread.h>
#include "config.h"


typedef struct {
int fd;
char *from;
char *to;
char *msg;
int len;
} MSG_INFO;


typedef struct {
int index;
int fd;
char ip[20];
char id[ID_SIZE];
char pw[ID_SIZE];
} CLIENT_INFO;


typedef struct {
unsigned char buf[UART_TXQ_CAP];
size_t head, tail, sz;
pthread_mutex_t lock;
} uart_txq_t;


typedef struct {
unsigned char buf[UART_RXQ_CAP];
size_t head, tail, sz;
pthread_mutex_t lock;
} uart_rxq_t;