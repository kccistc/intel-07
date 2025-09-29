#pragma once
#include "types.h"


void *clnt_connection(void *arg);
void send_msg(MSG_INFO *msg_info, CLIENT_INFO *first_client_info);


// 특정 ID에게 텍스트 전달 (라인 끝에 \n 포함해서 보냄)
void send_text_to_id(const char *to, const char *from, const char *fmt, ...);