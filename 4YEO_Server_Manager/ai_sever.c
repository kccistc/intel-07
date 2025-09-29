/* �쒖슱湲곗닠援먯쑁�쇳꽣 IoT */
/* author : KSH */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>

#define BUF_SIZE 1024
#define MAX_CLNT 32
#define ID_SIZE 10
#define ARR_CNT 5

typedef struct {
    int fd;
    char* from;
    char* to;
    char* msg;
    int len;
} MSG_INFO;

typedef struct {
    int index;
    int fd;
    char ip[20];
    char id[ID_SIZE];
    char pw[ID_SIZE];
    char last_detected_class[BUF_SIZE];
    char last_detected_color[BUF_SIZE];
} CLIENT_INFO;

typedef struct {
    CLIENT_INFO* clients;
    int client_index;
} CLNT_ARGS;

void* clnt_connection(void* arg);
void send_msg(MSG_INFO* msg_info, CLIENT_INFO* first_client_info);
void error_handling(char* msg);
void log_file(char* msgstr);
void save_image_from_socket(int sock, const char* filename, long filesize);

int clnt_cnt = 0;
pthread_mutex_t mutx;

int main(int argc, char* argv[])
{
    int serv_sock, clnt_sock;
    struct sockaddr_in serv_adr, clnt_adr;
    int clnt_adr_sz;
    int sock_option = 1;
    pthread_t t_id[MAX_CLNT] = { 0 };
    char msg[BUF_SIZE];

    if (argc != 2)
    {
        printf("Usage: %s <port>\n", argv[0]);
        exit(1);
    }

    fputs("IoT Server Start!!\n", stdout);
    if (pthread_mutex_init(&mutx, NULL)) error_handling("mutex init error");

    serv_sock = socket(PF_INET, SOCK_STREAM, 0);
    if (serv_sock < 0) error_handling("socket() error");

    memset(&serv_adr, 0, sizeof(serv_adr));
    serv_adr.sin_family = AF_INET;
    serv_adr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_adr.sin_port = htons(atoi(argv[1]));

    setsockopt(serv_sock, SOL_SOCKET, SO_REUSEADDR, &sock_option, sizeof(sock_option));
    if (bind(serv_sock, (struct sockaddr*)&serv_adr, sizeof(serv_adr)) == -1) error_handling("bind() error");
    if (listen(serv_sock, 5) == -1) error_handling("listen() error");

    mkdir("received_images", 0777);

    CLIENT_INFO client_info[MAX_CLNT] = { 0 };
    for (int i = 0;i < MAX_CLNT;i++) client_info[i].fd = -1;

    while (1)
    {
        clnt_adr_sz = sizeof(clnt_adr);
        clnt_sock = accept(serv_sock, (struct sockaddr*)&clnt_adr, &clnt_adr_sz);
        if (clnt_sock < 0) { perror("accept()"); continue; }

        int str_len = read(clnt_sock, msg, sizeof(msg) - 1);
        if (str_len <= 0) { close(clnt_sock); continue; }
        msg[str_len] = 0;

        char login_id[ID_SIZE], login_pw[ID_SIZE];
        if (sscanf(msg, "[%[^:]:%[^]]]", login_id, login_pw) != 2)
        {
            write(clnt_sock, "Login format error\n", 18);
            close(clnt_sock);
            continue;
        }

        int idx = -1;
        for (int i = 0;i < MAX_CLNT;i++)
        {
            if (client_info[i].fd == -1) { idx = i; break; }
        }
        if (idx == -1) { write(clnt_sock, "Server full\n", 12); close(clnt_sock); continue; }

        CLIENT_INFO* ci = &client_info[idx];
        ci->fd = clnt_sock;
        ci->index = idx;
        strcpy(ci->id, login_id);
        strcpy(ci->pw, login_pw);
        strcpy(ci->ip, inet_ntoa(clnt_adr.sin_addr));

        pthread_mutex_lock(&mutx);
        clnt_cnt++;
        pthread_mutex_unlock(&mutx);

        sprintf(msg, "[%s] Connected! (ip:%s)\n", ci->id, ci->ip);
        write(clnt_sock, msg, strlen(msg));
        log_file(msg);

        CLNT_ARGS* args = (CLNT_ARGS*)malloc(sizeof(CLNT_ARGS));
        if (args == NULL) {
            perror("malloc error");
            close(clnt_sock);
            continue;
        }
        args->clients = client_info;
        args->client_index = idx;

        pthread_create(t_id + idx, NULL, clnt_connection, (void*)args);
        pthread_detach(t_id[idx]);
    }
    return 0;
}

void* clnt_connection(void* arg)
{
    CLNT_ARGS* args = (CLNT_ARGS*)arg;
    CLIENT_INFO* ci = &args->clients[args->client_index];
    char msg[BUF_SIZE];
    MSG_INFO msg_info;
    char strBuff[BUF_SIZE * 2];

    memset(ci->last_detected_class, 0, sizeof(ci->last_detected_class));
    memset(ci->last_detected_color, 0, sizeof(ci->last_detected_color));

    while (1)
    {
        memset(msg, 0, sizeof(msg));
        int str_len = recv(ci->fd, msg, sizeof(msg) - 1, MSG_PEEK);
        if (str_len <= 0) {
            sprintf(strBuff, "[%s] Disconnected! (ip:%s)\n", ci->id, ci->ip);
            log_file(strBuff);
            break;
        }

        // 1. TEXT 硫붿떆吏� 泥섎━
        if (strncmp(msg, "TEXT:", 5) == 0) {
            str_len = read(ci->fd, msg, sizeof(msg) - 1);
            msg[str_len] = '\0';
            char* text_content = msg + 5;

            // �� �섏젙: TEXT: �댄썑�� �댁슜�� "異붿쿇" �ㅼ썙�쒓� �덈뒗吏� 諛붾줈 �뺤씤�⑸땲��.
            if (strstr(text_content, "異붿쿇") != NULL && strlen(ci->last_detected_class) > 0) {
                log_file("AI: LLM script will be executed now..."); // �� 濡쒓렇 異붽�
                char python_cmd[BUF_SIZE * 2 + 100];
                char llm_result_buf[BUF_SIZE];
                FILE* fp_pipe;

                snprintf(python_cmd, sizeof(python_cmd), "python3 /home/ubuntu/intel_ai_project/ai_groq.py \"%s\" \"%s\"", ci->last_detected_color, ci->last_detected_class);

                fp_pipe = popen(python_cmd, "r");

                if (fp_pipe != NULL) {
                    if (fgets(llm_result_buf, sizeof(llm_result_buf), fp_pipe) != NULL) {
                        llm_result_buf[strcspn(llm_result_buf, "\n")] = '\0';
                        MSG_INFO llm_msg_info;
                        llm_msg_info.fd = -1;
                        llm_msg_info.from = "AI_Recommender";
                        llm_msg_info.to = ci->id;
                        llm_msg_info.msg = llm_result_buf;
                        llm_msg_info.len = strlen(llm_result_buf);
                        send_msg(&llm_msg_info, args->clients);
                    }
                    pclose(fp_pipe);
                }
                else {
                    log_file("Failed to run Groq recommender script.");
                }
            }
            else {
                // �� �섏젙: '異붿쿇' �ㅼ썙�쒓� �녾굅�� �ъ쟾 �뺣낫媛� �놁쓣 寃쎌슦 �덈궡 硫붿떆吏� 異쒕젰
                if (strstr(text_content, "異붿쿇") != NULL && strlen(ci->last_detected_class) == 0) {
                    sprintf(strBuff, "AI: Please send an image first to get a recommendation.");
                    MSG_INFO msg_info;
                    msg_info.fd = -1;
                    msg_info.from = "AI_Recommender";
                    msg_info.to = ci->id;
                    msg_info.msg = strBuff;
                    msg_info.len = strlen(strBuff);
                    send_msg(&msg_info, args->clients);
                }
                else {
                    // TEXT: �ㅻ뜑瑜� �ъ슜�덉�留� '異붿쿇' �ㅼ썙�쒓� �녿뒗 寃쎌슦
                    log_file("Received TEXT message but '異붿쿇' keyword was not found.");
                }
            }
            continue;
        }

        // 2. IMAGE 硫붿떆吏� 泥섎━
        else if (strncmp(msg, "IMAGE:", 6) == 0)
        {
            char fname[256];
            long filesize = 0;
            char header_buf[512];
            int header_len = 0;

            char* newline = strchr(msg, '\n');
            if (newline) {
                header_len = newline - msg;
                memcpy(header_buf, msg, header_len);
                header_buf[header_len] = '\0';

                recv(ci->fd, msg, header_len + 1, 0);

                if (sscanf(header_buf + 6, "%[^:]:%ld", fname, &filesize) == 2) {
                    char savepath[300];
                    sprintf(savepath, "received_images/%s_%ld_%s", ci->id, time(NULL), fname);

                    save_image_from_socket(ci->fd, savepath, filesize);
                    sprintf(strBuff, "[%s] Image saved: %s\n", ci->id, savepath);
                    log_file(strBuff);

                    char python_cmd[512];
                    char result_buf[BUF_SIZE];
                    FILE* fp_pipe;

                    sprintf(python_cmd, "python3 /home/ubuntu/intel_ai_project/vit_det/infer_add_color.py %s", savepath);
                    fp_pipe = popen(python_cmd, "r");

                    if (fp_pipe == NULL) {
                        log_file("Error: Failed to run python script.");
                    }
                    else {
                        if (fgets(result_buf, sizeof(result_buf), fp_pipe) != NULL) {
                            result_buf[strcspn(result_buf, "\n")] = '\0';

                            // �� �섏젙: �덉쟾�� sscanf瑜� �ъ슜�섏뿬 臾몄옄�댁쓣 �뚯떛
                            char color_str[BUF_SIZE], class_str[BUF_SIZE];
                            if (sscanf(result_buf, "%s %s�낅땲��.", color_str, class_str) == 2) {
                                strcpy(ci->last_detected_color, color_str);
                                strcpy(ci->last_detected_class, class_str);
                            }

                            MSG_INFO msg_info;
                            msg_info.fd = -1;
                            msg_info.from = "AI_Inference";
                            msg_info.to = ci->id;
                            msg_info.msg = result_buf;
                            msg_info.len = strlen(result_buf);
                            send_msg(&msg_info, args->clients);
                        }
                        pclose(fp_pipe);
                    }
                }
                else {
                    log_file("Image header format error");
                }
            }
            else {
                log_file("Waiting for full image header...");
                usleep(10000);
            }
            continue;
        }

        // 3. �쇰컲 梨꾪똿 硫붿떆吏� 泥섎━
        else {
            str_len = read(ci->fd, msg, sizeof(msg) - 1);
            if (str_len <= 0) {
                sprintf(strBuff, "[%s] Disconnected! (ip:%s)\n", ci->id, ci->ip);
                log_file(strBuff);
                break;
            }
            msg[str_len] = '\0';

            char to[ID_SIZE];
            char content[BUF_SIZE];
            if (sscanf(msg, "[%[^]]] %[^\n]", to, content) == 2) {
                // �� 硫붿떆吏� 濡쒓렇 異붽�: �쒕쾭 肄섏넄�� 硫붿떆吏�瑜� �쒖떆�⑸땲��.
                sprintf(strBuff, "[%s -> %s]: %s", ci->id, to, content);
                log_file(strBuff);

                msg_info.fd = ci->fd;
                msg_info.from = ci->id;
                msg_info.to = to;
                msg_info.msg = content;
                msg_info.len = strlen(content);
                send_msg(&msg_info, args->clients);
            }
            else {
                log_file("Received invalid chat message format.");
            }
        }
    }

    close(ci->fd);
    ci->fd = -1;
    memset(ci->id, 0, ID_SIZE);
    memset(ci->pw, 0, ID_SIZE);
    memset(ci->ip, 0, 20);

    pthread_mutex_lock(&mutx);
    clnt_cnt--;
    pthread_mutex_unlock(&mutx);
    free(args);
    return NULL;
}

void save_image_from_socket(int sock, const char* filename, long filesize)
{
    FILE* fp = fopen(filename, "wb");
    if (!fp) { perror("fopen"); return; }

    char buf[BUF_SIZE];
    long recv_bytes = 0;
    while (recv_bytes < filesize)
    {
        int read_size = (filesize - recv_bytes) > BUF_SIZE ? BUF_SIZE : (filesize - recv_bytes);
        int n = read(sock, buf, read_size);
        if (n <= 0) break;
        fwrite(buf, 1, n, fp);
        recv_bytes += n;
    }
    fclose(fp);
}

void send_msg(MSG_INFO* msg_info, CLIENT_INFO* first_client_info)
{
    pthread_mutex_lock(&mutx);
    for (int i = 0; i < MAX_CLNT; i++) {
        CLIENT_INFO* ci = &first_client_info[i];
        if (ci->fd != -1 && ci->fd != msg_info->fd) {
            if (!strcmp(msg_info->to, "ALL")) {
                char final_msg[BUF_SIZE];
                sprintf(final_msg, "[%s]: %s\n", msg_info->from, msg_info->msg);
                if (write(ci->fd, final_msg, strlen(final_msg)) < 0) {
                    perror("write error");
                }
            }
            else if (!strcmp(ci->id, msg_info->to)) {
                char final_msg[BUF_SIZE];
                sprintf(final_msg, "[%s(DM)]: %s\n", msg_info->from, msg_info->msg);
                if (write(ci->fd, final_msg, strlen(final_msg)) < 0) {
                    perror("write error");
                }
                break;
            }
        }
    }
    pthread_mutex_unlock(&mutx);
}

void log_file(char* msgstr) { fputs(msgstr, stdout); fputc('\n', stdout); }

void error_handling(char* msg) { perror(msg); exit(1); }

/*
ubuntu@ubuntu19:~/intel_ai_project$ python3 /home/ubuntu/intel_ai_project/ai_groq.py "寃��뺤깋" "Jeans"
API Error: Error code: 404 - {'error': {'message': 'The model `llama-3.1-70b-8192` does not exist or you do not have access to it.', 'type': 'invalid_request_error', 'code': 'model_not_found'}}
*/