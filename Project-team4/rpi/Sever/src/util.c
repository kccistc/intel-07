#include <stdio.h>
#include <stdlib.h>
#include "util.h"

void error_handling(const char *msg)
{
    fputs(msg, stderr);
    fputc('\n', stderr);
    exit(1);
}

void log_file(const char *msgstr)
{
    fputs(msgstr, stdout);
}

// 누적초 → "hh:mm:ss" 문자열
void seconds_to_time_str(int total_sec, char *out)
{
    int h = total_sec / 3600;
    int m = (total_sec % 3600) / 60;
    int s = total_sec % 60;
    sprintf(out, "%02d:%02d:%02d", h, m, s);
}