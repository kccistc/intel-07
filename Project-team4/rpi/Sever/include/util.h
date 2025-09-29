#pragma once
void error_handling(const char *msg);
void log_file(const char *msgstr);
void seconds_to_time_str(int total_sec, char *out); // "hh:mm:ss"