#pragma once


// 오늘 날짜로 서버를 실행시 오늘 날짜로 출석 row를 자동 생성
void insert_attendance_today(void);


// 둘 다 1일 때, checkin_time 없으면 찍어주기 (FACE:OK 처리 후에도 쓰려고 분리)
void set_checkin_if_both_1(const char *name);


// return: 2=checkout 찍음, 1=checkin(IN) 처리, 0=아무것도 안 함, -1=에러
int rfid_toggle_and_update(const char *name);


int flag_check(const char *name);

void notify_att_times_all(const char *to, const char *name,
                                 const char *from_ymd, const char *to_ymd,
                                 int limit_rows);