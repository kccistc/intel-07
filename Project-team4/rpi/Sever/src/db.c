#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mysql/mysql.h>
#include "config.h"
#include "types.h"
#include "db.h"

static MYSQL *db_connect(void)
{
    MYSQL *c = mysql_init(NULL);
    if (!mysql_real_connect(c, DB_HOST, DB_USER, DB_PASS, DB_NAME, 0, NULL, 0))
    {
        fprintf(stderr, "DB connect err: %s\n", mysql_error(c));
        return NULL;
    }
    return c;
}

// 오늘 날짜로 서버를 실행시 오늘 날짜로 출석 row를 자동 생성
void insert_attendance_today(void)
{
    MYSQL *conn = db_connect();
    if (!conn)
        return;

    if (!mysql_query(conn, "SELECT emp_id, name FROM employees"))
    {
        MYSQL_RES *res = mysql_store_result(conn);
        MYSQL_ROW row;
        while ((row = mysql_fetch_row(res)))
        {
            int user_id = atoi(row[0]);
            const char *name = row[1];

            char check_sql[256];
            snprintf(check_sql, sizeof(check_sql),
                     "SELECT COUNT(*) FROM attendance WHERE user_id=%d AND date=CURDATE();",
                     user_id);

            int already = 0;
            if (!mysql_query(conn, check_sql))
            {
                MYSQL_RES *cres = mysql_store_result(conn);
                MYSQL_ROW crow = mysql_fetch_row(cres);
                if (crow && crow[0])
                    already = atoi(crow[0]);
                mysql_free_result(cres);
            }
            if (!already)
            {
                // 오늘 출석 row 없으면 새로 생성
                char insert_sql[512];
                snprintf(insert_sql, sizeof(insert_sql),
                         "INSERT INTO attendance (user_id, name, date, checkin_time) "
                         "VALUES (%d, '%s', CURDATE(), NULL);",
                         user_id, name);
                if (mysql_query(conn, insert_sql))
                    fprintf(stderr, "Attendance INSERT fail: %s\n", mysql_error(conn));
                else
                    printf("[DB] [ONCE] INSERT: %s(%d) today\n", name, user_id);
            }
        }
        mysql_free_result(res);
    }
    mysql_close(conn);
}

static void escape_name(MYSQL *c, const char *in, char *out, size_t outsz)
{
    // SQL 인젝션 방지용 이스케이프(최대 크기 넘을까봐 마지막 널문자 수동으로 넣음)
    mysql_real_escape_string(c, out, in, strlen(in));
    out[outsz - 1] = '\0';
}

// 둘 다 1일 때, checkin_time 없으면 찍어주기 (FACE:OK 처리 후에도 쓰려고 분리)
void set_checkin_if_both_1(const char *name)
{
    MYSQL *conn = db_connect();
    if (!conn)
        return;

    char esc[ID_SIZE * 2 + 1];
    escape_name(conn, name, esc, sizeof(esc));

    char sql[512];
    snprintf(sql, sizeof(sql),
             "UPDATE attendance SET checkin_time=NOW() "
             "WHERE name='%s' AND date=CURDATE() "
             "AND face_verified=1 AND rfid_verified=1 "
             "AND checkin_time IS NULL ORDER BY id DESC LIMIT 1",
             esc);

    if (mysql_query(conn, sql) != 0)
        fprintf(stderr, "UPDATE(checkin) err: %s\n", mysql_error(conn));

    mysql_close(conn);
}

// return: 2=checkout 찍음, 1=checkin(IN) 처리, 0=아무것도 안 함, -1=에러
int rfid_toggle_and_update(const char *name)
{
    int ret = 0;
    MYSQL *conn = db_connect();
    if (!conn)
        return -1;

    char esc[ID_SIZE * 2 + 1];
    escape_name(conn, name, esc, sizeof(esc));

    // 1) IN 처리 시도: (rfid_verified=0 -> 1), checkout_time IS NULL 인 최신 1건
    //    + face_verified=1 & checkin_time IS NULL 이면 checkin_time=NOW()
    char sql_in[1024];
    snprintf(sql_in, sizeof(sql_in),
             "UPDATE attendance SET rfid_verified=1, "
             "checkin_time = CASE WHEN face_verified=1 AND checkin_time IS NULL THEN NOW() ELSE checkin_time END "
             "WHERE name='%s' AND date=CURDATE() "
             "AND checkout_time IS NULL AND rfid_verified=0 ORDER BY id DESC LIMIT 1",
             esc);

    if (mysql_query(conn, sql_in) != 0)
    {
        fprintf(stderr, "UPDATE(IN) err: %s\n", mysql_error(conn));
        mysql_close(conn);
        return -1;
    }
    if (mysql_affected_rows(conn) == 1)
    {
        mysql_close(conn);
        return 1;
    }

    // 2) OUT 처리 시도: (rfid_verified=1 -> 0) + checkout_time 없으면 NOW()
    char sql_out[512];
    snprintf(sql_out, sizeof(sql_out),
             "UPDATE attendance SET rfid_verified=0, "
             "checkout_time = CASE WHEN checkout_time IS NULL THEN NOW() ELSE checkout_time END "
             "WHERE name='%s' AND date=CURDATE() AND rfid_verified=1 AND checkout_time IS NULL "
             "ORDER BY id DESC LIMIT 1",
             esc);

    if (mysql_query(conn, sql_out) != 0)
    {
        fprintf(stderr, "UPDATE(OUT) err: %s\n", mysql_error(conn));
        mysql_close(conn);
        return -1;
    }
    if (mysql_affected_rows(conn) == 1)
        ret = 2;

    mysql_close(conn);
    return ret;
}

int flag_check(const char *name)
{
    int face_verified = 0, rfid_verified = 0;
    MYSQL *conn = db_connect();
    if (!conn)
        return 0;

    char esc[ID_SIZE * 2 + 1];
    mysql_real_escape_string(conn, esc, name, strlen(name));

    char sql[512];
    // ⚠️ 공백들 꼭 넣기!
    snprintf(sql, sizeof(sql),
             "SELECT face_verified, rfid_verified FROM attendance "
             "WHERE name='%s' AND date=CURDATE() ORDER BY id DESC LIMIT 1",
             esc);

    if (mysql_query(conn, sql) == 0)
    {
        MYSQL_RES *res = mysql_store_result(conn);
        if (res)
        {
            MYSQL_ROW row = mysql_fetch_row(res);
            if (row)
            {
                face_verified = row[0] ? atoi(row[0]) : 0;
                rfid_verified = row[1] ? atoi(row[1]) : 0;
            }
            mysql_free_result(res);
        }
    }

    mysql_close(conn);
    printf("FLAG face : %d, rfid : %d\n", face_verified, rfid_verified);

    return (face_verified == 1 && rfid_verified == 1) ? 1 : 0;
}

void notify_att_times_all(const char *to, const char *name,
                          const char *from_ymd, const char *to_ymd,
                          int limit_rows)
{
    MYSQL *conn = mysql_init(NULL);
    if (!mysql_real_connect(conn, "localhost", "iot", "pwiot", "memberdb", 0, NULL, 0))
    {
        fprintf(stderr, "[notify_att_times_all] DB connect err: %s\n", mysql_error(conn));
        return;
    }

    // name escape
    char esc_name[ID_SIZE * 2 + 1];
    mysql_real_escape_string(conn, esc_name, name, strlen(name));

    // limit 보정
    if (limit_rows <= 0)
        limit_rows = 200;
    if (limit_rows > 2000)
        limit_rows = 2000;

    // absent_time(VARCHAR) 정규화 식:
    // - '' -> NULL 처리
    // - ^[0-9]+$  (숫자: 분) -> SEC_TO_TIME(분*60)
    // - ^HH:MM(:SS)?$ -> STR_TO_DATE 후 %H:%i:%s
    // - 그 외 -> 원문 (마지막 안전장치; 필요 시 NULL로 바꿔도 됨)
    const char *abs_norm =
        "COALESCE("
        "  CASE"
        "    WHEN NULLIF(absent_time,'') IS NULL THEN NULL"
        "    WHEN absent_time REGEXP '^[0-9]+$' THEN DATE_FORMAT(SEC_TO_TIME(CAST(absent_time AS UNSIGNED)*60),'%H:%i:%s')"
        "    WHEN absent_time REGEXP '^[0-2][0-9]:[0-5][0-9](:[0-5][0-9])?$' THEN "
        "         DATE_FORMAT(STR_TO_DATE(IF(LENGTH(absent_time)=5, CONCAT(absent_time,':00'), absent_time),'%H:%i:%s'),'%H:%i:%s')"
        "    ELSE absent_time"
        "  END,"
        "  'NULL'"
        ")";

    char sql[1024];
    if (from_ymd && to_ymd && *from_ymd && *to_ymd)
    {
        snprintf(sql, sizeof(sql),
                 "SELECT date,"
                 " COALESCE(DATE_FORMAT(checkin_time,'%%Y-%%m-%%d %%H:%%i:%%s'),'NULL'),"
                 " COALESCE(DATE_FORMAT(checkout_time,'%%Y-%%m-%%d %%H:%%i:%%s'),'NULL'),"
                 " %s "
                 "FROM attendance "
                 "WHERE name='%s' AND date BETWEEN '%s' AND '%s' "
                 "ORDER BY date DESC, id DESC "
                 "LIMIT %d",
                 abs_norm, esc_name, from_ymd, to_ymd, limit_rows);
    }
    else
    {
        snprintf(sql, sizeof(sql),
                 "SELECT date,"
                 " COALESCE(DATE_FORMAT(checkin_time,'%%Y-%%m-%%d %%H:%%i:%%s'),'NULL'),"
                 " COALESCE(DATE_FORMAT(checkout_time,'%%Y-%%m-%%d %%H:%%i:%%s'),'NULL'),"
                 " %s "
                 "FROM attendance "
                 "WHERE name='%s' "
                 "ORDER BY date DESC, id DESC "
                 "LIMIT %d",
                 abs_norm, esc_name, limit_rows);
    }

    if (mysql_query(conn, sql) != 0)
    {
        fprintf(stderr, "[notify_att_times_all] query err: %s\n", mysql_error(conn));
        mysql_close(conn);
        return;
    }

    MYSQL_RES *res = mysql_store_result(conn);
    if (!res)
    {
        fprintf(stderr, "[notify_att_times_all] store_result err: %s\n", mysql_error(conn));
        mysql_close(conn);
        return;
    }

    MYSQL_ROW row;
    while ((row = mysql_fetch_row(res)))
    {
        const char *d = row[0] ? row[0] : "NULL";
        const char *cin = row[1] ? row[1] : "NULL";
        const char *cout = row[2] ? row[2] : "NULL";
        const char *abs = row[3] ? row[3] : "NULL";
        // absent_time 추가 (정규화된 HH:MM:SS 또는 'NULL')
        send_text_to_id(to, name, "ATT|ITEM|%s|%s|%s|%s", d, cin, cout, abs);
    }
    mysql_free_result(res);
    mysql_close(conn);
}
