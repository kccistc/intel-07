#include <QMessageBox>
#include <QTime>
#include "attendancetable.h"
#include "ui_attendancetable.h"

AttendanceTable::AttendanceTable(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::AttendanceTable)
{
    ui->setupUi(this);
}

AttendanceTable::~AttendanceTable()
{
    delete ui;
}

//삭제버튼
void AttendanceTable::on_pPBdeleteDB_clicked()
{
    // 모든 행을 한 번에 삭제 (allRows 신경 쓸 필요 없음)
    ui->tableWidget->setRowCount(0);
}


//조회버튼
void AttendanceTable::on_pbSearch_clicked()
{
    QString name = ui->leName->text().trimmed();
    if (name.isEmpty()) {
        QMessageBox::warning(this, "경고", "이름을 입력하세요!");
        return;
    }

    // 서버로 보낼 명령 (기간 무시, 요구한 명령 그대로)
    QString cmd = QString("QT:%1:ATT:LIST\n").arg(name);

    // MainWidget에 신호로 전달 (서버로 보내달라고 요청)
    emit requestSendToServer(cmd);

    // 테이블 내용 초기화 (기존 데이터 지우기)
    ui->tableWidget->setRowCount(0);
    allRows.clear();  // ★ 추가: 내부 저장도 비움
}

void AttendanceTable::appendAttendanceRow(const QString &name,
                                          const QString &checkIn, const QString &checkOut, const QString &awayTime, const QString &sleepTime)
{

    // 순수업무시간 기본값: 빈 문자열
    QString pureWorkTime = "";

    // 출근, 퇴근 모두 값이 있을 때만 계산
    if (!checkIn.isEmpty() && !checkOut.isEmpty()) {
        QString ci = checkIn;
        QString co = checkOut;

        // 만약 "YYYY-MM-DD HH:mm:ss" 형식이면, 시간 부분만 추출
        if (ci.length() > 8) ci = ci.right(8);
        if (co.length() > 8) co = co.right(8);

        QTime inTime = QTime::fromString(ci, "HH:mm:ss");
        QTime outTime = QTime::fromString(co, "HH:mm:ss");

        if (inTime.isValid() && outTime.isValid()) {
            int totalSec = inTime.secsTo(outTime);

            QTime away = QTime::fromString(awayTime, "HH:mm:ss");
            QTime sleep = QTime::fromString(sleepTime, "HH:mm:ss");
            int awaySec = away.isValid() ? (away.hour()*3600 + away.minute()*60 + away.second()) : 0;
            int sleepSec = sleep.isValid() ? (sleep.hour()*3600 + sleep.minute()*60 + sleep.second()) : 0;

            int pureSec = totalSec - awaySec - sleepSec;
            if (pureSec < 0) pureSec = 0;

            int h = pureSec / 3600;
            int m = (pureSec % 3600) / 60;
            int s = pureSec % 60;
            pureWorkTime = QString("%1:%2:%3")
                               .arg(h, 2, 10, QChar('0'))
                               .arg(m, 2, 10, QChar('0'))
                               .arg(s, 2, 10, QChar('0'));
        }
    }
    QStringList rowData{ name, checkIn, checkOut, awayTime, sleepTime};
    allRows.append(rowData);

    int row = ui->tableWidget->rowCount();
    ui->tableWidget->insertRow(row);
    ui->tableWidget->setItem(row, 0, new QTableWidgetItem(name));
    ui->tableWidget->setItem(row, 1, new QTableWidgetItem(checkIn));
    ui->tableWidget->setItem(row, 2, new QTableWidgetItem(checkOut));
    ui->tableWidget->setItem(row, 3, new QTableWidgetItem(awayTime));
    ui->tableWidget->setItem(row, 4, new QTableWidgetItem(sleepTime));
    ui->tableWidget->setItem(row, 5, new QTableWidgetItem(pureWorkTime));
}




//평균 출퇴근,자리비움시간, 졸음시간, 순수 업무시간 계산
void AttendanceTable::showAverageRow()
{
    if (allRows.isEmpty()) return;

    // 자리비움 시간 평균 구하기
    int totalAwaySeconds = 0;
    int awayCount = 0;
    int checkInSum = 0, checkOutSum = 0, checkInCnt = 0, checkOutCnt = 0;

    for (const QStringList &row : allRows) {
        // 자리비움 시간 평균 계산 ("hh:mm:ss" 문자열)
        QTime awayTime = QTime::fromString(row[3], "hh:mm:ss");
        if (awayTime.isValid()) {
            totalAwaySeconds += QTime(0,0,0).secsTo(awayTime);
            awayCount++;
        }

        // 출근시간 평균 계산
        QTime inTime = QTime::fromString(row[1].mid(11), "hh:mm:ss");
        if (inTime.isValid()) {
            checkInSum += QTime(0,0,0).secsTo(inTime);
            checkInCnt++;
        }

        // 퇴근시간 평균 계산
        QTime outTime = QTime::fromString(row[2].mid(11), "hh:mm:ss");
        if (outTime.isValid()) {
            checkOutSum += QTime(0,0,0).secsTo(outTime);
            checkOutCnt++;
        }
    }

    // 평균값
    QString avgAway = "00:00:00";
    if (awayCount > 0)
        avgAway = QTime(0,0,0).addSecs(totalAwaySeconds / awayCount).toString("hh:mm:ss");

    QString avgIn = "";
    if (checkInCnt > 0)
        avgIn = QTime(0,0,0).addSecs(checkInSum / checkInCnt).toString("hh:mm:ss");

    QString avgOut = "";
    if (checkOutCnt > 0)
        avgOut = QTime(0,0,0).addSecs(checkOutSum / checkOutCnt).toString("hh:mm:ss");

    // 기존 평균 row 있으면 삭제
    if (ui->tableWidget->rowCount() > allRows.size()) {
        ui->tableWidget->removeRow(0);
    }

    // 평균 행 추가 (맨 위)
    ui->tableWidget->insertRow(0);
    ui->tableWidget->setItem(0, 0, new QTableWidgetItem("평균"));
    ui->tableWidget->setItem(0, 1, new QTableWidgetItem(avgIn));
    ui->tableWidget->setItem(0, 2, new QTableWidgetItem(avgOut));
    ui->tableWidget->setItem(0, 3, new QTableWidgetItem(avgAway));
}

//날짜 지정 조회
QDateTime AttendanceTable::getDateFrom() const {
    return ui->dateFrom->dateTime();
}
QDateTime AttendanceTable::getDateTo() const {
    return ui->dateTo->dateTime();
}


void AttendanceTable::updateRecvDataSlot(QString line)
{
    qDebug() << line;
    line = line.trimmed(); // \r, \n, 공백 제거(끝)

    QStringList parts = line.split("|");    // | 기준으로 분리

    if(parts.size() < 4)
    {
        qDebug() << "파싱 오류 : 토큰 부족" << parts;
        return;
    }

    QString currentUser = parts[1];     // 새로 들어온 사용자 이름

    // ✅ RFID
    if (parts[2] == "RFID") {
        if (parts[3] == "NO") {
            QString msg = QString("[%1] RFID를 태그하지 않았습니다!").arg(currentUser);
            warningManager->showWarning(warningManager->rfidBoxes, currentUser,
                                        "출근 인식 경고", msg, warningManager->rfidList, QMessageBox::Critical);
        }
        else if (parts[3] == "YES") {
            if (warningManager->rfidBoxes.contains(currentUser)) {
                warningManager->rfidBoxes[currentUser]->close();    // 경고창 닫기
                warningManager->rfidBoxes.remove(currentUser);      // Map에서 key값 삭제
            }
            warningManager->rfidList.removeAll(currentUser); // 리스트에서 제거
        }
    }

    // ✅ AWAY
    if (parts[2] == "AWAY") {
        if (parts[3] == "ON") {
            QString msg = QString("[%1] 자리 비움 시간 5분 경과").arg(currentUser);
            warningManager->showWarning(warningManager->awayBoxes, currentUser,
                                        "자리 비움 경고", msg, warningManager->awayList);
        }
        else if (parts[3] == "OFF") {
            // 창은 닫지 않고 유지
            warningManager->awayList.removeAll(currentUser); // 리스트에서만 제거
        }
    }
}
