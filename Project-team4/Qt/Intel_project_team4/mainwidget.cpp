#include "mainwidget.h"
#include "ui_mainwidget.h"
#include <QDateTime>  // 날짜/시간 처리

MainWidget::MainWidget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::MainWidget)
{
    ui->setupUi(this);

    qDebug() << "Start";

    mjpgview1 = new MjpgView(QUrl("http://192.168.0.122:8081/processed"), this);
    mjpgview2 = new MjpgView(QUrl("http://192.168.0.111:8081/video"), this);

    // (1) 레이아웃 생성
    QVBoxLayout* vlayout = new QVBoxLayout();
    vlayout->addWidget(mjpgview1);
    vlayout->addWidget(mjpgview2);

    // (2) 탭에 레이아웃 연결 (딱 1번만!)
    ui->tab->setLayout(vlayout);

    attendanceTable = new AttendanceTable();
    ui->tab_2->setLayout(attendanceTable->layout());

    sockclient = new SockClient(this);
    mjpgview1->play();
    mjpgview2->play();

    connect(sockclient, SIGNAL(socketRevcDataSig(QString)), this, SLOT(updateRecvDataSlot(QString)));

    connect(attendanceTable, &AttendanceTable::requestSendToServer, this, [this](const QString &cmd)
            {
                sockclient->socketWriteDataSlot(cmd); // sockclient에서 서버로 명령 전송
            });
}

MainWidget::~MainWidget()
{
    delete ui;
}

void MainWidget::updateRecvDataSlot(QString line)
{
    qDebug() << line;
    line = line.trimmed(); // \r, \n, 공백 제거(끝)

    //자세 AI:이름:POSTURE:BAD:leg.neck.back 들어왔을 때 자세경고 출력문
    QStringList parts = line.split("|");
    if (parts.size() >= 4 && parts[2] == "POSTURE") {
        if (parts[3] == "OK") {
            ui->lblPostureStatus->setText("정상 자세입니다.");
            ui->lblPostureStatus->setStyleSheet("color: #34c759; font-weight: bold;"); // 초록색
        }
        else if (parts[3] == "BAD" && parts.size() >= 5) {
            QString badPart = parts[4];
            QString msg = (badPart == "neck" ? "목" : badPart == "back" ? "허리" : badPart == "leg" ? "다리" : badPart)
                          + " 자세를 고쳐앉으세요!!!";
            ui->lblPostureStatus->setText(msg);
            ui->lblPostureStatus->setStyleSheet("color: #ff3b30; font-weight: bold;"); // 빨간색
        }
    }

    if (line.startsWith("SERVER|") && line.contains("|ATT|ITEM|")) {

        // 1. 날짜 범위 읽기
        QDateTime dtStart = attendanceTable->getDateFrom();
        QDateTime dtEnd   = attendanceTable->getDateTo();


        QStringList lines = line.split('\n', Qt::SkipEmptyParts);
        for (const QString& l : lines) {
            QStringList parts = l.split('|', Qt::SkipEmptyParts);
            if (parts.size() >= 9) {
                QString name = parts[1];
                QString checkIn = parts[5] == "NULL" ? "" : parts[5];
                QString checkOut = parts[6] == "NULL" ? "" : parts[6];
                QString awayTime = parts[7];
                QString sleepTime = parts[8];

                // --- 날짜 필터링 추가 ---
                // checkIn 값이 있어야 필터 가능
                if (!checkIn.isEmpty()) {
                    QDateTime dtCheckIn = QDateTime::fromString(checkIn, "yyyy-MM-dd HH:mm:ss");
                    if (dtCheckIn.isValid() && dtCheckIn >= dtStart && dtCheckIn <= dtEnd) {
                        attendanceTable->appendAttendanceRow(name, checkIn, checkOut, awayTime, sleepTime);
                    }
                }
            }
        }
        attendanceTable->showAverageRow();
    }

    // 3. RFID / AWAY 처리
    parts = line.split("|");    // | 기준으로 분리(재정의)
    if(parts.size() < 4) {
        qDebug() << "파싱 오류 : 토큰 부족" << parts;
        return;
    }
    QString currentUser = parts[1];     // 새로 들어온 사용자 이름

    // ✅ RFID
    if (parts[2] == "RFID") {
        if (parts[3] == "NO") {
            QString msg = QString("[%1] RFID를 태그하지 않았습니다!").arg(currentUser);
            warningManager.showWarning(warningManager.rfidBoxes, currentUser,
                                       "출근 인식 경고", msg, warningManager.rfidList, QMessageBox::Critical);
        }
        else if (parts[3] == "YES") {
            if (warningManager.rfidBoxes.contains(currentUser)) {
                warningManager.rfidBoxes[currentUser]->close();    // 경고창 닫기
                warningManager.rfidBoxes.remove(currentUser);      // Map에서 key값 삭제
            }
            warningManager.rfidList.removeAll(currentUser); // 리스트에서 제거
        }
        return;
    }

    // ✅ AWAY
    if (parts[2] == "AWAY") {
        if (parts[3] == "ON")
        {
            QString msg = QString("[%1] 자리 비움 시간 5분 경과").arg(currentUser);
            warningManager.showWarning(warningManager.awayBoxes, currentUser,
                                       "자리 비움 경고", msg, warningManager.awayList);
        }
        else if (parts[3] == "OFF") {
            // 창은 닫지 않고 유지
            warningManager.awayList.removeAll(currentUser); // 리스트에서만 제거
        }
        return;
}
}
