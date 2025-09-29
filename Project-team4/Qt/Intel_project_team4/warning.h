#ifndef WARNING_H
#define WARNING_H

#include <QWidget>
#include <QMessageBox>
#include <QDebug>

class Warning : public QWidget
{
    Q_OBJECT

public:
    explicit Warning(QWidget *parent = nullptr);

    // ✅ 사용자별 경고창 관리 (실제 창 포인터 저장)
    QMap<QString, QMessageBox*> rfidBoxes;  // RFID 경고창
    QMap<QString, QMessageBox*> awayBoxes;  // AWAY 경고창

    // ✅ 사용자 기록 관리 (리스트만)
    QStringList rfidList;   // RFID 경고 사용자 기록
    QStringList awayList;   // AWAY 경고 사용자 기록

    // ✅ 경고창 생성 함수 (오버로드된 헬퍼 함수)
    void showWarning(QMap<QString, QMessageBox*> &warningMap,
                     const QString &user,
                     const QString &title,
                     const QString &text,
                     QStringList &userList,
                     QMessageBox::Icon icon = QMessageBox::Warning);

signals:
};

#endif // WARNING_H
