#include "warning.h"

Warning::Warning(QWidget *parent)
    : QWidget{parent}
{}

void Warning::showWarning(QMap<QString, QMessageBox*> &warningMap,
                             const QString &user,
                             const QString &title,
                             const QString &text,
                             QStringList &userList,
                             QMessageBox::Icon icon)
{
    if (warningMap.contains(user)) return; // 이미 창이 떠 있으면 무시

    QMessageBox *box = new QMessageBox(icon, title, text, QMessageBox::Ok, this);   // 동적

    box->setStyleSheet("QLabel{ font-size: 18pt; font-weight: bold; }"
                       "QPushButton{ font-size: 14pt; }");  // 경고창 크기 조정
    box->setAttribute(Qt::WA_DeleteOnClose); // 닫힐 때 메모리 해제

    // 창이 닫혀도 리스트에서는 삭제 ❌
    connect(box, &QMessageBox::finished, this, [&, user](int){
        warningMap.remove(user); // 맵에서는 제거
    });

    box->show();
    warningMap[user] = box;

    if (!userList.contains(user))
        userList.append(user); // 리스트에 기록
}
