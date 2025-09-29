#ifndef MAINWIDGET_H
#define MAINWIDGET_H

#include <QWidget>
#include <QMessageBox>
#include <QDebug>
#include <QRegularExpression>
#include <QLineEdit>

#include "mjpgview.h"
#include "sockclient.h"
#include "attendancetable.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWidget;
}
QT_END_NAMESPACE

class MainWidget : public QWidget
{
    Q_OBJECT

public:
    MainWidget(QWidget *parent = nullptr);
    ~MainWidget();

private:
    Ui::MainWidget *ui;

    MjpgView * mjpgview1;
    MjpgView * mjpgview2;
    AttendanceTable *attendanceTable;
    SockClient *sockclient;
    Warning warningManager;

private slots:
    void updateRecvDataSlot(QString);

};
#endif // MAINWIDGET_H
