#ifndef ATTENDANCETABLE_H
#define ATTENDANCETABLE_H

#include "warning.h"
#pragma once
#include <QWidget>
#include <QVector>
#include <QStringList>

namespace Ui {
class AttendanceTable;
}

class AttendanceTable : public QWidget
{
    Q_OBJECT

public:
    explicit AttendanceTable(QWidget *parent = nullptr);
    ~AttendanceTable();

    void appendAttendanceRow(const QString &name, const QString &checkIn, const QString &checkOut, const QString &awayTime, const QString &sleepTime);
    void showAverageRow();
    QDateTime getDateFrom() const;
    QDateTime getDateTo() const;

signals:
    void requestSendToServer(const QString &cmd);

private slots:
    void on_pPBdeleteDB_clicked();
    void on_pbSearch_clicked();
    void updateRecvDataSlot(QString line);

private:
    Ui::AttendanceTable *ui;
    QVector<QStringList> allRows; // 모든 데이터 저장 (평균 계산용)
    Warning* warningManager;
};

#endif // ATTENDANCETABLE_H
