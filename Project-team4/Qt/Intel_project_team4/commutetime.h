#ifndef COMMUTETIME_H
#define COMMUTETIME_H

#include <QWidget>

namespace Ui {
class CommuteTime;
}

class CommuteTime : public QWidget
{
    Q_OBJECT

public:
    explicit CommuteTime(QWidget *parent = nullptr);
    ~CommuteTime();

private:
    Ui::CommuteTime *ui;
};

#endif // COMMUTETIME_H
