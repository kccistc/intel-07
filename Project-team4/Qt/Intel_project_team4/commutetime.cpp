#include "commutetime.h"
#include "ui_commutetime.h"

CommuteTime::CommuteTime(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::CommuteTime)
{
    ui->setupUi(this);
}

CommuteTime::~CommuteTime()
{
    delete ui;
}
