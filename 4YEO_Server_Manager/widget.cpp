#include "widget.h"
#include "ui_widget.h"
#include "tab1.h"
#include <QDebug>

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
    , m_tab1Widget(nullptr)
{
    ui->setupUi(this);

    // 탭 설정
    setupTabs();

    // 시그널-슬롯 연결
    setupConnections();
}

Widget::~Widget()
{
    // m_tab1Widget은 QTabWidget의 자식으로 추가되어 자동으로 삭제됨
    delete ui;
}

void Widget::setupTabs()
{
    // 기존 더미 탭들 제거 (widget.ui에서 생성된 tab과 tab_2)
    ui->tabWidget->clear();

    // tab1 위젯 생성 및 탭에 추가
    m_tab1Widget = new tab1(this);
    ui->tabWidget->addTab(m_tab1Widget, tr("서버 통신"));

    // 추가 탭들이 필요한 경우 여기에 추가
    // 예: ui->tabWidget->addTab(new SomeOtherTab(this), tr("다른 기능"));
}

void Widget::setupConnections()
{
    // 탭 변경 시그널 연결
    connect(ui->tabWidget, &QTabWidget::currentChanged,
            this, &Widget::onTabChanged);

    // tab1의 시그널들을 필요에 따라 연결
    if (m_tab1Widget) {
        // 필요한 경우 tab1의 다른 시그널들을 여기서 연결
        // connect(m_tab1Widget, &tab1::someSignal,
        //         this, &Widget::someSlot);
    }
}

void Widget::onTabChanged(int index)
{
    qDebug() << "탭 변경됨 - 인덱스:" << index;

    // 탭 변경 시 필요한 처리 로직
    switch (index) {
    case 0:  // tab1 (서버 통신 탭)
        // 서버 통신 탭 활성화 시 필요한 처리
        break;
    default:
        break;
    }
}

tab1* Widget::getTab1Widget() const
{
    return m_tab1Widget;
}
