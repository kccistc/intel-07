#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>

// 전방 선언
class tab1;

QT_BEGIN_NAMESPACE
namespace Ui {
class Widget;
}
QT_END_NAMESPACE

class Widget : public QWidget
{
    Q_OBJECT

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();
    
    // tab1 위젯 접근자
    tab1* getTab1Widget() const;

private slots:
    // 탭 관련 슬롯들 (필요한 경우 추가)
    void onTabChanged(int index);

private:
    Ui::Widget *ui;
    tab1 *m_tab1Widget;  // tab1 위젯 포인터
    
    // 초기화 함수들
    void setupTabs();
    void setupConnections();
};

#endif // WIDGET_H