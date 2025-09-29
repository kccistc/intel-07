#ifndef TAB1_H
#define TAB1_H

#include <QWidget>
#include <QLabel>
#include <QPixmap>
#include <QScrollArea>

// 전방 선언
class SocketClient;

namespace Ui {
class tab1;
}

class tab1 : public QWidget
{
    Q_OBJECT

public:
    explicit tab1(QWidget *parent = nullptr);
    ~tab1();

    // 소켓 클라이언트 접근자
    SocketClient* getSocketClient();

private slots:
    // UI 이벤트 핸들러들
    void on_pPBserverConnect_toggled(bool checked);
    void on_pPBrecvDataClear_clicked();
    void on_pPBSend_clicked();
    void on_pPBclearImage_clicked();

    // 소켓 데이터 수신 처리
    void updateRecvDataSlot(QString strRecvData);
    void updateImageSlot(QByteArray imageData);

private:
    Ui::tab1 *ui;
    SocketClient *pSocketClient;  // 소켓 클라이언트 포인터
    
    // 이미지 관련 메서드
    void displayImage(const QByteArray &imageData);
    void clearImage();
    QPixmap scaleImageToFit(const QPixmap &pixmap, const QSize &labelSize);
};

#endif // TAB1_H