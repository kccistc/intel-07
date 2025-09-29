#ifndef SOCKETCLIENT_H
#define SOCKETCLIENT_H

#include <QWidget>
#include <QTcpSocket>
#include <QHostAddress>
#include <QInputDialog>
#include <QDebug>
#include <QMessageBox>

#define BLOCK_SIZE 1024

class SocketClient : public QWidget
{
    Q_OBJECT
    QTcpSocket *pQTcpSocket;
    QString SERVERIP = "192.168.0.43";
    int SERVERPORT = 5000;
    QString LOGID = "11";
    QString LOGPW = "PASSWD";

    // 이미지 수신 관련 변수들
    bool m_receivingImage;
    QByteArray m_imageBuffer;
    long m_expectedImageSize;
    QString m_imageFileName;

public:
    explicit SocketClient(QWidget *parent = nullptr);
    ~SocketClient();

signals:
    void socketRecvDataSig(QString strRecvData);
    void socketRecvImageSig(QByteArray imageData);

private slots:
    void socketReadDataSlot();
    void socketErrorSlot();

public slots:
    void connectToServerSlot(bool &);
    void socketClosedServerSlot();
    void socketWriteDataSlot(QString);

private:
    void processImageData();
    void resetImageReceiving();
};

#endif // SOCKETCLIENT_H