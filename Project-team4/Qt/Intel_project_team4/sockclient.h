#ifndef SOCKCLIENT_H
#define SOCKCLIENT_H

#include <QWidget>
#include <QTcpSocket>
#include <QHostAddress>
#include <QInputDialog>
#include <QMessageBox>
#include <QDebug>

#define BLOCK_SIZE 1024

class SockClient : public QWidget
{
    Q_OBJECT
    QTcpSocket *pQTcpSocket;
    QString SERVERIP = "192.168.0.158"; // 라즈베리파이 주소
    int SERVERPORT = 5000;
    QString LOGID = "QT";
    QString LOGPW = "PASSWD";
public:
    explicit SockClient(QWidget *parent = nullptr);
    ~SockClient();

signals:
    void socketRevcDataSig(QString);

private slots:
    void socketReadDataSlot();
    void socketErrorSlot();
    void socketConnectServerSlot();

public slots:
    void connectToServerSlot(bool &);
    void socketClosedServerSlot();
    void socketWriteDataSlot(QString);
};

#endif // SOCKCLIENT_H
