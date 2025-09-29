#include "sockclient.h"

SockClient::SockClient(QWidget *parent)
    : QWidget{parent}
{
    pQTcpSocket = new QTcpSocket();

    // 서버에 성공적으로 연결되었을 때 호출될 슬롯을 연결
    connect(pQTcpSocket, SIGNAL(connected()), this, SLOT(socketConnectServerSlot()));

    // 서버와의 연결이 끊겼을 때 호출될 슬롯을 연결
    connect(pQTcpSocket, SIGNAL(disconnected()), this, SLOT(socketClosedServerSlot()));

    // 수신할 데이터가 도착했을 때 호출될 슬롯을 연결
    connect(pQTcpSocket, SIGNAL(readyRead()), this, SLOT(socketReadDataSlot()));

    // 소켓 오류가 발생했을 때 호출될 슬롯을 연결
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    connect(pQTcpSocket, SIGNAL(errorOccurred(QAbstractSocket::SocketError)), this, SLOT(socketErrorSlot()));
#else
    connect(pQTcpSocket, SIGNAL(error(QAbstractSocket::SocketError)), this, SLOT(socketErrorSlot()));
#endif

    pQTcpSocket->connectToHost(SERVERIP, SERVERPORT);

}

void SockClient::connectToServerSlot(bool &bFlag)
{
    QString strHostIp;
    strHostIp = QInputDialog::getText(this, "Host Ip", "Input Server IP", QLineEdit::Normal, SERVERIP, &bFlag);
    if(bFlag)
    {
        if(strHostIp.isEmpty()) // 입력창이 비어있다면
            pQTcpSocket->connectToHost(SERVERIP, SERVERPORT); // 디폴트로 연결
        else
            pQTcpSocket->connectToHost(strHostIp, SERVERPORT); // 입력창으로 연결
    }
}

void SockClient::socketReadDataSlot()
{
    while (pQTcpSocket->canReadLine()) {
        QByteArray line = pQTcpSocket->readLine(); // '\n' 포함
        emit socketRevcDataSig(QString::fromUtf8(line));
    }
}

void SockClient::socketErrorSlot()
{
    QString strError = pQTcpSocket->errorString();
    QMessageBox::information(this, "socket", "error : " + strError);
}
void SockClient::socketConnectServerSlot()
{
    QString strIdPw = "[" + LOGID + ":" + LOGPW + "]";
    // qDebug() << strIdPw;
    QByteArray byteIdPw = strIdPw.toLocal8Bit(); // QString(유니코드)를 현재 OS의 기본 인코딩(CP949, UTF-8 등)을 사용하는 배열로 변환
    pQTcpSocket->write(byteIdPw);

}

void SockClient::socketClosedServerSlot()
{
    pQTcpSocket->close(); // 서버 연결 닫기
}

void SockClient::socketWriteDataSlot(QString strData)
{
    strData = strData + "\n";
    QByteArray byteData = strData.toLocal8Bit(); // QString(유니코드)를 현재 OS의 기본 인코딩(CP949, UTF-8 등)을 사용하는 배열로 변환
    pQTcpSocket->write(byteData);
}

SockClient::~SockClient()
{

}
