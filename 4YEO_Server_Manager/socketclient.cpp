#include "socketclient.h"
#include <QBuffer>
#include <QImageReader>

SocketClient::SocketClient(QWidget *parent)
    : QWidget{parent}
    , m_receivingImage(false)
    , m_expectedImageSize(0)
{
    pQTcpSocket = new QTcpSocket();

    connect(pQTcpSocket, &QTcpSocket::connected, this, [this]() {
        QString loginPacket = QString("[%1:%2]").arg(LOGID).arg(LOGPW);
        pQTcpSocket->write(loginPacket.toUtf8());
        qDebug() << "로그인 패킷 자동 전송:" << loginPacket;
        resetImageReceiving();
    });
    
    connect(pQTcpSocket, SIGNAL(disconnected()), this, SLOT(socketClosedServerSlot()));
    connect(pQTcpSocket, SIGNAL(readyRead()), this, SLOT(socketReadDataSlot()));
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    connect(pQTcpSocket, SIGNAL(errorOccurred(QAbstractSocket::SocketError)), this, SLOT(socketErrorSlot()));
#else
    connect(pQTcpSocket, SIGNAL(error(QAbstractSocket::SocketError)), this, SLOT(socketErrorSlot()));
#endif

    resetImageReceiving();
}

void SocketClient::connectToServerSlot(bool &bFlag)
{
    QString strHostIp;
    strHostIp = QInputDialog::getText(this,"Host Ip", "Input Server IP",QLineEdit::Normal,SERVERIP, &bFlag);
    if(bFlag)
    {
        if(strHostIp.isEmpty())
            pQTcpSocket->connectToHost(SERVERIP, SERVERPORT);
        else
            pQTcpSocket->connectToHost(strHostIp, SERVERPORT);
    }
}

void SocketClient::socketReadDataSlot()
{
    if (pQTcpSocket->state() != QAbstractSocket::ConnectedState) {
        qDebug() << "소켓이 연결되지 않음 - 데이터 읽기 중단";
        resetImageReceiving();
        return;
    }
    
    qDebug() << "socketReadDataSlot 호출됨 - 사용 가능한 바이트:" << pQTcpSocket->bytesAvailable() << "수신 중:" << m_receivingImage;
    
    while (pQTcpSocket->bytesAvailable() > 0)
    {
        if (!m_receivingImage)
        {
            if (pQTcpSocket->bytesAvailable() < 10) {
                qDebug() << "데이터가 너무 적음, 대기 중...";
                return;
            }
                
            QByteArray headerData = pQTcpSocket->peek(1024);
            QString headerStr = QString::fromLocal8Bit(headerData);
            
            qDebug() << "수신된 데이터 헤더 (전체):" << headerStr;
            
            if (headerStr.startsWith("IMAGE:"))
            {
                qDebug() << "*** 새로운 이미지 헤더 감지됨 ***";
                int newlineIndex = headerStr.indexOf('\n');
                if (newlineIndex == -1)
                {
                    qDebug() << "이미지 헤더가 완전히 수신되지 않음, 대기 중...";
                    return;
                }
                
                QString imageHeader = headerStr.left(newlineIndex);
                qDebug() << "완전한 이미지 헤더:" << imageHeader;
                
                QByteArray consumedHeader = pQTcpSocket->read(newlineIndex + 1);
                qDebug() << "헤더 소비됨:" << consumedHeader.size() << "바이트";
                
                QStringList parts = imageHeader.split(':');
                if (parts.size() == 3 && parts[0] == "IMAGE")
                {
                    m_imageFileName = parts[1];
                    m_expectedImageSize = parts[2].toLong();
                    
                    if (m_expectedImageSize > 0 && m_expectedImageSize < 10000000)
                    {
                        m_receivingImage = true;
                        m_imageBuffer.clear();
                        m_imageBuffer.reserve(m_expectedImageSize);
                        
                        qDebug() << "*** 새로운 이미지 수신 시작 ***";
                        qDebug() << "파일명:" << m_imageFileName;
                        qDebug() << "예상 크기:" << m_expectedImageSize;
                    }
                    else
                    {
                        qDebug() << "잘못된 이미지 크기:" << m_expectedImageSize;
                    }
                }
                else
                {
                    qDebug() << "이미지 헤더 파싱 실패:" << imageHeader;
                }
            }
            else
            {
                QByteArray data = pQTcpSocket->readAll();
                QString strRecvData = QString::fromLocal8Bit(data);
                qDebug() << "*** 텍스트 메시지 수신 *** 전체 내용:" << strRecvData;
                
                if (!strRecvData.trimmed().isEmpty()) {
                    emit socketRecvDataSig(strRecvData);
                    qDebug() << "텍스트 시그널 발송 완료";
                }
            }
        }
        
        if (m_receivingImage)
        {
            qDebug() << "이미지 데이터 수신 모드 - 현재 버퍼:" << m_imageBuffer.size() << "/" << m_expectedImageSize;
            processImageData();
        }
    }
    
    qDebug() << "socketReadDataSlot 완료 - 최종 m_receivingImage 상태:" << m_receivingImage;
}

void SocketClient::processImageData()
{
    qint64 remainingBytes = m_expectedImageSize - m_imageBuffer.size();
    qint64 availableBytes = pQTcpSocket->bytesAvailable();
    
    if (availableBytes > 0 && remainingBytes > 0)
    {
        qint64 bytesToRead = qMin(remainingBytes, availableBytes);
        
        QByteArray data = pQTcpSocket->read(bytesToRead);
        if (!data.isEmpty())
        {
            m_imageBuffer.append(data);
            
            double progress = (double)m_imageBuffer.size() / m_expectedImageSize * 100.0;
            qDebug() << "이미지 데이터 수신:" << data.size() << "bytes, 총:" << m_imageBuffer.size() << "/" << m_expectedImageSize << "(" << QString::number(progress, 'f', 1) << "%)";
            
            if (m_imageBuffer.size() >= 500) {
                emit socketRecvImageSig(m_imageBuffer);
            }
            
            if (m_imageBuffer.size() >= m_expectedImageSize)
            {
                qDebug() << "이미지 수신 완료:" << m_imageFileName << "실제 크기:" << m_imageBuffer.size();
                
                QByteArray imageDataCopy = m_imageBuffer;
                resetImageReceiving();
                
                qDebug() << "최종 이미지 시그널 발송 중...";
                emit socketRecvImageSig(imageDataCopy);
                qDebug() << "최종 이미지 시그널 발송 완료";
            }
        }
    }
}

void SocketClient::resetImageReceiving()
{
    qDebug() << "=== resetImageReceiving 호출됨 ===";
    if (m_receivingImage) {
        qDebug() << "이전 상태 - receivingImage:" << m_receivingImage << "bufferSize:" << m_imageBuffer.size() << "expectedSize:" << m_expectedImageSize;
    }
    
    m_receivingImage = false;
    m_imageBuffer.clear();
    m_expectedImageSize = 0;
    m_imageFileName.clear();
    
    qDebug() << "이미지 수신 상태 초기화 완료";
}

void SocketClient::socketErrorSlot()
{
    QString strError = pQTcpSocket->errorString();
    QMessageBox::information(this, "socket", "error : "+strError);
    qDebug() << "소켓 오류:" << strError;
    
    if (m_receivingImage && !m_imageBuffer.isEmpty()) {
        qDebug() << "소켓 오류 발생 - 부분 이미지 데이터 처리:" << m_imageBuffer.size() << "bytes";
        QByteArray partialData = m_imageBuffer;
        resetImageReceiving();
        emit socketRecvImageSig(partialData);
    } else {
        resetImageReceiving();
    }
}

void SocketClient::socketClosedServerSlot()
{
    qDebug() << "서버 연결 해제 - 이미지 수신 상태 확인";
    
    if (m_receivingImage && !m_imageBuffer.isEmpty()) {
        qDebug() << "연결 끊김 - 부분 이미지 데이터 처리 시도:" << m_imageBuffer.size() << "bytes";
        QByteArray partialData = m_imageBuffer;
        resetImageReceiving();
        emit socketRecvImageSig(partialData);
    } else {
        resetImageReceiving();
    }
    
    pQTcpSocket->close();
}

void SocketClient::socketWriteDataSlot(QString strData)
{
    strData = strData + "\n";
    QByteArray byteData = strData.toLocal8Bit();
    pQTcpSocket->write(byteData);
}

SocketClient::~SocketClient()
{

}