#include "tab1.h"
#include "ui_tab1.h"
#include "socketclient.h"
#include <QDebug>
#include <QMessageBox>
#include <QTime>
#include <QPixmap>
#include <QBuffer>
#include <QImageReader>

tab1::tab1(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::tab1)
{
    ui->setupUi(this);
    ui->pPBSend->setEnabled(false);
    
    qDebug() << "=== tab1 생성자 시작 ===";
    
    if (ui->pLabelImage) {
        qDebug() << "이미지 레이블 존재 확인됨";
        ui->pLabelImage->setText("이미지 없음");
        ui->pLabelImage->setAlignment(Qt::AlignCenter);
        ui->pLabelImage->setStyleSheet("QLabel { border: 2px solid gray; border-radius: 5px; background-color: #f0f0f0; color: #666; font-size: 12px; }");
        qDebug() << "이미지 레이블 직접 초기화 완료";
    } else {
        qDebug() << "ERROR: 이미지 레이블이 null입니다!";
    }
    
    pSocketClient = new SocketClient(this);
    qDebug() << "SocketClient 생성 완료";
    
    connect(pSocketClient, SIGNAL(socketRecvDataSig(QString)), this, SLOT(updateRecvDataSlot(QString)));
    qDebug() << "텍스트 시그널 연결 완료";
    
    bool connected = connect(pSocketClient, SIGNAL(socketRecvImageSig(QByteArray)), 
                            this, SLOT(updateImageSlot(QByteArray)));
    qDebug() << "이미지 시그널 연결 상태:" << connected;
    
    if (ui->pTErecvData) {
        ui->pTErecvData->append("프로그램 시작됨 - " + QTime::currentTime().toString());
        qDebug() << "시작 메시지 추가 완료";
    } else {
        qDebug() << "ERROR: 텍스트 에디터가 null입니다!";
    }
    
    qDebug() << "=== tab1 생성자 완료 ===";
}

tab1::~tab1()
{
    delete ui;
}

void tab1::on_pPBserverConnect_toggled(bool checked)
{
    bool bFlag;
    if(checked)
    {
        qDebug() << "서버 연결 시도 중...";
        pSocketClient->connectToServerSlot(bFlag);
        if(bFlag)
        {
            ui->pPBserverConnect->setText("서버 해제");
            ui->pPBSend->setEnabled(true);
            
            QString connectMsg = QTime::currentTime().toString() + " [서버 연결 성공]";
            ui->pTErecvData->append(connectMsg);
            qDebug() << "서버 연결 성공";
        }
        else
        {
            ui->pPBserverConnect->setChecked(false);
            QMessageBox::warning(this, tr("연결 실패"),
                                 tr("서버에 연결할 수 없습니다."));
            qDebug() << "서버 연결 실패";
        }
    }
    else {
        qDebug() << "서버 연결 해제 중...";
        pSocketClient->socketClosedServerSlot();
        ui->pPBserverConnect->setText("서버 연결");
        ui->pPBSend->setEnabled(false);
        
        QString disconnectMsg = QTime::currentTime().toString() + " [서버 연결 해제]";
        ui->pTErecvData->append(disconnectMsg);
        qDebug() << "서버 연결 해제";
    }
}

void tab1::updateRecvDataSlot(QString strRecvData)
{
    qDebug() << "updateRecvDataSlot 호출됨 - 데이터 길이:" << strRecvData.length();
    qDebug() << "수신된 텍스트 데이터:" << strRecvData.left(100);
    
    // 개행문자 처리
    QStringList lines = strRecvData.split('\n', Qt::SkipEmptyParts);
    
    for(const QString &line : lines) {
        if (!line.trimmed().isEmpty()) {
            QTime time = QTime::currentTime();
            QString strTime = time.toString() + " " + line.trimmed();
            ui->pTErecvData->append(strTime);
            qDebug() << "로그 추가됨:" << strTime;
        }
    }
}

void tab1::updateImageSlot(QByteArray imageData)
{
    qDebug() << "=== updateImageSlot 호출됨 - 데이터 크기:" << imageData.size() << "===";
    
    if (!ui || !ui->pLabelImage) {
        qDebug() << "UI 또는 이미지 레이블이 null입니다!";
        return;
    }
    
    qDebug() << "UI 레이블 상태 - 크기:" << ui->pLabelImage->size() << "가시성:" << ui->pLabelImage->isVisible();
    
    displayImage(imageData);
    
    qDebug() << "=== updateImageSlot 완료 ===";
}

void tab1::on_pPBrecvDataClear_clicked()
{
    ui->pTErecvData->clear();
}

void tab1::on_pPBclearImage_clicked()
{
    qDebug() << "=== 이미지 지우기 버튼 클릭됨 ===";
    
    QString testMsg = QTime::currentTime().toString() + " - 이미지 지우기 버튼 클릭됨";
    ui->pTErecvData->append(testMsg);
    qDebug() << "텍스트 로그 추가:" << testMsg;
    
    static int testCount = 0;
    testCount++;
    
    if (testCount % 2 == 1) {
        ui->pLabelImage->setText("테스트 " + QString::number(testCount));
        ui->pLabelImage->setStyleSheet("QLabel { background-color: lightblue; color: black; font-size: 16px; border: 2px solid blue; }");
        qDebug() << "테스트 텍스트 표시";
    } else {
        clearImage();
        qDebug() << "이미지 지우기 실행";
    }
}

void tab1::on_pPBSend_clicked()
{
    QString strRecvId = ui->pLErecvId->text();
    QString strSendData = ui->pLEsendData->text();

    if(strSendData.isEmpty())
    {
        QMessageBox::information(this, tr("입력 오류"),
                                 tr("전송할 메시지를 입력해주세요."));
        return;
    }

    if(strRecvId.isEmpty())
    {
        strSendData = "[ALL] " + strSendData;
    }
    else
    {
        strSendData = "[" + strRecvId + "] " + strSendData;
    }

    pSocketClient->socketWriteDataSlot(strSendData);
    ui->pLEsendData->clear();
    ui->pLEsendData->setFocus();
}

void tab1::displayImage(const QByteArray &imageData)
{
    qDebug() << "=== displayImage 시작 - 데이터 크기:" << imageData.size() << "===";
    
    if (imageData.isEmpty())
    {
        qDebug() << "이미지 데이터가 비어있음";
        ui->pLabelImage->setText("이미지 데이터 없음");
        return;
    }
    
    QByteArray header = imageData.left(10);
    qDebug() << "이미지 헤더 (hex):" << header.toHex();
    
    QPixmap pixmap;
    if (pixmap.loadFromData(imageData))
    {
        qDebug() << "QPixmap 로드 성공 - 크기:" << pixmap.size();
        
        QSize labelSize = ui->pLabelImage->size();
        qDebug() << "레이블 크기:" << labelSize;
        
        QPixmap scaledPixmap = scaleImageToFit(pixmap, labelSize);
        qDebug() << "스케일된 이미지 크기:" << scaledPixmap.size();
        
        ui->pLabelImage->setPixmap(scaledPixmap);
        ui->pLabelImage->setScaledContents(false);
        ui->pLabelImage->setAlignment(Qt::AlignCenter);
        ui->pLabelImage->update();
        
        qDebug() << "이미지 표시 완료 - 원본:" << pixmap.size() << "스케일:" << scaledPixmap.size();
    }
    else
    {
        QImage image;
        if (image.loadFromData(imageData))
        {
            qDebug() << "QImage 로드 성공, QPixmap으로 변환 시도";
            QPixmap pixmap = QPixmap::fromImage(image);
            
            if (!pixmap.isNull())
            {
                QSize labelSize = ui->pLabelImage->size();
                QPixmap scaledPixmap = scaleImageToFit(pixmap, labelSize);
                ui->pLabelImage->setPixmap(scaledPixmap);
                ui->pLabelImage->setAlignment(Qt::AlignCenter);
                ui->pLabelImage->update();
                qDebug() << "QImage -> QPixmap 변환 성공";
            }
            else
            {
                qDebug() << "QPixmap 변환 실패";
                ui->pLabelImage->setText("이미지 변환 실패");
            }
        }
        else
        {
            qDebug() << "이미지 로드 완전 실패";
            qDebug() << "데이터 처음 32바이트:" << imageData.left(32).toHex();
            qDebug() << "데이터 마지막 32바이트:" << imageData.right(32).toHex();
            ui->pLabelImage->setText("이미지 로드 실패\n데이터 크기: " + QString::number(imageData.size()));
        }
    }
    
    qDebug() << "=== displayImage 완료 ===";
}

void tab1::clearImage()
{
    qDebug() << "clearImage 호출됨";
    
    if (!ui) {
        qDebug() << "ERROR: UI가 null - clearImage 실패";
        return;
    }
    
    if (!ui->pLabelImage) {
        qDebug() << "ERROR: 이미지 레이블이 null - clearImage 실패";
        return;
    }
    
    ui->pLabelImage->clear();
    ui->pLabelImage->setText("이미지 없음");
    ui->pLabelImage->setAlignment(Qt::AlignCenter);
    ui->pLabelImage->setStyleSheet("QLabel { border: 2px solid gray; border-radius: 5px; background-color: #f0f0f0; color: #666; font-size: 12px; }");
    
    qDebug() << "clearImage 완료 - 레이블 크기:" << ui->pLabelImage->size();
}

QPixmap tab1::scaleImageToFit(const QPixmap &pixmap, const QSize &labelSize)
{
    return pixmap.scaled(labelSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
}

SocketClient* tab1::getSocketClient()
{
    return pSocketClient;
}