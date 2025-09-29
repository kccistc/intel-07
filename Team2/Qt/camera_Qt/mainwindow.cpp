// mainwindow.cpp
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QMessageBox>
#include <QDateTime>
#include <QDebug>
#include <QBuffer>
#include <QHttpMultiPart>
#include <QHttpPart>
#include <QNetworkRequest>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSettings>
#include <QProgressDialog>
#include <QTimer>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QPushButton>
#include <QFont>
#include <QResizeEvent>
#include <QApplication>
#include <QScreen>
#include <QPainter>
#include <QProcess>
#include <QCoreApplication>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , isCameraRunning(false)
    , isNameEntered(false)
    , cameraPreviewLabel(nullptr)
    , gstreamerProcess(nullptr)
    , overlayWidget(nullptr)
    , overlayTimer(nullptr)
{
    ui->setupUi(this);

    // 화면 크기에 맞춰 창을 최대화하고 동적 크기 조정 가능하게 설정
    setupWindowSizing();

    // 레이아웃 설정 (UI 파일에 레이아웃이 없는 경우)
    setupUILayout();

    // 네트워크 매니저 초기화
    networkManager = std::make_unique<QNetworkAccessManager>(this);
    connect(networkManager.get(), &QNetworkAccessManager::finished,
            this, &MainWindow::onUploadFinished);

    // 서버 설정 로드
    loadServerConfig();

    // 데이터베이스 초기화
    initializeDatabase();

    // 초기 화면 설정 (이름 입력 단계)
    setupInitialView();

    // 상태바에 서버 정보 표시
    ui->statusbar->showMessage(QString("Server: %1").arg(serverUrl));
}

MainWindow::~MainWindow()
{
    // rpicam 프로세스 중지
    if (gstreamerProcess && gstreamerProcess->state() == QProcess::Running) {
        gstreamerProcess->terminate();
        if (!gstreamerProcess->waitForFinished(2000)) {
            gstreamerProcess->kill();
        }
    }

    // 오버레이 정리
    if (overlayWidget) {
        delete overlayWidget;
        overlayWidget = nullptr;
    }

    if (overlayTimer) {
        overlayTimer->stop();
        delete overlayTimer;
        overlayTimer = nullptr;
    }

    delete ui;
}

void MainWindow::loadServerConfig()
{
    // QSettings를 사용하여 설정 파일에서 서버 정보 읽기
    // 또는 하드코딩된 값 사용
    QSettings settings("config.ini", QSettings::IniFormat);

    // 기본값 설정 (필요에 따라 수정)
    serverUrl = settings.value("Server/url", "http://192.168.0.90:5000").toString();
    serverEndpoint = settings.value("Server/endpoint", "/upload").toString();
    raspUrl = settings.value("Server/rasp_url", "http://localhost:5000").toString();

    // 설정 파일이 없으면 생성
    if (!settings.contains("Server/url")) {
        settings.setValue("Server/url", serverUrl);
        settings.setValue("Server/endpoint", serverEndpoint);
        settings.setValue("Server/rasp_url", raspUrl);
        settings.sync();

        qDebug() << "Created config.ini with default server settings";
    }

    qDebug() << "Server URL:" << serverUrl + serverEndpoint;
    qDebug() << "Rasp URL:" << raspUrl;
}

void MainWindow::setupWindowSizing()
{
    // 현재 사용 가능한 화면 크기 가져오기
    QScreen *screen = QApplication::primaryScreen();
    if (!screen) return;
    
    QRect screenGeometry = screen->availableGeometry();
    
    // 전체화면으로 창 크기 설정
    int windowWidth = screenGeometry.width();
    int windowHeight = screenGeometry.height();
    
    // 최소 크기 설정
    setMinimumSize(600, 500);
    
    // 창 크기 설정 (전체화면)
    resize(windowWidth, windowHeight);
    
    // 창을 화면 왼쪽 상단에 위치
    move(screenGeometry.x(), screenGeometry.y());
    
    qDebug() << "Screen geometry:" << screenGeometry;
    qDebug() << "Window size set to:" << windowWidth << "x" << windowHeight;
    qDebug() << "Window positioned at:" << screenGeometry.x() << "," << screenGeometry.y();
}


void MainWindow::startCamera()
{
    // rpicam 모드: GStreamer로 실시간 비디오 스트림
    startGStreamerCamera();
    isCameraRunning = true;

    if (ui->camStartButton) {
        ui->camStartButton->setText("카메라 중지");
    }
    if (ui->snapShotButton) {
        ui->snapShotButton->setEnabled(true);
    }
}

void MainWindow::stopCamera()
{
    // rpicam 모드: GStreamer 중지
    stopGStreamerCamera();
    isCameraRunning = false;

    if (ui->camStartButton) {
        ui->camStartButton->setText("카메라 시작");
    }
    if (ui->snapShotButton) {
        ui->snapShotButton->setEnabled(false);
    }
}

void MainWindow::on_camStartButton_clicked()
{
    if (isCameraRunning) {
        stopCamera();
    } else {
        startCamera();
    }
}

void MainWindow::on_snapShotButton_clicked()
{
    if (!isNameEntered) {
        QMessageBox::warning(this, "오류", "먼저 이름을 입력해주세요.");
        return;
    }

    // rpicam 모드: rpicam-still로 직접 촬영
    captureWithRpicam();
}


void MainWindow::uploadImageToServer(const QImage& image)
{
    // Progress dialog 생성
    QProgressDialog* progressDialog = new QProgressDialog("Uploading image to server...", "Cancel", 0, 100, this);
    progressDialog->setWindowModality(Qt::WindowModal);
    progressDialog->show();

    // 이미지를 JPEG 형식으로 변환
    QByteArray imageData;
    QBuffer buffer(&imageData);
    buffer.open(QIODevice::WriteOnly);
    image.save(&buffer, "JPG", 90); // 90% 품질로 저장

    // HTTP multipart 요청 생성
    QHttpMultiPart *multiPart = new QHttpMultiPart(QHttpMultiPart::FormDataType);

    // 이미지 파트 생성
    QHttpPart imagePart;
    imagePart.setHeader(QNetworkRequest::ContentTypeHeader, QVariant("image/jpeg"));
    imagePart.setHeader(QNetworkRequest::ContentDispositionHeader,
                        QVariant(QString("form-data; name=\"image\"; filename=\"snapshot_%1.jpg\"")
                                     .arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"))));
    imagePart.setBody(imageData);

    // 추가 메타데이터 파트 (옵션)
    QHttpPart metadataPart;
    metadataPart.setHeader(QNetworkRequest::ContentDispositionHeader,
                           QVariant("form-data; name=\"metadata\""));

    QJsonObject metadata;
    metadata["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    metadata["device_id"] = QSysInfo::machineHostName();
    metadata["image_width"] = image.width();
    metadata["image_height"] = image.height();

    QJsonDocument doc(metadata);
    metadataPart.setBody(doc.toJson());

    multiPart->append(imagePart);
    multiPart->append(metadataPart);

    // 요청 생성
    QNetworkRequest request;
    request.setUrl(QUrl(serverUrl + serverEndpoint));
    request.setRawHeader("User-Agent", "Qt Camera Client 1.0");

    // POST 요청 전송
    QNetworkReply* reply = networkManager->post(request, multiPart);
    multiPart->setParent(reply); // reply가 삭제될 때 multiPart도 삭제

    // Progress 업데이트 연결
    connect(reply, &QNetworkReply::uploadProgress,
            [progressDialog](qint64 bytesSent, qint64 bytesTotal) {
                if (bytesTotal > 0) {
                    int progress = static_cast<int>((bytesSent * 100) / bytesTotal);
                    progressDialog->setValue(progress);
                }
            });

    // 취소 버튼 처리
    connect(progressDialog, &QProgressDialog::canceled,
            [reply]() {
                reply->abort();
            });

    // 완료 시 dialog 삭제
    connect(reply, &QNetworkReply::finished,
            [progressDialog]() {
                progressDialog->deleteLater();
            });

    qDebug() << "Uploading image to:" << serverUrl + serverEndpoint;
    qDebug() << "Image size:" << imageData.size() << "bytes";
}

void MainWindow::onUploadFinished(QNetworkReply* reply)
{
    // 버튼 다시 활성화
    ui->snapShotButton->setEnabled(true);
    ui->snapShotButton->setText("Upload Snapshot");

    // 응답 처리
    if (reply->error() == QNetworkReply::NoError) {
        QByteArray response = reply->readAll();

        // JSON 응답 파싱 시도
        QJsonDocument jsonResponse = QJsonDocument::fromJson(response);
        QString message = "Image uploaded successfully!";

        if (!jsonResponse.isNull() && jsonResponse.isObject()) {
            QJsonObject obj = jsonResponse.object();

            if (obj.contains("message")) {
                QString serverMessage = obj["message"].toString();
                message = serverMessage;
            }
            if (obj.contains("file_id")) {
                message += QString("\nFile ID: %1").arg(obj["file_id"].toString());
            }
        }

        ui->statusbar->showMessage("Upload successful - Fetching analysis result...", 3000);
        qDebug() << "Server response:" << response;
        
        // 업로드 성공 후 분석 결과 가져오기 (약간의 지연 후)
        QTimer::singleShot(10000, this, &MainWindow::fetchAnalysisResult);
        
    } else {
        // 업로드 실패 시에도 서버 응답을 확인 (ROI detection 실패 메시지 처리)
        QByteArray response = reply->readAll();
        qDebug() << "Upload failed. Server response:" << response;

        // JSON 응답 파싱 시도
        QJsonDocument jsonResponse = QJsonDocument::fromJson(response);
        if (!jsonResponse.isNull() && jsonResponse.isObject()) {
            QJsonObject obj = jsonResponse.object();

            if (obj.contains("message")) {
                QString serverMessage = obj["message"].toString();

                if (serverMessage.startsWith("ROI detection failed") || serverMessage.startsWith("ROI 탐지 실패")) {
                    // "ROI detection failed : part1, part2, ..." 또는 "ROI 탐지 실패: part1, part2, ..." 형태에서 부위 추출
                    QStringList parts;
                    QString retryMessage = "사진을 다시 찍어 주세요";

                    if (serverMessage.contains(" : ")) {
                        parts = serverMessage.split(" : ");
                    } else if (serverMessage.contains(": ")) {
                        parts = serverMessage.split(": ");
                    }

                    if (parts.size() > 1) {
                        QString failedParts = parts[1].trimmed();
                        // "facepart::" 제거
                        failedParts = failedParts.replace("facepart::", "");
                        retryMessage = QString("사진을 다시 찍어 주세요 : %1").arg(failedParts);
                    }

                    QMessageBox::warning(this, "분석 실패", retryMessage);
                    qDebug() << "ROI detection failed for parts:" << parts;

                    // 카메라를 다시 활성화하여 재촬영 가능하게 함
                    ui->snapShotButton->setEnabled(true);
                    ui->snapShotButton->setText("사진 촬영");
                    ui->statusbar->showMessage("ROI 검출 실패 - 사진을 다시 촬영해주세요", 5000);

                    reply->deleteLater();
                    return;
                }
            }
        }

        // 일반적인 업로드 에러 처리
        QString errorMsg = QString("Upload failed!\nError: %1\n%2")
        .arg(reply->error())
            .arg(reply->errorString());

        // HTTP 상태 코드 확인
        int statusCode = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        if (statusCode) {
            errorMsg += QString("\nHTTP Status: %1").arg(statusCode);
        }

        QMessageBox::critical(this, "Upload Error", errorMsg);
        ui->statusbar->showMessage("Upload failed", 3000);

        qDebug() << "Upload error:" << reply->errorString();
    }

    reply->deleteLater();
}

void MainWindow::onUploadProgress(qint64 bytesSent, qint64 bytesTotal)
{
    if (bytesTotal > 0) {
        int progress = static_cast<int>((bytesSent * 100) / bytesTotal);
        ui->statusbar->showMessage(QString("Uploading... %1%").arg(progress));
    }
}


void MainWindow::setupUILayout()
{
    // 기존 UI가 레이아웃이 없는 경우, 코드로 추가

    // 메인 레이아웃 생성
    QVBoxLayout *mainLayout = new QVBoxLayout();
    mainLayout->setContentsMargins(15, 15, 15, 15);
    mainLayout->setSpacing(20);  // 간격을 늘려서 버튼과 카메라 분리

    // 카메라 뷰어를 동적 크기 조정 가능하게 설정
    ui->camViewer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->camViewer->setMinimumSize(480, 360);  // 최소 크기를 더 크게 설정 (16:9 비율)
    ui->camViewer->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);  // 최대 크기 제한 제거
    ui->camViewer->setAlignment(Qt::AlignCenter);  // 중앙 정렬
    
    // 카메라 뷰어를 메인 레이아웃에 직접 추가하여 공간을 최대한 활용
    mainLayout->addWidget(ui->camViewer, 1);  // stretch factor를 1로 설정하여 확장 가능

    // 카메라와 버튼 사이에 여백 추가
    mainLayout->addSpacing(15);

    // 버튼 레이아웃 생성 (카메라 화면 밖에 배치)
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    buttonLayout->setSpacing(30);  // 버튼 간격 더 증가

    // 버튼들을 중앙에 배치
    buttonLayout->addStretch();  // 왼쪽 여백

    // 버튼 크기를 동적으로 조정 가능하게 설정
    ui->camStartButton->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    ui->camStartButton->setMinimumSize(180, 50);  // 크기 더 증가
    ui->camStartButton->setMaximumSize(300, 70);  // 최대 크기도 증가
    ui->camStartButton->setStyleSheet(
        "QPushButton {"
        "    background-color: #3498db;"
        "    color: white;"
        "    border: none;"
        "    border-radius: 8px;"
        "    font-size: 14px;"
        "    font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "    background-color: #2980b9;"
        "}"
        "QPushButton:pressed {"
        "    background-color: #21618c;"
        "}"
    );
    buttonLayout->addWidget(ui->camStartButton);

    ui->snapShotButton->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    ui->snapShotButton->setMinimumSize(180, 50);  // 크기 더 증가
    ui->snapShotButton->setMaximumSize(300, 70);  // 최대 크기도 증가
    ui->snapShotButton->setStyleSheet(
        "QPushButton {"
        "    background-color: #27ae60;"
        "    color: white;"
        "    border: none;"
        "    border-radius: 8px;"
        "    font-size: 14px;"
        "    font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "    background-color: #2ecc71;"
        "}"
        "QPushButton:pressed {"
        "    background-color: #239653;"
        "}"
        "QPushButton:disabled {"
        "    background-color: #95a5a6;"
        "}"
    );
    buttonLayout->addWidget(ui->snapShotButton);

    buttonLayout->addStretch();  // 오른쪽 여백

    // 버튼 레이아웃을 메인 레이아웃에 추가
    mainLayout->addLayout(buttonLayout, 0);  // stretch factor 0으로 고정

    // centralwidget에 레이아웃 설정
    ui->centralwidget->setLayout(mainLayout);
}

void MainWindow::fetchAnalysisResult()
{
    // rasp.py에서 분석 결과 가져오기 - 별도의 네트워크 매니저 사용
    QNetworkAccessManager* analysisNetworkManager = new QNetworkAccessManager(this);
    
    QNetworkRequest request;
    request.setUrl(QUrl(raspUrl + "/get_analysis"));
    request.setRawHeader("User-Agent", "Qt Camera Client 1.0");
    request.setRawHeader("Accept", "application/json");
    request.setRawHeader("Connection", "close");
    
    QNetworkReply* reply = analysisNetworkManager->get(request);
    
    // 타임아웃 설정
    QTimer::singleShot(10000, reply, &QNetworkReply::abort); // 10초 타임아웃
    
    connect(reply, &QNetworkReply::finished, [this, reply, analysisNetworkManager]() {
        qDebug() << "Network reply finished with error:" << reply->error();
        qDebug() << "HTTP status code:" << reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        qDebug() << "Bytes available:" << reply->bytesAvailable();
        
        // 모든 데이터를 받을 때까지 기다림
        reply->waitForReadyRead(3000);
        
        if (reply->error() == QNetworkReply::NoError) {
            QByteArray response;
            
            // 데이터를 조각별로 읽기
            while (!reply->atEnd()) {
                QByteArray chunk = reply->read(1024);
                response.append(chunk);
                qDebug() << "Read chunk of size:" << chunk.size();
            }
            
            // readAll()로도 한번 더 시도
            QByteArray remaining = reply->readAll();
            response.append(remaining);
            
            qDebug() << "Raw response:" << response;
            qDebug() << "Response length:" << response.length();
            
            if (response.isEmpty()) {
                qDebug() << "Empty response received!";
                ui->statusbar->showMessage("Empty response from server", 3000);
                reply->deleteLater();
                return;
            }
            
            QJsonParseError parseError;
            QJsonDocument jsonDoc = QJsonDocument::fromJson(response, &parseError);
            
            if (parseError.error != QJsonParseError::NoError) {
                qDebug() << "JSON parse error:" << parseError.errorString();
                qDebug() << "Invalid JSON response:" << response;
                ui->statusbar->showMessage("Invalid JSON response", 3000);
                reply->deleteLater();
                return;
            }
            
            if (jsonDoc.isObject()) {
                QJsonObject responseObj = jsonDoc.object();
                qDebug() << "Parsed JSON object:" << responseObj;

                // ROI detection 실패 처리 추가
                if (responseObj.contains("analysis_data")) {
                    QJsonObject analysisData = responseObj["analysis_data"].toObject();

                    // ROI detection 실패 체크
                    if (analysisData.contains("status") &&
                        analysisData["status"].toString() == "roi_failed") {

                        QString failedMessage = analysisData["message"].toString();
                        QString retryMessage = "사진을 다시 찍어 주세요";

                        QStringList parts;
                        if (failedMessage.contains(" : ")) {
                            parts = failedMessage.split(" : ");
                        } else if (failedMessage.contains(": ")) {
                            parts = failedMessage.split(": ");
                        }

                        if (parts.size() > 1) {
                            QString failedParts = parts[1].trimmed();
                            // "facepart::" 제거
                            failedParts = failedParts.replace("facepart::", "");
                            retryMessage = QString("사진을 다시 찍어 주세요 : %1").arg(failedParts);
                        }

                        QMessageBox::warning(this, "분석 실패", retryMessage);
                        qDebug() << "ROI detection failed:" << failedMessage;

                        // 카메라를 다시 활성화하여 재촬영 가능하게 함
                        ui->snapShotButton->setEnabled(true);
                        ui->snapShotButton->setText("사진 촬영");
                        ui->statusbar->showMessage("ROI 검출 실패 - 사진을 다시 촬영해주세요", 5000);

                        reply->deleteLater();
                        analysisNetworkManager->deleteLater();
                        return;
                    }
                }

                if (responseObj["status"].toString() == "success" &&
                    responseObj.contains("analysis_data")) {
                    
                    QJsonObject analysisData = responseObj["analysis_data"].toObject();
                    
                    qDebug() << "Analysis result received:" << analysisData;
                    
                    // 이전 기록 조회 (비교용)
                    DatabaseManager& dbManager = DatabaseManager::instance();
                    QList<AnalysisRecord> userHistory = dbManager.getUserHistory(currentUserName);
                    
                    AnalysisResultDialog *dialog = nullptr;
                    
                    // 이전 기록이 있으면 비교 모드로 표시
                    if (!userHistory.isEmpty()) {
                        QJsonObject previousData = userHistory.first().analysisData;
                        qDebug() << "Found previous record for comparison";
                        
                        // 비교 다이얼로그 생성
                        dialog = new AnalysisResultDialog(analysisData, previousData, currentUserName, this);
                    } else {
                        // 이전 기록이 없으면 일반 모드로 표시
                        qDebug() << "No previous record found, showing normal view";
                        dialog = new AnalysisResultDialog(analysisData, currentUserName, this);
                    }
                    
                    // 현재 분석 결과를 데이터베이스에 저장 (다이얼로그 표시 후)
                    if (dbManager.saveAnalysisResult(currentUserName, analysisData)) {
                        qDebug() << "Analysis result saved to database successfully";
                    } else {
                        qWarning() << "Failed to save analysis result to database";
                    }
                    
                    // 분석 결과 표시 전에 카메라 중지 및 오버레이 완전 삭제
                    if (isCameraRunning) {
                        stopCamera();
                        qDebug() << "Camera stopped before showing analysis result";
                    }

                    // 상태를 먼저 초기화하여 새로운 타이머들이 실행되지 않도록 함
                    isCameraRunning = false;
                    isNameEntered = false;

                    // 오버레이 완전히 삭제
                    hideOpenCVOverlay();
                    if (overlayWidget) {
                        overlayWidget->deleteLater();
                        overlayWidget = nullptr;
                        qDebug() << "OpenCV overlay completely removed";
                    }

                    // rpicam 창 숨기기
                    QProcess::execute("bash", QStringList() << "-c" << "wmctrl -c 'rpicam-hello' 2>/dev/null || true");
                    QProcess::execute("bash", QStringList() << "-c" << "pkill -f 'rpicam-hello' 2>/dev/null || true");

                    // MainWindow를 최상단으로 가져오기
                    this->show();
                    this->raise();
                    this->activateWindow();
                    this->setFocus();
                    qDebug() << "MainWindow brought to front for analysis dialog";

                    // 다이얼로그 표시 (최상단에 표시)
                    if (dialog) {
                        dialog->setWindowFlags(dialog->windowFlags() | Qt::WindowStaysOnTopHint);
                        dialog->show();
                        dialog->raise();
                        dialog->activateWindow();
                        dialog->exec();
                        dialog->deleteLater();

                        qDebug() << "Analysis dialog closed. Returning to initial view...";

                        // 모든 리소스 정리
                        if (isCameraRunning) {
                            stopCamera();
                        }
                        hideOpenCVOverlay();
                        if (overlayWidget) {
                            overlayWidget->deleteLater();
                            overlayWidget = nullptr;
                        }

                        // 상태 초기화
                        currentUserName.clear();
                        isNameEntered = false;
                        isCameraRunning = false;

                        // 모든 활성 타이머 취소 (restartPreview, showOpenCVOverlay 등)
                        QTimer::singleShot(0, [this]() {
                            // 이미 실행 중인 타이머들을 무시하도록 상태 재설정
                            isCameraRunning = false;
                        });

                        // 이름 입력 화면으로 돌아가기
                        QTimer::singleShot(200, this, &MainWindow::setupInitialView);
                        ui->statusbar->showMessage("분석 완료. 새 사용자를 위해 초기화되었습니다.", 3000);
                    }

                    ui->statusbar->showMessage("Analysis result displayed and saved", 3000);
                    
                } else if (responseObj["status"].toString() == "no_data") {
                    ui->statusbar->showMessage("No analysis data available yet", 3000);
                    qDebug() << "No analysis data available";
                } else {
                    ui->statusbar->showMessage("Failed to get analysis result", 3000);
                    qDebug() << "Failed to get analysis result:" << responseObj;
                }
            } else {
                ui->statusbar->showMessage("Response is not JSON object", 3000);
                qDebug() << "Response is not JSON object:" << jsonDoc;
            }
        } else {
            ui->statusbar->showMessage("Failed to connect to analysis server", 3000);
            qDebug() << "Network error:" << reply->errorString();
            qDebug() << "Error code:" << reply->error();
        }
        
        reply->deleteLater();
        analysisNetworkManager->deleteLater();
    });
    
    qDebug() << "Fetching analysis result from:" << raspUrl + "/get_analysis";
}

void MainWindow::initializeDatabase()
{
    DatabaseManager& dbManager = DatabaseManager::instance();
    if (!dbManager.initializeDatabase()) {
        QMessageBox::critical(this, "데이터베이스 오류", 
            "데이터베이스를 초기화할 수 없습니다.\n애플리케이션을 계속 사용할 수 있지만 분석 결과가 저장되지 않습니다.");
        return;
    }
    
    qDebug() << "Database initialized successfully";
    
    // 통계 정보 로드
    int totalUsers = dbManager.getTotalUsers();
    int totalRecords = dbManager.getTotalRecords();
    QDateTime lastActivity = dbManager.getLastActivity();
    
    QString statsMsg = QString("DB 통계 - 사용자: %1명, 분석기록: %2건")
                      .arg(totalUsers).arg(totalRecords);
    
    if (lastActivity.isValid()) {
        statsMsg += QString(", 마지막 활동: %1").arg(lastActivity.toString("MM-dd hh:mm"));
    }
    
    qDebug() << statsMsg;
}

void MainWindow::setupInitialView()
{
    qDebug() << "Setting up initial view - simplified version";

    // 카메라 완전히 중지
    if (isCameraRunning) {
        stopCamera();
    }

    // 상태 정리 - 오버레이 완전 제거 (강화된 정리)
    hideOpenCVOverlay();
    if (overlayWidget) {
        overlayWidget->hide();
        overlayWidget->close();
        overlayWidget->deleteLater();
        overlayWidget = nullptr;
        qDebug() << "Overlay widget completely deleted in setupInitialView";
    }

    // 추가로 모든 rpicam 관련 프로세스 정리
    QProcess::execute("bash", QStringList() << "-c" << "pkill -f 'rpicam-hello' 2>/dev/null || true");
    QProcess::execute("bash", QStringList() << "-c" << "wmctrl -c 'rpicam-hello' 2>/dev/null || true");

    // 상태 초기화
    currentUserName.clear();
    isNameEntered = false;
    isCameraRunning = false;

    // 기존 중앙 위젯의 내용만 변경 (새로 생성하지 않음)
    QWidget *central = centralWidget();
    if (central) {
        // 기존 레이아웃과 자식 위젯들 제거
        QLayout* layout = central->layout();
        if (layout) {
            QLayoutItem* item;
            while ((item = layout->takeAt(0)) != nullptr) {
                delete item->widget();
                delete item;
            }
            delete layout;
        }
    } else {
        central = new QWidget();
        setCentralWidget(central);
    }

    // 중앙 위젯에 이름 입력 단계 UI 추가
    QVBoxLayout *nameLayout = new QVBoxLayout(central);
    nameLayout->setAlignment(Qt::AlignCenter);
    nameLayout->setSpacing(20);
    
    // 환영 메시지
    QLabel *welcomeLabel = new QLabel("피부 개선 디스펜서에 오신 것을 환영합니다!");
    welcomeLabel->setAlignment(Qt::AlignCenter);
    welcomeLabel->setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; margin: 20px;");
    nameLayout->addWidget(welcomeLabel);
    
    // 안내 메시지
    QLabel *instructionLabel = new QLabel("피부 분석을 시작하기 전에 이름을 입력해주세요.");
    instructionLabel->setAlignment(Qt::AlignCenter);
    instructionLabel->setStyleSheet("font-size: 14px; color: #34495e; margin: 10px;");
    nameLayout->addWidget(instructionLabel);
    
    // 이름 입력 필드
    QLineEdit *nameLineEdit = new QLineEdit();
    nameLineEdit->setPlaceholderText("이름을 입력하세요 (2-20자)");
    nameLineEdit->setMaxLength(20);
    nameLineEdit->setMaximumWidth(300);
    nameLineEdit->setStyleSheet(
        "QLineEdit {"
        "    padding: 12px;"
        "    border: 2px solid #bdc3c7;"
        "    border-radius: 8px;"
        "    font-size: 14px;"
        "    background-color: white;"
        "}"
        "QLineEdit:focus {"
        "    border-color: #3498db;"
        "    outline: none;"
        "}"
    );
    nameLayout->addWidget(nameLineEdit, 0, Qt::AlignCenter);
    
    // 버튼 레이아웃
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    buttonLayout->setSpacing(15);
    
    QPushButton *startButton = new QPushButton("피부 분석 시작");
    startButton->setMinimumSize(150, 45);
    startButton->setStyleSheet(
        "QPushButton {"
        "    padding: 12px 24px;"
        "    background-color: #27ae60;"
        "    color: white;"
        "    border: none;"
        "    border-radius: 8px;"
        "    font-size: 14px;"
        "    font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "    background-color: #2ecc71;"
        "}"
        "QPushButton:pressed {"
        "    background-color: #239653;"
        "}"
    );
    
    QPushButton *exitButton = new QPushButton("종료");
    exitButton->setMinimumSize(100, 45);
    exitButton->setStyleSheet(
        "QPushButton {"
        "    padding: 12px 24px;"
        "    background-color: #e74c3c;"
        "    color: white;"
        "    border: none;"
        "    border-radius: 8px;"
        "    font-size: 14px;"
        "}"
        "QPushButton:hover {"
        "    background-color: #c0392b;"
        "}"
        "QPushButton:pressed {"
        "    background-color: #a93226;"
        "}"
    );
    
    buttonLayout->addStretch();
    buttonLayout->addWidget(startButton);
    buttonLayout->addWidget(exitButton);
    buttonLayout->addStretch();
    
    nameLayout->addLayout(buttonLayout);
    nameLayout->addStretch();
    
    // 연결 설정
    connect(startButton, &QPushButton::clicked, [this, nameLineEdit]() {
        QString name = nameLineEdit->text().trimmed();
        if (name.length() < 2) {
            QMessageBox::warning(this, "입력 오류", "이름은 최소 2자 이상 입력해주세요.");
            nameLineEdit->setFocus();
            return;
        }
        if (name.length() > 20) {
            QMessageBox::warning(this, "입력 오류", "이름은 최대 20자까지 입력 가능합니다.");
            nameLineEdit->setFocus();
            return;
        }
        
        currentUserName = name;
        isNameEntered = true;
        
        qDebug() << "User name entered:" << currentUserName;
        ui->statusbar->showMessage(QString("사용자: %1님, 환영합니다!").arg(currentUserName), 3000);
        
        // 카메라 화면으로 전환
        switchToCameraView();

        // 카메라 화면 진입 후 자동으로 카메라 시작
        QTimer::singleShot(500, this, &MainWindow::startCamera);
    });
    
    connect(exitButton, &QPushButton::clicked, this, &QWidget::close);
    
    connect(nameLineEdit, &QLineEdit::returnPressed, [startButton]() {
        startButton->click();
    });
    
    nameLineEdit->setFocus();
}

void MainWindow::switchToCameraView()
{
    // 새로운 중앙 위젯 생성 및 UI 설정
    QWidget *newCentralWidget = new QWidget();
    setCentralWidget(newCentralWidget);

    // rpicam 모드: 간단한 안내 라벨 사용
    cameraPreviewLabel = new QLabel(newCentralWidget);
    cameraPreviewLabel->setText("카메라 프리뷰는 별도 창에서 실행됩니다.\n\n'카메라 시작' 버튼을 클릭하세요.");
    cameraPreviewLabel->setAlignment(Qt::AlignCenter);
    cameraPreviewLabel->setStyleSheet(
        "QLabel {"
        "    background-color: #f8f9fa;"
        "    border: 2px solid #3498db;"
        "    border-radius: 8px;"
        "    padding: 20px;"
        "    font-size: 16px;"
        "    color: #2c3e50;"
        "}"
    );
    cameraPreviewLabel->setMinimumSize(640, 480);
    
    QPushButton *camStartButton = new QPushButton("카메라 시작", newCentralWidget);
    QPushButton *snapShotButton = new QPushButton("사진 촬영", newCentralWidget);
    
    // UI 포인터 업데이트
    ui->camStartButton = camStartButton;
    ui->snapShotButton = snapShotButton;
    ui->centralwidget = newCentralWidget;
    
    // 레이아웃 설정
    QVBoxLayout *mainLayout = new QVBoxLayout(newCentralWidget);
    mainLayout->setContentsMargins(15, 15, 15, 15);
    mainLayout->setSpacing(20);  // 간격을 늘려서 버튼과 카메라 분리

    if (cameraPreviewLabel) {
        // rpicam 모드: 카메라 영역을 Qt 윈도우의 50% 너비, 70% 높이로 설정
        QRect windowGeometry = this->geometry();
        int cameraWidth = windowGeometry.width() / 2;                    // 너비는 50%
        int cameraHeight = static_cast<int>(windowGeometry.height() * 0.7); // 높이는 70%

        cameraPreviewLabel->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        cameraPreviewLabel->setFixedSize(cameraWidth, cameraHeight); // 50% 너비, 70% 높이
        cameraPreviewLabel->setAlignment(Qt::AlignCenter);

        // 카메라 영역을 중앙에 배치하기 위한 레이아웃
        QHBoxLayout *cameraLayout = new QHBoxLayout();
        cameraLayout->addStretch();
        cameraLayout->addWidget(cameraPreviewLabel);
        cameraLayout->addStretch();

        mainLayout->addLayout(cameraLayout, 1);
    }

    // 카메라와 버튼 사이에 여백 추가
    mainLayout->addSpacing(30);

    // 버튼 레이아웃 (카메라 화면 밑에 배치)
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    buttonLayout->setSpacing(50);  // 버튼 간격 증가
    buttonLayout->addStretch();
    
    camStartButton->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    camStartButton->setMinimumSize(180, 50);  // 크기 더 증가
    camStartButton->setMaximumSize(300, 70);  // 최대 크기도 증가
    camStartButton->setStyleSheet(
        "QPushButton {"
        "    background-color: #3498db;"
        "    color: white;"
        "    border: none;"
        "    border-radius: 8px;"
        "    font-size: 14px;"
        "    font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "    background-color: #2980b9;"
        "}"
        "QPushButton:pressed {"
        "    background-color: #21618c;"
        "}"
    );
    buttonLayout->addWidget(camStartButton);

    snapShotButton->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    snapShotButton->setMinimumSize(180, 50);  // 크기 더 증가
    snapShotButton->setMaximumSize(300, 70);  // 최대 크기도 증가
    snapShotButton->setEnabled(false);
    snapShotButton->setStyleSheet(
        "QPushButton {"
        "    background-color: #27ae60;"
        "    color: white;"
        "    border: none;"
        "    border-radius: 8px;"
        "    font-size: 14px;"
        "    font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "    background-color: #2ecc71;"
        "}"
        "QPushButton:pressed {"
        "    background-color: #239653;"
        "}"
        "QPushButton:disabled {"
        "    background-color: #95a5a6;"
        "}"
    );
    buttonLayout->addWidget(snapShotButton);
    
    buttonLayout->addStretch();
    mainLayout->addLayout(buttonLayout, 0);
    
    // 시그널 연결
    connect(camStartButton, &QPushButton::clicked, this, &MainWindow::on_camStartButton_clicked);
    connect(snapShotButton, &QPushButton::clicked, this, &MainWindow::on_snapShotButton_clicked);

    // 상태바에 현재 사용자 표시
    ui->statusbar->showMessage(QString("현재 사용자: %1").arg(currentUserName), 5000);
}


void MainWindow::startGStreamerCamera()
{
    if (gstreamerProcess && gstreamerProcess->state() == QProcess::Running) {
        qDebug() << "Camera process already running";
        return;
    }

    // 프로세스 생성
    if (gstreamerProcess) {
        delete gstreamerProcess;
    }

    gstreamerProcess = new QProcess(this);

    // Qt 윈도우의 현재 위치와 크기 가져오기
    QRect windowGeometry = this->geometry();
    qDebug() << "Qt window geometry:" << windowGeometry;

    // 카메라 영역 위치 계산 (Qt 윈도우 내부의 카메라 프리뷰 영역)
    int cameraWidth = windowGeometry.width() / 2;                        // Qt 윈도우 너비의 50%
    int cameraHeight = static_cast<int>(windowGeometry.height() * 0.7);  // Qt 윈도우 높이의 70%
    int cameraX = windowGeometry.x() + (windowGeometry.width() - cameraWidth) / 2; // x축 중앙 정렬
    int cameraY = windowGeometry.y() + 50; // 상단 마진 (타이틀바 + 여백)

    // rpicam-hello를 정확한 위치에 실행
    QStringList args;
    args << "-t" << "0"          // 무한 실행
         << "--width" << QString::number(cameraWidth)
         << "--height" << QString::number(cameraHeight)
         << "--preview" << QString("%1,%2,%3,%4").arg(cameraX).arg(cameraY).arg(cameraWidth).arg(cameraHeight);

    // X11 환경에서 창 위치 조정을 위한 환경 변수 설정
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert("DISPLAY", ":0");
    gstreamerProcess->setProcessEnvironment(env);

    qDebug() << "Starting rpicam-hello at position:" << cameraX << cameraY;
    qDebug() << "Camera size:" << cameraWidth << "x" << cameraHeight;
    qDebug() << "Args:" << args.join(" ");

    gstreamerProcess->start("rpicam-hello", args);

    if (!gstreamerProcess->waitForStarted(3000)) {
        qDebug() << "Failed to start rpicam-hello:" << gstreamerProcess->errorString();
        ui->statusbar->showMessage("카메라 시작 실패", 3000);
        return;
    }

    qDebug() << "rpicam-hello started successfully";
    ui->statusbar->showMessage("카메라 시작됨", 2000);

    // 안내 라벨 숨기기 (카메라가 정확한 위치에 표시되므로)
    if (cameraPreviewLabel) {
        cameraPreviewLabel->hide();
    }

    // OpenCV 기반 오버레이 표시
    QTimer::singleShot(1500, this, &MainWindow::showOpenCVOverlay);
}

void MainWindow::stopGStreamerCamera()
{
    if (gstreamerProcess && gstreamerProcess->state() == QProcess::Running) {
        qDebug() << "Stopping camera process";

        // 정상 종료 시도
        gstreamerProcess->terminate();

        if (!gstreamerProcess->waitForFinished(3000)) {
            // 강제 종료
            qDebug() << "Force killing camera process";
            gstreamerProcess->kill();
            gstreamerProcess->waitForFinished(1000);
        }

        qDebug() << "Camera process stopped";
    }

    // rpicam-hello 관련 프로세스만 종료 (더 안전하게)
    QProcess *killProcess = new QProcess(this);
    killProcess->start("bash", QStringList() << "-c" << "pkill -f 'rpicam-hello' 2>/dev/null || true");
    killProcess->waitForFinished(2000);
    killProcess->deleteLater();
    qDebug() << "Killed rpicam-hello processes";

    // X11 창 강제 닫기 (libcamera 관련)
    QProcess *xkillProcess = new QProcess(this);
    xkillProcess->start("bash", QStringList() << "-c" << "wmctrl -c 'rpicam-hello' 2>/dev/null || true");
    xkillProcess->waitForFinished(1000);
    xkillProcess->deleteLater();

    // 추가적으로 libcamera 관련 창들 닫기
    QProcess *libcameraKill = new QProcess(this);
    libcameraKill->start("bash", QStringList() << "-c" << "wmctrl -l | grep -i libcamera | awk '{print $1}' | xargs -r wmctrl -i -c 2>/dev/null || true");
    libcameraKill->waitForFinished(1000);
    libcameraKill->deleteLater();

    ui->statusbar->showMessage("비디오 스트림 중지됨", 2000);

    // 안내 라벨 다시 표시
    if (cameraPreviewLabel) {
        cameraPreviewLabel->show();
        cameraPreviewLabel->setText("카메라가 중지되었습니다.\n\n'카메라 시작' 버튼을 클릭하세요.");
        cameraPreviewLabel->setStyleSheet(
            "QLabel {"
            "    background-color: #f8f9fa;"
            "    border: 2px solid #6c757d;"
            "    border-radius: 8px;"
            "    padding: 20px;"
            "    font-size: 16px;"
            "    color: #495057;"
            "}"
        );
    }

    // 오버레이 완전히 제거 (카메라 중지 후)
    hideOpenCVOverlay();
    if (overlayWidget) {
        overlayWidget->deleteLater();
        overlayWidget = nullptr;
        qDebug() << "Overlay widget deleted after camera stop";
    }
}

void MainWindow::captureWithRpicam()
{
    // 버튼 비활성화
    ui->snapShotButton->setEnabled(false);
    ui->snapShotButton->setText("촬영 중...");
    ui->statusbar->showMessage(QString("촬영 중... 사용자: %1").arg(currentUserName), 2000);

    // 1단계: 먼저 rpicam-hello 프로세스 중지
    if (gstreamerProcess && gstreamerProcess->state() == QProcess::Running) {
        qDebug() << "Stopping rpicam-hello for capture...";
        gstreamerProcess->terminate();
        gstreamerProcess->waitForFinished(2000);
    }

    // rpicam-hello 프로세스 안전하게 종료
    QProcess::execute("bash", QStringList() << "-c" << "pkill -f 'rpicam-hello' 2>/dev/null || true");
    QProcess::execute("bash", QStringList() << "-c" << "wmctrl -c 'rpicam-hello' 2>/dev/null || true");
    qDebug() << "Safely killed rpicam-hello before capture";

    // 2단계: rpicam-still로 고품질 이미지 캡처
    QProcess *captureProcess = new QProcess(this);
    QString tempFile = QString("/tmp/capture_%1.jpg").arg(QDateTime::currentMSecsSinceEpoch());

    QStringList args;
    args << "-o" << tempFile         // 출력 파일
         << "--width" << "1640"      // 고해상도
         << "--height" << "1232"
         << "--quality" << "95"      // 고품질
         << "--nopreview"            // 프리뷰 창 비활성화
         << "-t" << "1";             // 1ms만 실행 (즉시 촬영)

    qDebug() << "Capturing with rpicam-still:" << args.join(" ");

    // 비동기 실행
    connect(captureProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            [this, captureProcess, tempFile](int exitCode, QProcess::ExitStatus exitStatus) {
                Q_UNUSED(exitStatus);

                if (exitCode == 0 && QFile::exists(tempFile)) {
                    // 이미지 로드 및 서버 전송
                    QImage capturedImage(tempFile);
                    if (!capturedImage.isNull()) {
                        qDebug() << "Image captured successfully:" << capturedImage.size();
                        uploadImageToServer(capturedImage);

                        // 촬영 완료 후 rpicam-hello 다시 시작
                        QTimer::singleShot(1000, this, &MainWindow::restartPreview);
                    } else {
                        qDebug() << "Failed to load captured image";
                        ui->snapShotButton->setEnabled(true);
                        ui->snapShotButton->setText("사진 촬영");
                        QMessageBox::warning(this, "촬영 실패", "이미지를 불러올 수 없습니다.");

                        // 실패해도 프리뷰 재시작
                        QTimer::singleShot(1000, this, &MainWindow::restartPreview);
                    }

                    // 임시 파일 삭제
                    QFile::remove(tempFile);
                } else {
                    qDebug() << "rpicam-still capture failed with code:" << exitCode;
                    ui->snapShotButton->setEnabled(true);
                    ui->snapShotButton->setText("사진 촬영");
                    QMessageBox::warning(this, "촬영 실패", "카메라 촬영에 실패했습니다.");

                    // 실패해도 프리뷰 재시작
                    QTimer::singleShot(1000, this, &MainWindow::restartPreview);
                }

                captureProcess->deleteLater();
            });

    captureProcess->start("rpicam-still", args);
}

void MainWindow::restartPreview()
{
    // 카메라가 실행 중이고 사용자가 로그인된 상태인 경우에만 프리뷰 재시작
    if (isCameraRunning && isNameEntered && !currentUserName.isEmpty()) {
        qDebug() << "Restarting rpicam-hello preview after capture";

        // 기존 오버레이 제거
        hideOpenCVOverlay();
        if (overlayWidget) {
            overlayWidget->deleteLater();
            overlayWidget = nullptr;
        }

        startGStreamerCamera();
    }
}

// 단순한 Qt 기반 오버레이 위젯 (OpenCV로 생성한 이미지 사용)
class SimpleOverlay : public QWidget
{
private:
    int overlayWidth;
    int overlayHeight;

public:
    SimpleOverlay(int width, int height, QWidget *parent = nullptr) : QWidget(parent), overlayWidth(width), overlayHeight(height)
    {
        // 윈도우 플래그 설정: 항상 최상단, 프레임 없음, 입력 투명
        setWindowFlags(Qt::WindowStaysOnTopHint | Qt::FramelessWindowHint | Qt::Tool);
        setAttribute(Qt::WA_TranslucentBackground);  // 배경 투명
        setAttribute(Qt::WA_TransparentForMouseEvents); // 마우스 이벤트 투과

        // 크기 설정 (동적으로 설정)
        resize(overlayWidth, overlayHeight);
    }

protected:
    void paintEvent(QPaintEvent *event) override
    {
        Q_UNUSED(event);

        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);

        // 원 그리기 (카메라 창 안에 완전히 들어가도록 조정)
        int centerX = width() / 2;
        int centerY = height() / 2 - 100; // 위로 이동
        int radius = qMin(width(), height()) * 0.40; // 40% 크기로 조정

        // 원 스타일 설정
        QPen pen(QColor(0, 255, 0, 200), 4); // 반투명 초록색, 4픽셀 두께
        painter.setPen(pen);
        painter.setBrush(Qt::NoBrush); // 내부는 투명

        // 원 그리기
        painter.drawEllipse(centerX - radius, centerY - radius, radius * 2, radius * 2);

        // 텍스트 그리기 (새로운 카메라 크기에 맞춰 조정)
        QFont font("Arial", 20, QFont::Bold); // 폰트 크기를 20으로 조정
        painter.setFont(font);
        painter.setPen(QColor(0, 255, 0, 220)); // 진한 초록색

        QString text = "원에 얼굴을 맞춰주세요";
        QFontMetrics fm(font);
        QRect textRect = fm.boundingRect(text);

        int textX = centerX - textRect.width() / 2;
        int textY = height() - 100; // 원과 함께 아래로 이동하여 겹침 상태 유지

        // 텍스트 배경 (가독성 향상)
        QRect bgRect(textX - 15, textY - textRect.height() - 10, textRect.width() + 30, textRect.height() + 20);
        painter.fillRect(bgRect, QColor(0, 0, 0, 150)); // 반투명 검은 배경

        painter.drawText(textX, textY, text);
    }
};

void MainWindow::createOpenCVOverlay()
{
    // 기존 오버레이가 있으면 완전히 제거
    if (overlayWidget) {
        qDebug() << "Deleting existing overlay before creating new one";
        overlayWidget->hide();
        overlayWidget->deleteLater();
        overlayWidget = nullptr;
    }

    try {
        // Qt 윈도우 크기에 맞춰 오버레이 크기 설정 (50% 너비, 70% 높이)
        QRect windowGeometry = this->geometry();
        int cameraWidth = windowGeometry.width() / 2;                        // 너비 50%
        int cameraHeight = static_cast<int>(windowGeometry.height() * 0.7);  // 높이 70%

        overlayWidget = new SimpleOverlay(cameraWidth, cameraHeight);
        qDebug() << "Simple overlay created successfully with size:" << cameraWidth << "x" << cameraHeight;
    } catch (const std::exception& e) {
        qDebug() << "Failed to create overlay:" << e.what();
        overlayWidget = nullptr;
    }
}

void MainWindow::showOpenCVOverlay()
{
    // 카메라가 실행 중이고 사용자가 로그인된 상태에서만 오버레이 표시
    if (!isCameraRunning || !isNameEntered || currentUserName.isEmpty()) {
        qDebug() << "Overlay display cancelled: camera not running or user not logged in";
        return;
    }

    if (!overlayWidget) {
        createOpenCVOverlay();
    }

    if (!overlayWidget) {
        qDebug() << "Failed to create overlay widget";
        return;
    }

    try {
        // Qt 윈도우의 현재 위치 가져오기
        QRect windowGeometry = this->geometry();

        // 카메라 위치 계산 (startGStreamerCamera와 동일한 로직)
        int cameraWidth = windowGeometry.width() / 2;   // Qt 윈도우 너비의 50%
        int cameraX = windowGeometry.x() + (windowGeometry.width() - cameraWidth) / 2;
        int cameraY = windowGeometry.y() + 50;

        // 오버레이를 카메라와 같은 위치에 배치
        overlayWidget->move(cameraX, cameraY);
        overlayWidget->show();

        qDebug() << "Simple overlay shown at position:" << cameraX << cameraY;
    } catch (const std::exception& e) {
        qDebug() << "Failed to show overlay:" << e.what();
    }
}

void MainWindow::hideOpenCVOverlay()
{
    if (overlayWidget) {
        try {
            overlayWidget->hide();
            qDebug() << "Simple overlay hidden";
        } catch (const std::exception& e) {
            qDebug() << "Error hiding overlay:" << e.what();
        }
    }
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event);

    // 카메라가 실행 중일 때만 레이아웃 업데이트
    if (isCameraRunning && cameraPreviewLabel) {
        updateCameraLayout();
    }
}

void MainWindow::updateCameraLayout()
{
    if (!cameraPreviewLabel) return;

    // Qt 윈도우 크기에 맞춰 카메라 프리뷰 라벨 크기 업데이트 (50% 너비, 70% 높이)
    QRect windowGeometry = this->geometry();
    int cameraWidth = windowGeometry.width() / 2;                        // 너비 50%
    int cameraHeight = static_cast<int>(windowGeometry.height() * 0.7);  // 높이 70%

    cameraPreviewLabel->setFixedSize(cameraWidth, cameraHeight);

    // 카메라가 실행 중이면 실제 카메라 윈도우도 업데이트
    if (isCameraRunning && gstreamerProcess && gstreamerProcess->state() == QProcess::Running) {
        // 기존 카메라 프로세스 중지
        stopGStreamerCamera();

        // 새 크기로 카메라 재시작 (짧은 지연 후)
        QTimer::singleShot(500, this, &MainWindow::startGStreamerCamera);
    }
}

