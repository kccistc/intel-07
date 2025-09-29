// mainwindow.h
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <memory>

// Qt Multimedia 헤더들 제거 (rpicam 사용)
#include <QTimer>
#include <QProcess>
#include <QLabel>
#include <QWidget>
#include <QResizeEvent>
#include "analysisresultdialog.h"
#include "nameinputdialog.h"
#include "databasemanager.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_camStartButton_clicked();
    void on_snapShotButton_clicked();
    void onUploadFinished(QNetworkReply* reply);
    void onUploadProgress(qint64 bytesSent, qint64 bytesTotal);
    void fetchAnalysisResult();
    void initializeDatabase();
    void startGStreamerCamera(); // rpicam 카메라 시작
    void stopGStreamerCamera();  // rpicam 카메라 중지
    void captureWithRpicam();    // rpicam-still로 직접 촬영
    void restartPreview();       // 프리뷰 재시작
    void createOpenCVOverlay();  // OpenCV 기반 오버레이 생성
    void showOpenCVOverlay();    // OpenCV 오버레이 표시
    void hideOpenCVOverlay();    // OpenCV 오버레이 숨김

private:
    Ui::MainWindow *ui;

    std::unique_ptr<QNetworkAccessManager> networkManager;

    // 서버 설정
    QString serverUrl;
    QString serverEndpoint;
    QString raspUrl;

    bool isCameraRunning;
    QString currentUserName;
    bool isNameEntered;

    // rpicam 기반 카메라
    QLabel* cameraPreviewLabel;
    QProcess* gstreamerProcess;

    // OpenCV 기반 오버레이
    QWidget* overlayWidget;
    QTimer* overlayTimer;
    
    void startCamera();
    void stopCamera();
    void uploadImageToServer(const QImage& image);
    void loadServerConfig();
    void setupUILayout();
    void setupInitialView();
    void switchToCameraView();
    void setupWindowSizing();
    void updateCameraLayout(); // 카메라 레이아웃 업데이트

protected:
    void resizeEvent(QResizeEvent *event) override; // 윈도우 크기 변경 이벤트
};

#endif // MAINWINDOW_H
