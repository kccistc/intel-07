#ifndef MJPGVIEW_H
#define MJPGVIEW_H
#include "ui_mjpgview.h"
#include "warning.h"
#include <QWidget>
#include <QUrl>
#include <QTimer>
#include <QLabel>
#include <QVBoxLayout>
#include <QElapsedTimer>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/rtsp/gstrtsptransport.h>
#include <gst/video/video.h>

class MjpgView : public QWidget
{
    Q_OBJECT
public:
    // explicit MjpgView(QWidget *parent = nullptr);
    // 주소를 인자로 받는 생성자 추가!
    explicit MjpgView(const QUrl& url, QWidget *parent = nullptr);
    ~MjpgView();

public slots:
    void setUrl(const QUrl &url)        { m_url = url; }
    void setUrlString(const QString &s) { m_url = QUrl(s); }
    void play();
    void stop();

private slots:
    void pollSample();   // appsink에서 프레임 뽑기
    void pollBus();      // 버스 메시지(에러 등) 체크
    void watchdogCheck();
    void doReconnect();
    void updateRecvDataSlot(QString line);

private:
    Ui::RtspView *ui;

    bool buildPipeline();               // uridecodebin 기반(주로 RTSP/일반 HTTP)
    bool buildPipelineHttpMjpeg();      // HTTP MJPEG 전용(Flask/OpenCV 등)
    void teardownPipeline();
    void setStatus(const QString &s);
    void scheduleReconnect();           // 백오프 스케줄
    void resetBackoff();                // 첫 프레임 수신 시

private:
    // UI(.ui 없이 코드로 구성)

    QLabel *m_videoLabel = nullptr;
    QLabel *m_statusLabel = nullptr;

    // // URL (기본 고정값)
    // QUrl m_url { QStringLiteral("http://192.168.0.122:8081/processed") };
    QUrl m_url; // ★ 기본값 삭제, 생성자에서 세팅


    // 타이머
    QTimer m_videoPoll{this};
    QTimer m_busPoll{this};
    QTimer m_watchdog{this};            // 프레임 감시
    QTimer m_reconnectTimer{this};      // 재연결 예약용

    // Watchdog/Backoff
    QElapsedTimer m_lastFrameTs;
    int m_stallMs = 3000;               // 3초 동안 프레임 없으면 재연결
    int m_backoffMs = 1000;             // 시작 백오프 1초
    const int m_backoffMaxMs = 10000;   // 최대 10초
    bool m_firstFrameSeen = false;

    // GStreamer 요소
    GstElement *m_pipeline = nullptr;
    GstElement *m_source   = nullptr;   // uridecodebin 또는 souphttpsrc
    GstElement *m_convert  = nullptr;   // videoconvert
    GstElement *m_filter   = nullptr;   // (사용 안할 수도 있음)
    GstElement *m_sink     = nullptr;   // appsink

    // 캡처된 프레임의 캡스/비디오정보
    GstVideoInfo m_vi{};
    Warning *warningManager;
};

#endif // MJPGVIEW_H
