#include "mjpgview.h"
#include "warning.h"

#include <QImage>
#include <QPixmap>
#include <QDebug>
#include <QSizePolicy>

// 에러 문자열 헬퍼
static QString gstMessageToString(GstMessage *msg) {
    if (!msg) return "unknown";
    GError *err = nullptr; gchar *dbg = nullptr;
    gst_message_parse_error(msg, &err, &dbg);
    QString s = err ? QString::fromUtf8(err->message) : "unknown";
    if (err) g_error_free(err);
    if (dbg) g_free(dbg);
    return s;
}

// MjpgView::MjpgView(QWidget *parent)
//     : QWidget(parent)
MjpgView::MjpgView(const QUrl& url, QWidget *parent)
    : QWidget(parent), m_url(url)
{

    auto *layout = new QVBoxLayout(this);

    m_videoLabel = new QLabel(this);
    m_statusLabel = new QLabel(this);

    m_videoLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_videoLabel->setAlignment(Qt::AlignCenter);
    m_videoLabel->setText(QStringLiteral("No Video"));
    m_statusLabel->setText(QStringLiteral("Idle"));

    layout->addWidget(m_videoLabel, /*stretch*/1);
    layout->addWidget(m_statusLabel, /*stretch*/0);
    setLayout(layout);

    // GStreamer 전역 초기화(앱에서 1회면 충분)
    static bool inited = false;
    if (!inited) { int argc=0; char**argv=nullptr; gst_init(&argc, &argv); inited = true; }

    // 타이머 설정
    m_videoPoll.setInterval(15);    // ~66fps 폴링
    m_busPoll.setInterval(100);
    m_watchdog.setInterval(1000);   // 1초마다 체크
    m_reconnectTimer.setSingleShot(true);

    connect(&m_videoPoll, &QTimer::timeout, this, &MjpgView::pollSample);
    connect(&m_busPoll,   &QTimer::timeout, this, &MjpgView::pollBus);
    connect(&m_watchdog,  &QTimer::timeout, this, &MjpgView::watchdogCheck);
    connect(&m_reconnectTimer, &QTimer::timeout, this, &MjpgView::doReconnect);

    setStatus("Idle");
}

MjpgView::~MjpgView()
{
    stop();
}

void MjpgView::play()
{
    if (m_pipeline) stop();

    // HTTP면 MJPEG로 가정(사용자 환경에서 검증됨)
    const QString scheme = m_url.scheme().toLower();
    bool ok = false;
    if (scheme == "http" || scheme == "https") {
        ok = buildPipelineHttpMjpeg();
    } else {
        ok = buildPipeline();
    }
    if (!ok) {
        setStatus("Failed to build pipeline");
        scheduleReconnect();  // 즉시 재시도 스케줄
        return;
    }

    m_firstFrameSeen = false;
    m_lastFrameTs.restart();

    gst_element_set_state(m_pipeline, GST_STATE_PLAYING);

    m_videoPoll.start();
    m_busPoll.start();
    m_watchdog.start();

    setStatus(QStringLiteral("Playing (awaiting frames)"));
}

void MjpgView::stop()
{
    m_videoPoll.stop();
    m_busPoll.stop();
    m_watchdog.stop();
    m_reconnectTimer.stop();

    if (m_pipeline) gst_element_set_state(m_pipeline, GST_STATE_NULL);
    teardownPipeline();
    setStatus(QStringLiteral("Stopped"));
    resetBackoff(); // 다음 시작을 위해 초기화
}

bool MjpgView::buildPipeline()
{
    m_pipeline = gst_pipeline_new("qt-stream-pipeline");
    m_source   = gst_element_factory_make("uridecodebin", "src");
    m_convert  = gst_element_factory_make("videoconvert", "convert");
    m_sink     = gst_element_factory_make("appsink", "sink");

    if (!m_pipeline || !m_source || !m_convert || !m_sink) {
        qWarning() << "Failed to create GStreamer elements. Check plugins.";
        return false;
    }

    QByteArray uri = m_url.toString().toUtf8();
    g_object_set(G_OBJECT(m_source),
                 "uri", uri.constData(),
                 "use-buffering", FALSE,
                 NULL);

    // RTSP일 경우만 rtspsrc child에 TCP/저지연 옵션 전달
    if (m_url.scheme().compare("rtsp", Qt::CaseInsensitive) == 0) {
        gst_child_proxy_set(GST_CHILD_PROXY(m_source),
                            "source::latency", 0,
                            "source::protocols", GST_RTSP_LOWER_TRANS_TCP,
                            NULL);
    }

    // appsink에 BGRA/BGRx 중 하나를 사용. 여기서는 BGRx로 고정.
    GstCaps *caps = gst_caps_from_string("video/x-raw,format=BGRx");
    g_object_set(G_OBJECT(m_sink),
                 "emit-signals", FALSE,
                 "sync", FALSE,
                 "max-buffers", 2,
                 "drop", TRUE,
                 "caps", caps,
                 NULL);
    gst_caps_unref(caps);

    gst_bin_add_many(GST_BIN(m_pipeline), m_source, m_convert, m_sink, NULL);

    if (!gst_element_link(m_convert, m_sink)) {
        qWarning() << "Link failed (convert->sink)";
        return false;
    }

    g_signal_connect(m_source, "pad-added",
                     G_CALLBACK(+[] (GstElement* /*src*/, GstPad* new_pad, gpointer user_data) {
                         GstElement* convert = static_cast<GstElement*>(user_data);
                         GstPad* sinkpad = gst_element_get_static_pad(convert, "sink");
                         if (!sinkpad) return;
                         if (gst_pad_is_linked(sinkpad)) { gst_object_unref(sinkpad); return; }

                         // 비디오 패드만 연결
                         GstCaps *caps = gst_pad_query_caps(new_pad, NULL);
                         bool is_video = false;
                         if (caps) {
                             GstStructure *st = gst_caps_get_structure(caps, 0);
                             const gchar *name = gst_structure_get_name(st);
                             if (name && g_str_has_prefix(name, "video/")) is_video = true;
                             gst_caps_unref(caps);
                         }
                         if (is_video) {
                             if (gst_pad_link(new_pad, sinkpad) != GST_PAD_LINK_OK) {
                                 g_printerr("Failed to link uridecodebin -> videoconvert\n");
                             }
                         }
                         gst_object_unref(sinkpad);
                     }), m_convert);

    gst_video_info_init(&m_vi);
    return true;
}

bool MjpgView::buildPipelineHttpMjpeg()
{
    m_pipeline = gst_pipeline_new("qt-mjpeg-http-pipeline");
    GstElement *httpsrc = gst_element_factory_make("souphttpsrc", "src");
    GstElement *demux   = gst_element_factory_make("multipartdemux", "demux");
    GstElement *q1      = gst_element_factory_make("queue", "q1");
    GstElement *jpegdec = gst_element_factory_make("jpegdec", "jpegdec");
    GstElement *q2      = gst_element_factory_make("queue", "q2");
    m_convert  = gst_element_factory_make("videoconvert", "convert");
    m_sink     = gst_element_factory_make("appsink", "sink");

    if (!m_pipeline || !httpsrc || !demux || !q1 || !jpegdec || !q2 || !m_convert || !m_sink) {
        qWarning() << "Failed to create HTTP MJPEG elements. Check plugins.";
        return false;
    }

    g_object_set(G_OBJECT(httpsrc),
                 "location", m_url.toString().toUtf8().constData(),
                 "is-live", TRUE,
                 // 옵션: 타임아웃(초). 서버가 죽었을 때 더 빨리 에러를 내고 재연결 루프가 돌게 함.
                 "timeout", 5,
                 NULL);

    // appsink에 BGRx 고정(caps) → Qt RGB32와 잘 맞음
    GstCaps *caps = gst_caps_from_string("video/x-raw,format=BGRx");
    g_object_set(G_OBJECT(m_sink),
                 "emit-signals", FALSE,
                 "sync", FALSE,
                 "max-buffers", 2,
                 "drop", TRUE,
                 "caps", caps,
                 NULL);
    gst_caps_unref(caps);

    gst_bin_add_many(GST_BIN(m_pipeline), httpsrc, demux, q1, jpegdec, q2, m_convert, m_sink, NULL);

    if (!gst_element_link(httpsrc, demux)) {
        qWarning() << "Link failed (httpsrc->multipartdemux)";
        return false;
    }

    // demux는 동적 pad → q1에 연결
    g_signal_connect(demux, "pad-added",
                     G_CALLBACK(+[] (GstElement*, GstPad* new_pad, gpointer user_data) {
                         GstElement* q1 = static_cast<GstElement*>(user_data);
                         GstPad* sinkpad = gst_element_get_static_pad(q1, "sink");
                         if (!sinkpad) return;
                         if (!gst_pad_is_linked(sinkpad)) {
                             if (gst_pad_link(new_pad, sinkpad) != GST_PAD_LINK_OK) {
                                 g_printerr("Failed to link multipartdemux -> queue\n");
                             }
                         }
                         gst_object_unref(sinkpad);
                     }), q1);

    if (!gst_element_link_many(q1, jpegdec, q2, m_convert, m_sink, NULL)) {
        qWarning() << "Link failed (queue->jpegdec->queue->convert->sink)";
        return false;
    }

    gst_video_info_init(&m_vi);
    return true;
}

void MjpgView::teardownPipeline()
{
    if (!m_pipeline) return;
    gst_object_unref(GST_OBJECT(m_pipeline));
    m_pipeline = nullptr;
    m_source = m_convert = m_filter = m_sink = nullptr;
    gst_video_info_init(&m_vi);
}

void MjpgView::pollSample()
{
    if (!m_sink) return;

    // 조금 기다리며 수신(프레임 타이밍에 여유)
    GstSample *sample = gst_app_sink_try_pull_sample(GST_APP_SINK(m_sink), 50 * GST_MSECOND);
    if (!sample) return;

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    GstCaps   *caps   = gst_sample_get_caps(sample);
    if (!buffer || !caps) { gst_sample_unref(sample); return; }

    // caps에서 stride, w, h 파싱
    GstVideoInfo vi;
    if (!gst_video_info_from_caps(&vi, caps)) {
        gst_sample_unref(sample);
        return;
    }
    const int w = GST_VIDEO_INFO_WIDTH(&vi);
    const int h = GST_VIDEO_INFO_HEIGHT(&vi);
    const int stride = GST_VIDEO_INFO_PLANE_STRIDE(&vi, 0);

    GstMapInfo map;
    if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        // BGRx를 Qt RGB32로 표시 (B,G,R,X 순서 호환)
        QImage img(reinterpret_cast<const uchar*>(map.data), w, h, stride, QImage::Format_RGB32);
        if (!img.isNull()) {
            if (!m_firstFrameSeen) {
                m_firstFrameSeen = true;
                resetBackoff();
                setStatus(QStringLiteral("Playing"));
            }
            m_videoLabel->setPixmap(QPixmap::fromImage(img.copy())); // copy()로 버퍼 생명주기 분리
            m_lastFrameTs.restart();
        }
        gst_buffer_unmap(buffer, &map);
    }
    gst_sample_unref(sample);
}

void MjpgView::pollBus()
{
    if (!m_pipeline) return;
    GstBus *bus = gst_element_get_bus(m_pipeline);
    if (!bus) return;

    while (true) {
        GstMessage *msg = gst_bus_pop_filtered(bus,
                                               (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS | GST_MESSAGE_STATE_CHANGED));
        if (!msg) break;

        switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_ERROR: {
            QString errStr = gstMessageToString(msg);
            QString src    = QString::fromUtf8(GST_OBJECT_NAME(msg->src));
            qWarning() << "GStreamer ERROR from" << src << ":" << errStr;
            setStatus("Error: " + src + " - " + errStr);
            scheduleReconnect();
            break;
        }
        case GST_MESSAGE_EOS:
            setStatus("EOS");
            scheduleReconnect();
            break;
        case GST_MESSAGE_STATE_CHANGED:
            // 필요 시 상태 로깅
            break;
        default: break;
        }
        gst_message_unref(msg);
    }
    gst_object_unref(bus);
}

void MjpgView::watchdogCheck()
{
    if (!m_pipeline) return;

    // 재생 중인데 일정 시간 프레임이 없으면(서버 꺼짐/정지/네트워크 문제) 재연결
    if (m_lastFrameTs.isValid()) {
        if (m_lastFrameTs.elapsed() > m_stallMs) {
            qWarning() << "[RtspView] Stall detected. No frames for" << m_lastFrameTs.elapsed() << "ms. Reconnecting...";
            setStatus(QStringLiteral("Stalled. Reconnecting..."));
            scheduleReconnect();
        }
    } else {
        // 시작 이후 아직 프레임 한 번도 못 받았고, 파이프라인은 돌아가는 중 → 재시도
        static int firstWait = 0;
        firstWait += m_watchdog.interval();
        if (firstWait > m_stallMs) {
            firstWait = 0;
            qWarning() << "[RtspView] No first frame yet. Reconnecting...";
            setStatus(QStringLiteral("Waiting for stream... Reconnecting..."));
            scheduleReconnect();
        }
    }
}

void MjpgView::scheduleReconnect()
{
    // 중복 예약 방지
    if (m_reconnectTimer.isActive()) return;

    // 파이프라인 정지
    if (m_pipeline) {
        gst_element_set_state(m_pipeline, GST_STATE_NULL);
        teardownPipeline();
    }

    setStatus(QString("Reconnecting in %1 ms...").arg(m_backoffMs));
    m_reconnectTimer.start(m_backoffMs);

    // 백오프 증가(최대치 제한)
    m_backoffMs = qMin(m_backoffMs * 2, m_backoffMaxMs);
}

void MjpgView::doReconnect()
{
    qWarning() << "[RtspView] Reconnect attempt...";
    play(); // 위에서 scheme에 따라 적절한 파이프라인 재생성
}

void MjpgView::resetBackoff()
{
    m_backoffMs = 1000;
    m_lastFrameTs.restart(); // 이후 watchdog 기준
}

void MjpgView::setStatus(const QString &s)
{
    if (m_statusLabel) m_statusLabel->setText(s);
}

void MjpgView::updateRecvDataSlot(QString line)
{
    qDebug() << line;
    line = line.trimmed(); // \r, \n, 공백 제거(끝)

    QStringList parts = line.split("|");    // | 기준으로 분리

    if(parts.size() < 4)
    {
        qDebug() << "파싱 오류 : 토큰 부족" << parts;
        return;
    }

    QString currentUser = parts[1];     // 새로 들어온 사용자 이름

    // ✅ RFID
    if (parts[2] == "RFID") {
        if (parts[3] == "NO") {
            QString msg = QString("[%1] RFID를 태그하지 않았습니다!").arg(currentUser);
            warningManager->showWarning(warningManager->rfidBoxes, currentUser,
                                       "출근 인식 경고", msg, warningManager->rfidList, QMessageBox::Critical);
        }
        else if (parts[3] == "YES") {
            if (warningManager->rfidBoxes.contains(currentUser)) {
                warningManager->rfidBoxes[currentUser]->close();    // 경고창 닫기
                warningManager->rfidBoxes.remove(currentUser);      // Map에서 key값 삭제
            }
            warningManager->rfidList.removeAll(currentUser); // 리스트에서 제거
        }
    }

    // ✅ AWAY
    if (parts[2] == "AWAY") {
        if (parts[3] == "ON") {
            QString msg = QString("[%1] 자리 비움 시간 5분 경과").arg(currentUser);
            warningManager->showWarning(warningManager->awayBoxes, currentUser,
                                       "자리 비움 경고", msg, warningManager->awayList);
        }
        else if (parts[3] == "OFF") {
            // 창은 닫지 않고 유지
            warningManager->awayList.removeAll(currentUser); // 리스트에서만 제거
        }
    }
}
