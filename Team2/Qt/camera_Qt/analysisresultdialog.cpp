#include "analysisresultdialog.h"
#include "nameinputdialog.h"
#include <QApplication>
#include <QFont>
#include <QFrame>
#include <QJsonArray>
#include <QMetaObject>
#include <QDebug>
#include <QtCharts/QChart>
#include <QtCharts/QBarSeries>
#include <QtCharts/QBarSet>
#include <QtCharts/QValueAxis>
#include <QtCharts/QBarCategoryAxis>
#include <QtCharts/QChartView>

QT_CHARTS_USE_NAMESPACE

AnalysisResultDialog::AnalysisResultDialog(QJsonObject analysisData, QString userName, QWidget *parent)
    : QDialog(parent), analysisData(analysisData), userName(userName), isComparison(false), recommendationViewed(false)
{
    setWindowTitle(QString("피부 분석 결과 - %1").arg(userName.isEmpty() ? "사용자" : userName));
    setModal(true);
    
    // 전체화면으로 설정
    QScreen *screen = QApplication::primaryScreen();
    if (screen) {
        QRect screenGeometry = screen->availableGeometry();
        resize(screenGeometry.width(), screenGeometry.height());
        move(screenGeometry.x(), screenGeometry.y());
    } else {
        setMinimumSize(480, 400);
        resize(600, 550);
    }
    
    setupUI();
    displayAnalysisData(analysisData);
}

AnalysisResultDialog::AnalysisResultDialog(QJsonObject currentData, QJsonObject previousData, QString userName, QWidget *parent)
    : QDialog(parent), analysisData(currentData), previousData(previousData), userName(userName), isComparison(true), recommendationViewed(false)
{
    setWindowTitle(QString("피부 분석 비교 - %1").arg(userName.isEmpty() ? "사용자" : userName));
    setModal(true);
    
    // 전체화면으로 설정
    QScreen *screen = QApplication::primaryScreen();
    if (screen) {
        QRect screenGeometry = screen->availableGeometry();
        resize(screenGeometry.width(), screenGeometry.height());
        move(screenGeometry.x(), screenGeometry.y());
    } else {
        setMinimumSize(550, 450);
        resize(700, 600);
    }
    
    setupUI();
    displayComparisonData(currentData, previousData);
}

void AnalysisResultDialog::setupUI()
{
    mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(8);
    mainLayout->setContentsMargins(15, 15, 15, 15);
    
    // 제목 라벨
    QString titleText = isComparison ? "피부 분석 비교" : "피부 분석 결과";
    QLabel *titleLabel = new QLabel(titleText);
    QFont titleFont = titleLabel->font();
    titleFont.setPointSize(14);
    titleFont.setBold(true);
    titleLabel->setFont(titleFont);
    titleLabel->setAlignment(Qt::AlignCenter);
    titleLabel->setStyleSheet("color: #2c3e50; margin: 8px 0px;");
    mainLayout->addWidget(titleLabel);
    
    // 사용자 이름 라벨 (있는 경우)
    if (!userName.isEmpty()) {
        QLabel *userLabel = new QLabel(QString("사용자: %1").arg(userName));
        QFont userFont = userLabel->font();
        userFont.setPointSize(10);
        userLabel->setFont(userFont);
        userLabel->setAlignment(Qt::AlignCenter);
        userLabel->setStyleSheet("color: #7f8c8d; margin-bottom: 12px;");
        mainLayout->addWidget(userLabel);
    }
    
    // 비교 모드일 때 안내 라벨
    if (isComparison) {
        QLabel *infoLabel = new QLabel("현재 ↔ 이전 (변화량)");
        QFont infoFont = infoLabel->font();
        infoFont.setPointSize(9);
        infoLabel->setFont(infoFont);
        infoLabel->setAlignment(Qt::AlignCenter);
        infoLabel->setStyleSheet("color: #34495e; background-color: #ecf0f1; padding: 4px; border-radius: 3px; margin-bottom: 10px;");
        mainLayout->addWidget(infoLabel);
    }
    
    // 스크롤 영역 설정
    scrollArea = new QScrollArea();
    scrollArea->setWidgetResizable(true);
    scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    
    contentWidget = new QWidget();
    scrollArea->setWidget(contentWidget);
    
    mainLayout->addWidget(scrollArea);
    
    // 닫기 버튼
    closeButton = new QPushButton("닫기");
    closeButton->setMinimumHeight(32);
    closeButton->setStyleSheet(
        "QPushButton {"
        "  background-color: #3498db;"
        "  border: none;"
        "  color: white;"
        "  padding: 6px 12px;"
        "  border-radius: 4px;"
        "  font-size: 12px;"
        "}"
        "QPushButton:hover {"
        "  background-color: #2980b9;"
        "}"
        "QPushButton:pressed {"
        "  background-color: #21618c;"
        "}"
    );
    
    connect(closeButton, &QPushButton::clicked, [this]() {
        this->accept();
        // 분석 결과 다이얼로그 닫기 후 이름 입력으로 이동
        showNameInputDialog();
    });
    mainLayout->addWidget(closeButton);
}

void AnalysisResultDialog::displayAnalysisData(const QJsonObject &data)
{
    QVBoxLayout *contentLayout = new QVBoxLayout(contentWidget);
    contentLayout->setSpacing(15);
    contentLayout->setContentsMargins(10, 10, 10, 10);
    
    // 통합 막대그래프 생성
    QChartView *chartView = createUnifiedBarChart(data);
    contentLayout->addWidget(chartView);
    
    // 추천 화장품 버튼 추가
    QPushButton *recommendationButton = new QPushButton("추천 화장품 보기");
    recommendationButton->setMinimumHeight(40);
    recommendationButton->setStyleSheet(
        "QPushButton {"
        "  background-color: #27ae60;"
        "  border: none;"
        "  color: white;"
        "  padding: 8px 16px;"
        "  border-radius: 6px;"
        "  font-size: 14px;"
        "  font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "  background-color: #2ecc71;"
        "}"
        "QPushButton:pressed {"
        "  background-color: #229954;"
        "}"
    );
    
    connect(recommendationButton, &QPushButton::clicked, [this, data]() {
        showRecommendationsDialog(data);
        recommendationViewed = true; // 추천을 봤다고 표시
    });
    
    contentLayout->addWidget(recommendationButton);
    contentLayout->addStretch();
}

QWidget* AnalysisResultDialog::createRegionWidget(const QString &regionName, const QJsonObject &regionData)
{
    QGroupBox *groupBox = new QGroupBox(regionName);
    groupBox->setStyleSheet(
        "QGroupBox {"
        "  font-weight: bold;"
        "  font-size: 12px;"
        "  color: #2c3e50;"
        "  border: 2px solid #bdc3c7;"
        "  border-radius: 6px;"
        "  margin-top: 8px;"
        "  padding-top: 8px;"
        "}"
        "QGroupBox::title {"
        "  subcontrol-origin: margin;"
        "  left: 8px;"
        "  padding: 0 4px 0 4px;"
        "}"
    );
    
    QGridLayout *gridLayout = new QGridLayout(groupBox);
    gridLayout->setSpacing(8);
    gridLayout->setContentsMargins(10, 15, 10, 10);
    
    int row = 0;
    int col = 0;
    const int maxCols = 2; // 한 줄에 최대 2개 메트릭
    
    // 메트릭들을 그리드로 배치
    for (auto it = regionData.begin(); it != regionData.end(); ++it) {
        if (it.value().isDouble()) {
            int value = static_cast<int>(it.value().toDouble());
            QWidget *metricWidget = createMetricWidget(it.key(), value);
            gridLayout->addWidget(metricWidget, row, col);
            
            col++;
            if (col >= maxCols) {
                col = 0;
                row++;
            }
        }
    }
    
    return groupBox;
}

QWidget* AnalysisResultDialog::createMetricWidget(const QString &metricName, int value)
{
    QWidget *widget = new QWidget();
    widget->setMinimumHeight(120);
    widget->setMinimumWidth(100);
    widget->setStyleSheet("background-color: #f8f9fa; border-radius: 6px; padding: 8px;");
    
    QVBoxLayout *layout = new QVBoxLayout(widget);
    layout->setSpacing(6);
    layout->setContentsMargins(10, 8, 10, 8);
    
    // 메트릭 이름
    QLabel *nameLabel = new QLabel(getMetricDescription(metricName));
    QFont nameFont = nameLabel->font();
    nameFont.setPointSize(10);
    nameFont.setBold(true);
    nameLabel->setFont(nameFont);
    nameLabel->setStyleSheet("color: #2c3e50;");
    nameLabel->setAlignment(Qt::AlignCenter);
    layout->addWidget(nameLabel);
    
    // 수직 진행률 바 컨테이너
    QWidget *barContainer = new QWidget();
    barContainer->setFixedHeight(60);
    barContainer->setMinimumWidth(40);
    
    QHBoxLayout *barLayout = new QHBoxLayout(barContainer);
    barLayout->setContentsMargins(0, 0, 0, 0);
    barLayout->addStretch();
    
    // 수직 진행률 바
    QProgressBar *progressBar = new QProgressBar();
    progressBar->setOrientation(Qt::Vertical);
    progressBar->setRange(0, 100);
    progressBar->setValue(value);
    progressBar->setTextVisible(false);
    progressBar->setFixedWidth(20);
    progressBar->setFixedHeight(60);
    
    QColor barColor = getScoreColor(value);
    progressBar->setStyleSheet(QString(
        "QProgressBar {"
        "  border: 1px solid #bdc3c7;"
        "  border-radius: 3px;"
        "  background-color: #ecf0f1;"
        "}"
        "QProgressBar::chunk {"
        "  background-color: %1;"
        "  border-radius: 2px;"
        "}"
    ).arg(barColor.name()));
    
    barLayout->addWidget(progressBar);
    barLayout->addStretch();
    
    layout->addWidget(barContainer);
    
    // 점수와 설명
    QLabel *scoreLabel = new QLabel(QString("%1").arg(value));
    QFont scoreFont = scoreLabel->font();
    scoreFont.setPointSize(12);
    scoreFont.setBold(true);
    scoreLabel->setFont(scoreFont);
    scoreLabel->setStyleSheet(QString("color: %1;").arg(barColor.name()));
    scoreLabel->setAlignment(Qt::AlignCenter);
    layout->addWidget(scoreLabel);
    
    QLabel *descLabel = new QLabel(getScoreDescription(value));
    QFont descFont = descLabel->font();
    descFont.setPointSize(9);
    descLabel->setFont(descFont);
    descLabel->setStyleSheet("color: #7f8c8d;");
    descLabel->setAlignment(Qt::AlignCenter);
    layout->addWidget(descLabel);
    
    return widget;
}

QString AnalysisResultDialog::getMetricDescription(const QString &metricName)
{
    if (metricName == "moisture") return "수분";
    if (metricName == "elasticity") return "탄력";
    if (metricName == "pigmentation") return "색소침착";
    if (metricName == "pore") return "모공";
    return metricName;
}

QString AnalysisResultDialog::getScoreDescription(int score)
{
    if (score >= 80) return "매우 좋음";
    if (score >= 60) return "좋음";
    if (score >= 40) return "보통";
    if (score >= 20) return "주의필요";
    return "관리필요";
}

QColor AnalysisResultDialog::getScoreColor(int score)
{
    if (score >= 80) return QColor("#27ae60"); // 초록
    if (score >= 60) return QColor("#2ecc71"); // 연한 초록
    if (score >= 40) return QColor("#f39c12"); // 주황
    if (score >= 20) return QColor("#e67e22"); // 진한 주황
    return QColor("#e74c3c"); // 빨강
}

void AnalysisResultDialog::displayComparisonData(const QJsonObject &currentData, const QJsonObject &previousData)
{
    QVBoxLayout *contentLayout = new QVBoxLayout(contentWidget);
    contentLayout->setSpacing(12);
    contentLayout->setContentsMargins(8, 8, 8, 8);
    
    // 통합 비교 막대그래프 생성
    QChartView *chartView = createUnifiedComparisonChart(currentData, previousData);
    contentLayout->addWidget(chartView);
    
    // 추천 화장품 버튼 추가 (현재 데이터 기준)
    QPushButton *recommendationButton = new QPushButton("추천 화장품 보기");
    recommendationButton->setMinimumHeight(40);
    recommendationButton->setStyleSheet(
        "QPushButton {"
        "  background-color: #27ae60;"
        "  border: none;"
        "  color: white;"
        "  padding: 8px 16px;"
        "  border-radius: 6px;"
        "  font-size: 14px;"
        "  font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "  background-color: #2ecc71;"
        "}"
        "QPushButton:pressed {"
        "  background-color: #229954;"
        "}"
    );
    
    connect(recommendationButton, &QPushButton::clicked, [this, currentData]() {
        showRecommendationsDialog(currentData);
        recommendationViewed = true; // 추천을 봤다고 표시
    });
    
    contentLayout->addWidget(recommendationButton);
    contentLayout->addStretch();
}

QWidget* AnalysisResultDialog::createComparisonRegionWidget(const QString &regionName, const QJsonObject &currentData, const QJsonObject &previousData)
{
    QGroupBox *groupBox = new QGroupBox(regionName);
    groupBox->setStyleSheet(
        "QGroupBox {"
        "  font-weight: bold;"
        "  font-size: 12px;"
        "  color: #2c3e50;"
        "  border: 2px solid #bdc3c7;"
        "  border-radius: 6px;"
        "  margin-top: 8px;"
        "  padding-top: 8px;"
        "}"
        "QGroupBox::title {"
        "  subcontrol-origin: margin;"
        "  left: 8px;"
        "  padding: 0 4px 0 4px;"
        "}"
    );
    
    QGridLayout *gridLayout = new QGridLayout(groupBox);
    gridLayout->setSpacing(8);
    gridLayout->setContentsMargins(10, 15, 10, 10);
    
    int row = 0;
    int col = 0;
    const int maxCols = 2; // 한 줄에 최대 2개 메트릭
    
    // 메트릭들을 그리드로 배치
    for (auto it = currentData.begin(); it != currentData.end(); ++it) {
        if (it.value().isDouble()) {
            int currentValue = static_cast<int>(it.value().toDouble());
            int previousValue = previousData.contains(it.key()) ? 
                static_cast<int>(previousData[it.key()].toDouble()) : -1;
                
            QWidget *metricWidget = createComparisonMetricWidget(it.key(), currentValue, previousValue);
            gridLayout->addWidget(metricWidget, row, col);
            
            col++;
            if (col >= maxCols) {
                col = 0;
                row++;
            }
        }
    }
    
    return groupBox;
}

QWidget* AnalysisResultDialog::createComparisonMetricWidget(const QString &metricName, int currentValue, int previousValue)
{
    QWidget *widget = new QWidget();
    widget->setMinimumHeight(140);
    widget->setMinimumWidth(120);
    widget->setStyleSheet("background-color: #f8f9fa; border-radius: 6px; padding: 8px;");
    
    QVBoxLayout *layout = new QVBoxLayout(widget);
    layout->setSpacing(6);
    layout->setContentsMargins(10, 8, 10, 8);
    
    // 메트릭 이름
    QLabel *nameLabel = new QLabel(getMetricDescription(metricName));
    QFont nameFont = nameLabel->font();
    nameFont.setPointSize(10);
    nameFont.setBold(true);
    nameLabel->setFont(nameFont);
    nameLabel->setStyleSheet("color: #2c3e50;");
    nameLabel->setAlignment(Qt::AlignCenter);
    layout->addWidget(nameLabel);
    
    // 수직 진행률 바들 컨테이너
    QWidget *barsContainer = new QWidget();
    barsContainer->setFixedHeight(60);
    barsContainer->setMinimumWidth(60);
    
    QHBoxLayout *barsLayout = new QHBoxLayout(barsContainer);
    barsLayout->setContentsMargins(0, 0, 0, 0);
    barsLayout->setSpacing(8);
    barsLayout->addStretch();
    
    // 현재 값 수직 진행률 바
    QProgressBar *currentBar = new QProgressBar();
    currentBar->setOrientation(Qt::Vertical);
    currentBar->setRange(0, 100);
    currentBar->setValue(currentValue);
    currentBar->setTextVisible(false);
    currentBar->setFixedWidth(18);
    currentBar->setFixedHeight(60);
    
    QColor currentColor = getScoreColor(currentValue);
    currentBar->setStyleSheet(QString(
        "QProgressBar {"
        "  border: 1px solid #bdc3c7;"
        "  border-radius: 3px;"
        "  background-color: #ecf0f1;"
        "}"
        "QProgressBar::chunk {"
        "  background-color: %1;"
        "  border-radius: 2px;"
        "}"
    ).arg(currentColor.name()));
    
    barsLayout->addWidget(currentBar);
    
    // 이전 값 수직 진행률 바 (있는 경우)
    if (previousValue >= 0) {
        QProgressBar *previousBar = new QProgressBar();
        previousBar->setOrientation(Qt::Vertical);
        previousBar->setRange(0, 100);
        previousBar->setValue(previousValue);
        previousBar->setTextVisible(false);
        previousBar->setFixedWidth(18);
        previousBar->setFixedHeight(60);
        
        QColor previousColor = getScoreColor(previousValue);
        previousColor.setAlpha(120); // 반투명 효과
        previousBar->setStyleSheet(QString(
            "QProgressBar {"
            "  border: 1px solid #bdc3c7;"
            "  border-radius: 3px;"
            "  background-color: #ecf0f1;"
            "}"
            "QProgressBar::chunk {"
            "  background-color: %1;"
            "  border-radius: 2px;"
            "}"
        ).arg(previousColor.name()));
        
        barsLayout->addWidget(previousBar);
    }
    
    barsLayout->addStretch();
    layout->addWidget(barsContainer);
    
    // 값과 비교 정보
    QHBoxLayout *infoLayout = new QHBoxLayout();
    
    // 현재 값
    QLabel *currentLabel = new QLabel(QString("%1").arg(currentValue));
    QFont currentFont = currentLabel->font();
    currentFont.setPointSize(12);
    currentFont.setBold(true);
    currentLabel->setFont(currentFont);
    currentLabel->setStyleSheet(QString("color: %1;").arg(currentColor.name()));
    
    infoLayout->addWidget(currentLabel);
    
    // 비교 정보 (이전 값이 있을 경우)
    if (previousValue >= 0) {
        int change = currentValue - previousValue;
        
        QLabel *changeLabel = new QLabel();
        if (change > 0) {
            changeLabel->setText(QString("▲%1").arg(change));
            changeLabel->setStyleSheet("color: #27ae60; font-weight: bold; font-size: 10px;");
        } else if (change < 0) {
            changeLabel->setText(QString("▼%1").arg(-change));
            changeLabel->setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 10px;");
        } else {
            changeLabel->setText("→ 0");
            changeLabel->setStyleSheet("color: #95a5a6; font-weight: bold; font-size: 10px;");
        }
        
        infoLayout->addWidget(changeLabel);
        
        // 이전 값
        QLabel *previousLabel = new QLabel(QString("(%1)").arg(previousValue));
        QFont prevFont = previousLabel->font();
        prevFont.setPointSize(9);
        previousLabel->setFont(prevFont);
        previousLabel->setStyleSheet("color: #7f8c8d;");
        
        infoLayout->addWidget(previousLabel);
    }
    
    infoLayout->addStretch();
    layout->addLayout(infoLayout);
    
    // 설명
    QLabel *descLabel = new QLabel(getScoreDescription(currentValue));
    QFont descFont = descLabel->font();
    descFont.setPointSize(9);
    descLabel->setFont(descFont);
    descLabel->setStyleSheet("color: #7f8c8d;");
    descLabel->setAlignment(Qt::AlignCenter);
    layout->addWidget(descLabel);
    
    return widget;
}

QString AnalysisResultDialog::getChangeDescription(int change)
{
    if (change > 10) return "크게 개선";
    if (change > 5) return "개선";
    if (change > 0) return "약간 개선";
    if (change == 0) return "변화없음";
    if (change > -5) return "약간 악화";
    if (change > -10) return "악화";
    return "크게 악화";
}

QColor AnalysisResultDialog::getChangeColor(int change)
{
    if (change > 5) return QColor("#27ae60");      // 초록 (개선)
    if (change > 0) return QColor("#2ecc71");      // 연한 초록 (약간 개선)  
    if (change == 0) return QColor("#95a5a6");     // 회색 (변화없음)
    if (change > -5) return QColor("#f39c12");     // 주황 (약간 악화)
    return QColor("#e74c3c");                      // 빨강 (악화)
}

QChartView* AnalysisResultDialog::createUnifiedBarChart(const QJsonObject &data)
{
    QChart *chart = new QChart();
    chart->setTitle("피부 분석 결과");
    chart->setAnimationOptions(QChart::SeriesAnimations);
    
    QBarSeries *series = new QBarSeries();
    QBarSet *barSet = new QBarSet("점수");
    
    QStringList categories;
    QList<QColor> barColors;
    
    // 각 영역별 데이터 수집
    QStringList regions = {"forehead", "l_check", "r_check", "chin", "lib"};
    QStringList regionNames = {"이마", "왼쪽 볼", "오른쪽 볼", "턱", "입술"};
    QStringList metrics = {"moisture", "elasticity", "pigmentation", "pore"};
    QStringList metricNames = {"수분", "탄력", "색소침착", "모공"};
    
    for (int i = 0; i < regions.size(); ++i) {
        if (data.contains(regions[i]) && data[regions[i]].isObject()) {
            QJsonObject regionData = data[regions[i]].toObject();
            
            for (int j = 0; j < metrics.size(); ++j) {
                if (regionData.contains(metrics[j])) {
                    int value = static_cast<int>(regionData[metrics[j]].toDouble());
                    QString label = regionNames[i] + " " + metricNames[j];
                    
                    categories.append(label);
                    barSet->append(value);
                    barColors.append(getScoreColor(value));
                }
            }
        }
    }
    
    series->append(barSet);
    chart->addSeries(series);
    
    // X축 설정
    QBarCategoryAxis *axisX = new QBarCategoryAxis();
    axisX->append(categories);
    
    // X축 라벨 폰트 크기 설정
    QFont axisFont = axisX->labelsFont();
    axisFont.setPointSize(8);
    axisX->setLabelsFont(axisFont);
    axisX->setLabelsAngle(-45); // 라벨을 45도 기울여서 겹치지 않게
    
    chart->addAxis(axisX, Qt::AlignBottom);
    series->attachAxis(axisX);
    
    // Y축 설정
    QValueAxis *axisY = new QValueAxis();
    axisY->setRange(0, 100);
    axisY->setTickCount(6);
    axisY->setLabelFormat("%d");
    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisY);
    
    // 차트 스타일 설정
    chart->legend()->setVisible(false);
    chart->setBackgroundBrush(QBrush(QColor(248, 249, 250)));
    
    // ChartView 생성
    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);
    
    // 화면 크기에 맞게 차트 크기 설정
    QScreen *screen = QApplication::primaryScreen();
    if (screen) {
        QRect screenGeometry = screen->availableGeometry();
        chartView->setMinimumHeight(screenGeometry.height() * 0.7);
        chartView->setMinimumWidth(screenGeometry.width() * 0.9);
    } else {
        chartView->setMinimumHeight(400);
        chartView->setMinimumWidth(600);
    }
    
    return chartView;
}

QWidget* AnalysisResultDialog::createRecommendationsWidget(const QJsonObject &data)
{
    // recommendations 배열이 있는지 확인
    if (!data.contains("recommendations") || !data["recommendations"].isArray()) {
        return nullptr;
    }
    
    QJsonArray recommendations = data["recommendations"].toArray();
    if (recommendations.isEmpty()) {
        return nullptr;
    }
    
    // 추천 문구 위젯 생성
    QGroupBox *groupBox = new QGroupBox("제품 추천");
    groupBox->setStyleSheet(
        "QGroupBox {"
        "  font-weight: bold;"
        "  font-size: 14px;"
        "  color: #2c3e50;"
        "  border: 2px solid #3498db;"
        "  border-radius: 8px;"
        "  margin-top: 10px;"
        "  padding-top: 12px;"
        "  background-color: #f8f9fa;"
        "}"
        "QGroupBox::title {"
        "  subcontrol-origin: margin;"
        "  left: 12px;"
        "  padding: 0 8px 0 8px;"
        "  color: #3498db;"
        "}"
    );
    
    QVBoxLayout *layout = new QVBoxLayout(groupBox);
    layout->setSpacing(10);
    layout->setContentsMargins(15, 20, 15, 15);
    
    // 안내 메시지
    QLabel *infoLabel = new QLabel("분석 결과에 따른 추천 제품:");
    QFont infoFont = infoLabel->font();
    infoFont.setPointSize(12);
    infoFont.setBold(false);
    infoLabel->setFont(infoFont);
    infoLabel->setStyleSheet("color: #34495e; margin-bottom: 8px;");
    layout->addWidget(infoLabel);
    
    // 추천 목록을 그리드로 배치
    QGridLayout *gridLayout = new QGridLayout();
    gridLayout->setSpacing(10);
    gridLayout->setContentsMargins(0, 0, 0, 0);
    
    int row = 0;
    int col = 0;
    const int maxCols = 3; // 한 줄에 최대 3개
    
    for (int i = 0; i < recommendations.size(); ++i) {
        QString recommendation = recommendations[i].toString();
        
        QLabel *recLabel = new QLabel(recommendation);
        QFont recFont = recLabel->font();
        recFont.setPointSize(11);
        recFont.setBold(true);
        recLabel->setFont(recFont);
        recLabel->setAlignment(Qt::AlignCenter);
        recLabel->setStyleSheet(
            "QLabel {"
            "  background-color: #3498db;"
            "  color: white;"
            "  padding: 8px 12px;"
            "  border-radius: 6px;"
            "  margin: 2px;"
            "}"
        );
        recLabel->setMinimumHeight(35);
        recLabel->setMinimumWidth(120);
        
        gridLayout->addWidget(recLabel, row, col);
        
        col++;
        if (col >= maxCols) {
            col = 0;
            row++;
        }
    }
    
    layout->addLayout(gridLayout);
    
    return groupBox;
}

void AnalysisResultDialog::showRecommendationsDialog(const QJsonObject &data)
{
    // recommendations 배열이 있는지 확인
    if (!data.contains("recommendations") || !data["recommendations"].isArray()) {
        QMessageBox::information(this, "알림", "추천 데이터가 없습니다.");
        return;
    }
    
    QJsonArray recommendations = data["recommendations"].toArray();
    if (recommendations.isEmpty()) {
        QMessageBox::information(this, "알림", "추천할 제품이 없습니다.");
        return;
    }
    
    // 추천 다이얼로그 생성
    QDialog *recommendDialog = new QDialog(this);
    recommendDialog->setWindowTitle("추천 화장품");
    recommendDialog->setModal(true);
    recommendDialog->resize(400, 300);
    
    QVBoxLayout *layout = new QVBoxLayout(recommendDialog);
    layout->setSpacing(15);
    layout->setContentsMargins(20, 20, 20, 20);
    
    // 제목
    QLabel *titleLabel = new QLabel("분석 결과에 따른 추천 제품");
    QFont titleFont = titleLabel->font();
    titleFont.setPointSize(16);
    titleFont.setBold(true);
    titleLabel->setFont(titleFont);
    titleLabel->setAlignment(Qt::AlignCenter);
    titleLabel->setStyleSheet("color: #2c3e50; margin-bottom: 15px;");
    layout->addWidget(titleLabel);
    
    // 추천 목록을 그리드로 배치
    QGridLayout *gridLayout = new QGridLayout();
    gridLayout->setSpacing(10);
    gridLayout->setContentsMargins(0, 0, 0, 0);
    
    int row = 0;
    int col = 0;
    const int maxCols = 2; // 한 줄에 최대 2개
    
    for (int i = 0; i < recommendations.size(); ++i) {
        QString recommendation = recommendations[i].toString();
        
        QLabel *recLabel = new QLabel(recommendation);
        QFont recFont = recLabel->font();
        recFont.setPointSize(12);
        recFont.setBold(true);
        recLabel->setFont(recFont);
        recLabel->setAlignment(Qt::AlignCenter);
        recLabel->setStyleSheet(
            "QLabel {"
            "  background-color: #3498db;"
            "  color: white;"
            "  padding: 12px 16px;"
            "  border-radius: 8px;"
            "  margin: 4px;"
            "}"
        );
        recLabel->setMinimumHeight(50);
        recLabel->setMinimumWidth(150);
        
        gridLayout->addWidget(recLabel, row, col);
        
        col++;
        if (col >= maxCols) {
            col = 0;
            row++;
        }
    }
    
    layout->addLayout(gridLayout);
    layout->addStretch();
    
    // 닫기 버튼
    QPushButton *closeButton = new QPushButton("닫기");
    closeButton->setMinimumHeight(35);
    closeButton->setStyleSheet(
        "QPushButton {"
        "  background-color: #95a5a6;"
        "  border: none;"
        "  color: white;"
        "  padding: 8px 16px;"
        "  border-radius: 6px;"
        "  font-size: 12px;"
        "}"
        "QPushButton:hover {"
        "  background-color: #7f8c8d;"
        "}"
    );
    
    connect(closeButton, &QPushButton::clicked, [recommendDialog, this]() {
        recommendDialog->accept();
        // 추천 다이얼로그 닫기 후 분석 결과 다이얼로그도 닫고 이름 입력으로 이동
        this->accept();
        showNameInputDialog();
    });
    layout->addWidget(closeButton);
    
    recommendDialog->exec();
}

void AnalysisResultDialog::showNameInputDialog()
{
    // MainWindow에서 다이얼로그 종료 후 이름 입력 화면으로 돌아가는 처리를 담당
    qDebug() << "Dialog closed, MainWindow will handle returning to initial view";
}

QChartView* AnalysisResultDialog::createUnifiedComparisonChart(const QJsonObject &currentData, const QJsonObject &previousData)
{
    QChart *chart = new QChart();
    chart->setTitle("피부 분석 비교 결과");
    chart->setAnimationOptions(QChart::SeriesAnimations);
    
    QBarSeries *series = new QBarSeries();
    QBarSet *currentBarSet = new QBarSet("현재");
    QBarSet *previousBarSet = new QBarSet("이전");
    
    QStringList categories;
    
    // 각 영역별 데이터 수집
    QStringList regions = {"forehead", "l_check", "r_check", "chin", "lib"};
    QStringList regionNames = {"이마", "왼쪽 볼", "오른쪽 볼", "턱", "입술"};
    QStringList metrics = {"moisture", "elasticity", "pigmentation", "pore"};
    QStringList metricNames = {"수분", "탄력", "색소침착", "모공"};
    
    for (int i = 0; i < regions.size(); ++i) {
        if (currentData.contains(regions[i]) && currentData[regions[i]].isObject()) {
            QJsonObject currentRegionData = currentData[regions[i]].toObject();
            QJsonObject previousRegionData = previousData.contains(regions[i]) ? 
                previousData[regions[i]].toObject() : QJsonObject();
            
            for (int j = 0; j < metrics.size(); ++j) {
                if (currentRegionData.contains(metrics[j])) {
                    int currentValue = static_cast<int>(currentRegionData[metrics[j]].toDouble());
                    int previousValue = previousRegionData.contains(metrics[j]) ? 
                        static_cast<int>(previousRegionData[metrics[j]].toDouble()) : 0;
                    
                    QString label = regionNames[i] + "\n" + metricNames[j];
                    
                    categories.append(label);
                    currentBarSet->append(currentValue);
                    previousBarSet->append(previousValue);
                }
            }
        }
    }
    
    // 막대 색상 설정
    currentBarSet->setColor(QColor("#3498db"));
    previousBarSet->setColor(QColor("#95a5a6"));
    
    series->append(currentBarSet);
    series->append(previousBarSet);
    chart->addSeries(series);
    
    // X축 설정
    QBarCategoryAxis *axisX = new QBarCategoryAxis();
    axisX->append(categories);
    
    // X축 라벨 폰트 크기 설정
    QFont axisFont = axisX->labelsFont();
    axisFont.setPointSize(8);
    axisX->setLabelsFont(axisFont);
    axisX->setLabelsAngle(-45); // 라벨을 45도 기울여서 겹치지 않게
    
    chart->addAxis(axisX, Qt::AlignBottom);
    series->attachAxis(axisX);
    
    // Y축 설정
    QValueAxis *axisY = new QValueAxis();
    axisY->setRange(0, 100);
    axisY->setTickCount(6);
    axisY->setLabelFormat("%d");
    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisY);
    
    // 차트 스타일 설정
    chart->legend()->setVisible(true);
    chart->legend()->setAlignment(Qt::AlignBottom);
    chart->setBackgroundBrush(QBrush(QColor(248, 249, 250)));
    
    // ChartView 생성
    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);
    
    // 화면 크기에 맞게 차트 크기 설정
    QScreen *screen = QApplication::primaryScreen();
    if (screen) {
        QRect screenGeometry = screen->availableGeometry();
        chartView->setMinimumHeight(screenGeometry.height() * 0.7);
        chartView->setMinimumWidth(screenGeometry.width() * 0.9);
    } else {
        chartView->setMinimumHeight(450);
        chartView->setMinimumWidth(700);
    }
    
    return chartView;
}
