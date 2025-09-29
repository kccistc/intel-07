#ifndef ANALYSISRESULTDIALOG_H
#define ANALYSISRESULTDIALOG_H

#include <QDialog>
#include <QLabel>
#include <QProgressBar>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QPushButton>
#include <QJsonObject>
#include <QJsonArray>
#include <QScrollArea>
#include <QScreen>
#include <QApplication>
#include <QMessageBox>
#include <QtCharts/QChartView>
#include <QtCharts/QBarSeries>
#include <QtCharts/QBarSet>
#include <QtCharts/QValueAxis>
#include <QtCharts/QBarCategoryAxis>

// Forward declaration
class NameInputDialog;

class AnalysisResultDialog : public QDialog
{
    Q_OBJECT

public:
    explicit AnalysisResultDialog(QJsonObject analysisData, QString userName = QString(), QWidget *parent = nullptr);
    explicit AnalysisResultDialog(QJsonObject currentData, QJsonObject previousData, QString userName, QWidget *parent = nullptr);

private:
    void setupUI();
    void displayAnalysisData(const QJsonObject &data);
    void displayComparisonData(const QJsonObject &currentData, const QJsonObject &previousData);
    QWidget* createRegionWidget(const QString &regionName, const QJsonObject &regionData);
    QWidget* createComparisonRegionWidget(const QString &regionName, const QJsonObject &currentData, const QJsonObject &previousData);
    QWidget* createMetricWidget(const QString &metricName, int value);
    QWidget* createComparisonMetricWidget(const QString &metricName, int currentValue, int previousValue);
    QtCharts::QChartView* createUnifiedBarChart(const QJsonObject &data);
    QtCharts::QChartView* createUnifiedComparisonChart(const QJsonObject &currentData, const QJsonObject &previousData);
    QWidget* createRecommendationsWidget(const QJsonObject &data);
    void showRecommendationsDialog(const QJsonObject &data);
    void showNameInputDialog();
    QString getMetricDescription(const QString &metricName);
    QString getScoreDescription(int score);
    QColor getScoreColor(int score);
    QString getChangeDescription(int change);
    QColor getChangeColor(int change);
    
    QJsonObject analysisData;
    QJsonObject previousData;
    QString userName;
    bool isComparison;
    bool recommendationViewed;
    QScrollArea *scrollArea;
    QWidget *contentWidget;
    QVBoxLayout *mainLayout;
    QPushButton *closeButton;
};

#endif // ANALYSISRESULTDIALOG_H