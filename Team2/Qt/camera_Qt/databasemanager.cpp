#include "databasemanager.h"
#include <QDebug>
#include <QDir>
#include <QStandardPaths>
#include <QJsonDocument>
#include <QApplication>

DatabaseManager::DatabaseManager()
{
    // SQLite 드라이버 확인
    if (!QSqlDatabase::isDriverAvailable("QSQLITE")) {
        qCritical() << "SQLite driver not available!";
        return;
    }
}

DatabaseManager::~DatabaseManager()
{
    closeDatabase();
}

DatabaseManager& DatabaseManager::instance()
{
    static DatabaseManager instance;
    return instance;
}

bool DatabaseManager::initializeDatabase()
{
    // 데이터베이스 파일 경로 설정
    QString dataPath = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir().mkpath(dataPath);
    QString dbPath = dataPath + "/skin_analysis.db";
    
    qDebug() << "Database path:" << dbPath;
    
    // 데이터베이스 연결 설정
    database = QSqlDatabase::addDatabase("QSQLITE");
    database.setDatabaseName(dbPath);
    
    if (!database.open()) {
        qCritical() << "Failed to open database:" << database.lastError().text();
        return false;
    }
    
    qDebug() << "Database opened successfully";
    
    // 테이블 생성
    if (!createTables()) {
        qCritical() << "Failed to create database tables";
        return false;
    }
    
    return true;
}

void DatabaseManager::closeDatabase()
{
    if (database.isOpen()) {
        database.close();
        qDebug() << "Database closed";
    }
}

bool DatabaseManager::createTables()
{
    QStringList createQueries;
    
    // 분석 결과 테이블
    createQueries << R"(
        CREATE TABLE IF NOT EXISTS analysis_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT NOT NULL,
            capture_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            analysis_data TEXT NOT NULL
        )
    )";
    
    // 인덱스 생성
    createQueries << "CREATE INDEX IF NOT EXISTS idx_records_user ON analysis_records(user_name)";
    createQueries << "CREATE INDEX IF NOT EXISTS idx_records_time ON analysis_records(capture_time)";
    
    for (const QString &query : createQueries) {
        if (!executeQuery(query)) {
            return false;
        }
    }
    
    qDebug() << "Database tables created successfully";
    return true;
}

bool DatabaseManager::executeQuery(const QString &query, const QVariantList &params)
{
    QSqlQuery sqlQuery = prepareQuery(query, params);
    
    if (!sqlQuery.exec()) {
        qCritical() << "Query execution failed:" << sqlQuery.lastError().text();
        qCritical() << "Query:" << query;
        return false;
    }
    
    return true;
}

QSqlQuery DatabaseManager::prepareQuery(const QString &query, const QVariantList &params) const
{
    QSqlQuery sqlQuery(database);
    sqlQuery.prepare(query);
    
    for (int i = 0; i < params.size(); ++i) {
        sqlQuery.bindValue(i, params[i]);
    }
    
    return sqlQuery;
}

bool DatabaseManager::saveAnalysisResult(const QString &userName, const QJsonObject &analysisData)
{
    QString query = R"(
        INSERT INTO analysis_records (user_name, capture_time, analysis_data)
        VALUES (?, CURRENT_TIMESTAMP, ?)
    )";
    
    QJsonDocument doc(analysisData);
    QString jsonString = doc.toJson(QJsonDocument::Compact);
    
    QSqlQuery sqlQuery = prepareQuery(query, {userName, jsonString});
    
    if (sqlQuery.exec()) {
        int recordId = sqlQuery.lastInsertId().toInt();
        qDebug() << "Analysis result saved for:" << userName << "ID:" << recordId;
        return true;
    } else {
        qCritical() << "Failed to save analysis result:" << sqlQuery.lastError().text();
        return false;
    }
}

QList<AnalysisRecord> DatabaseManager::getUserHistory(const QString &userName) const
{
    QList<AnalysisRecord> records;
    QString query = "SELECT id, user_name, capture_time, analysis_data FROM analysis_records WHERE user_name = ? ORDER BY capture_time DESC";
    
    QSqlQuery sqlQuery = prepareQuery(query, {userName});
    
    if (sqlQuery.exec()) {
        while (sqlQuery.next()) {
            AnalysisRecord record;
            record.id = sqlQuery.value(0).toInt();
            record.userName = sqlQuery.value(1).toString();
            record.captureTime = sqlQuery.value(2).toDateTime();
            
            QString jsonString = sqlQuery.value(3).toString();
            QJsonDocument doc = QJsonDocument::fromJson(jsonString.toUtf8());
            record.analysisData = doc.object();
            
            records.append(record);
        }
    }
    
    return records;
}

QList<AnalysisRecord> DatabaseManager::getAllHistory() const
{
    QList<AnalysisRecord> records;
    QString query = "SELECT id, user_name, capture_time, analysis_data FROM analysis_records ORDER BY capture_time DESC";
    
    QSqlQuery sqlQuery = prepareQuery(query);
    
    if (sqlQuery.exec()) {
        while (sqlQuery.next()) {
            AnalysisRecord record;
            record.id = sqlQuery.value(0).toInt();
            record.userName = sqlQuery.value(1).toString();
            record.captureTime = sqlQuery.value(2).toDateTime();
            
            QString jsonString = sqlQuery.value(3).toString();
            QJsonDocument doc = QJsonDocument::fromJson(jsonString.toUtf8());
            record.analysisData = doc.object();
            
            records.append(record);
        }
    }
    
    return records;
}

bool DatabaseManager::deleteAnalysisRecord(int recordId)
{
    QString query = "DELETE FROM analysis_records WHERE id = ?";
    return executeQuery(query, {recordId});
}

int DatabaseManager::getTotalUsers() const
{
    QString query = "SELECT COUNT(DISTINCT user_name) FROM analysis_records";
    QSqlQuery sqlQuery = prepareQuery(query);
    
    if (sqlQuery.exec() && sqlQuery.next()) {
        return sqlQuery.value(0).toInt();
    }
    
    return 0;
}

int DatabaseManager::getTotalRecords() const
{
    QString query = "SELECT COUNT(*) FROM analysis_records";
    QSqlQuery sqlQuery = prepareQuery(query);
    
    if (sqlQuery.exec() && sqlQuery.next()) {
        return sqlQuery.value(0).toInt();
    }
    
    return 0;
}

QDateTime DatabaseManager::getLastActivity() const
{
    QString query = "SELECT MAX(capture_time) FROM analysis_records";
    QSqlQuery sqlQuery = prepareQuery(query);
    
    if (sqlQuery.exec() && sqlQuery.next()) {
        return sqlQuery.value(0).toDateTime();
    }
    
    return QDateTime();
}