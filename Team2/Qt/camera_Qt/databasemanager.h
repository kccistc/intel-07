#ifndef DATABASEMANAGER_H
#define DATABASEMANAGER_H

#include <QtSql/QSqlDatabase>
#include <QtSql/QSqlQuery>
#include <QtSql/QSqlError>
#include <QDateTime>
#include <QString>
#include <QVariant>
#include <QJsonObject>
#include <QJsonDocument>

struct AnalysisRecord {
    int id;
    QString userName;
    QDateTime captureTime;
    QJsonObject analysisData;
};

class DatabaseManager
{
public:
    static DatabaseManager& instance();
    
    bool initializeDatabase();
    void closeDatabase();
    
    // Analysis Record Management
    bool saveAnalysisResult(const QString &userName, const QJsonObject &analysisData);
    QList<AnalysisRecord> getUserHistory(const QString &userName) const;
    QList<AnalysisRecord> getAllHistory() const;
    bool deleteAnalysisRecord(int recordId);
    
    // Statistics
    int getTotalUsers() const;
    int getTotalRecords() const;
    QDateTime getLastActivity() const;
    
private:
    DatabaseManager();
    ~DatabaseManager();
    
    // Disable copy constructor and assignment operator
    DatabaseManager(const DatabaseManager&) = delete;
    DatabaseManager& operator=(const DatabaseManager&) = delete;
    
    QSqlDatabase database;
    
    bool createTables();
    bool executeQuery(const QString &query, const QVariantList &params = QVariantList());
    QSqlQuery prepareQuery(const QString &query, const QVariantList &params = QVariantList()) const;
};

#endif // DATABASEMANAGER_H