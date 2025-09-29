#ifndef NAMEINPUTDIALOG_H
#define NAMEINPUTDIALOG_H

#include <QDialog>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QMessageBox>

class NameInputDialog : public QDialog
{
    Q_OBJECT

public:
    explicit NameInputDialog(QWidget *parent = nullptr);
    QString getUserName() const;

private slots:
    void validateAndAccept();

private:
    QLineEdit *nameLineEdit;
    QPushButton *okButton;
    QPushButton *cancelButton;
    QLabel *instructionLabel;
    
    void setupUI();
    bool isValidName(const QString &name) const;
};

#endif // NAMEINPUTDIALOG_H