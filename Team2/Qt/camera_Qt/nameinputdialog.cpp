#include "nameinputdialog.h"
#include <QRegExp>

NameInputDialog::NameInputDialog(QWidget *parent)
    : QDialog(parent)
{
    setupUI();
    setWindowTitle("사용자 정보 입력");
    setModal(true);
    setFixedSize(350, 150);
    
    // Enter 키로 확인 가능하도록 설정
    okButton->setDefault(true);
    
    // 연결 설정
    connect(okButton, &QPushButton::clicked, this, &NameInputDialog::validateAndAccept);
    connect(cancelButton, &QPushButton::clicked, this, &QDialog::reject);
    connect(nameLineEdit, &QLineEdit::returnPressed, this, &NameInputDialog::validateAndAccept);
}

void NameInputDialog::setupUI()
{
    // 메인 레이아웃
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(15);
    mainLayout->setContentsMargins(20, 20, 20, 20);
    
    // 안내 라벨
    instructionLabel = new QLabel("피부 분석을 위해 이름을 입력해주세요:");
    instructionLabel->setWordWrap(true);
    instructionLabel->setStyleSheet("font-size: 12px; color: #333;");
    mainLayout->addWidget(instructionLabel);
    
    // 이름 입력 필드
    nameLineEdit = new QLineEdit();
    nameLineEdit->setPlaceholderText("이름을 입력하세요 (2-20자)");
    nameLineEdit->setMaxLength(20);
    nameLineEdit->setStyleSheet(
        "QLineEdit {"
        "    padding: 8px;"
        "    border: 2px solid #ccc;"
        "    border-radius: 5px;"
        "    font-size: 12px;"
        "}"
        "QLineEdit:focus {"
        "    border-color: #4CAF50;"
        "}"
    );
    mainLayout->addWidget(nameLineEdit);
    
    // 버튼 레이아웃
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    buttonLayout->setSpacing(10);
    
    cancelButton = new QPushButton("취소");
    cancelButton->setStyleSheet(
        "QPushButton {"
        "    padding: 8px 20px;"
        "    background-color: #f44336;"
        "    color: white;"
        "    border: none;"
        "    border-radius: 5px;"
        "    font-size: 11px;"
        "}"
        "QPushButton:hover {"
        "    background-color: #d32f2f;"
        "}"
    );
    
    okButton = new QPushButton("확인");
    okButton->setStyleSheet(
        "QPushButton {"
        "    padding: 8px 20px;"
        "    background-color: #4CAF50;"
        "    color: white;"
        "    border: none;"
        "    border-radius: 5px;"
        "    font-size: 11px;"
        "}"
        "QPushButton:hover {"
        "    background-color: #45a049;"
        "}"
        "QPushButton:disabled {"
        "    background-color: #cccccc;"
        "}"
    );
    
    buttonLayout->addStretch();
    buttonLayout->addWidget(cancelButton);
    buttonLayout->addWidget(okButton);
    
    mainLayout->addLayout(buttonLayout);
    
    // 포커스를 이름 입력 필드에 설정
    nameLineEdit->setFocus();
}

QString NameInputDialog::getUserName() const
{
    return nameLineEdit->text().trimmed();
}

bool NameInputDialog::isValidName(const QString &name) const
{
    // 이름 유효성 검사
    if (name.length() < 2) {
        return false;
    }
    
    if (name.length() > 20) {
        return false;
    }
    
    // 특수문자 제한 (한글, 영문, 숫자, 공백만 허용)
    QRegExp validChars("[가-힣a-zA-Z0-9\\s]+");
    return validChars.exactMatch(name);
}

void NameInputDialog::validateAndAccept()
{
    QString name = getUserName();
    
    if (name.isEmpty()) {
        QMessageBox::warning(this, "입력 오류", "이름을 입력해주세요.");
        nameLineEdit->setFocus();
        return;
    }
    
    if (!isValidName(name)) {
        QMessageBox::warning(this, "입력 오류", 
            "이름은 2-20자의 한글, 영문, 숫자만 입력 가능합니다.");
        nameLineEdit->setFocus();
        nameLineEdit->selectAll();
        return;
    }
    
    accept();
}