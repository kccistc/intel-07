/********************************************************************************
** Form generated from reading UI file 'tab1.ui'
**
** Created by: Qt User Interface Compiler version 6.8.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TAB1_H
#define UI_TAB1_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_tab1
{
public:
    QHBoxLayout *mainHorizontalLayout;
    QGroupBox *imageGroupBox;
    QVBoxLayout *imageVerticalLayout;
    QLabel *pLabelImage;
    QPushButton *pPBclearImage;
    QSpacerItem *imageVerticalSpacer;
    QVBoxLayout *rightVerticalLayout;
    QHBoxLayout *horizontalLayout;
    QLabel *label;
    QPushButton *pPBrecvDataClear;
    QPushButton *pPBserverConnect;
    QTextEdit *pTErecvData;
    QHBoxLayout *horizontalLayout_2;
    QLineEdit *pLErecvId;
    QLineEdit *pLEsendData;
    QPushButton *pPBSend;

    void setupUi(QWidget *tab1)
    {
        if (tab1->objectName().isEmpty())
            tab1->setObjectName("tab1");
        tab1->resize(800, 600);
        mainHorizontalLayout = new QHBoxLayout(tab1);
        mainHorizontalLayout->setObjectName("mainHorizontalLayout");
        imageGroupBox = new QGroupBox(tab1);
        imageGroupBox->setObjectName("imageGroupBox");
        imageGroupBox->setMinimumSize(QSize(300, 0));
        imageGroupBox->setMaximumSize(QSize(350, 16777215));
        imageVerticalLayout = new QVBoxLayout(imageGroupBox);
        imageVerticalLayout->setObjectName("imageVerticalLayout");
        pLabelImage = new QLabel(imageGroupBox);
        pLabelImage->setObjectName("pLabelImage");
        pLabelImage->setAlignment(Qt::AlignCenter);
        pLabelImage->setMinimumSize(QSize(280, 280));
        pLabelImage->setScaledContents(true);

        imageVerticalLayout->addWidget(pLabelImage);

        pPBclearImage = new QPushButton(imageGroupBox);
        pPBclearImage->setObjectName("pPBclearImage");

        imageVerticalLayout->addWidget(pPBclearImage);

        imageVerticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Policy::Minimum, QSizePolicy::Policy::Expanding);

        imageVerticalLayout->addItem(imageVerticalSpacer);


        mainHorizontalLayout->addWidget(imageGroupBox);

        rightVerticalLayout = new QVBoxLayout();
        rightVerticalLayout->setObjectName("rightVerticalLayout");
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName("horizontalLayout");
        label = new QLabel(tab1);
        label->setObjectName("label");

        horizontalLayout->addWidget(label);

        pPBrecvDataClear = new QPushButton(tab1);
        pPBrecvDataClear->setObjectName("pPBrecvDataClear");

        horizontalLayout->addWidget(pPBrecvDataClear);

        pPBserverConnect = new QPushButton(tab1);
        pPBserverConnect->setObjectName("pPBserverConnect");
        pPBserverConnect->setCheckable(true);

        horizontalLayout->addWidget(pPBserverConnect);

        horizontalLayout->setStretch(0, 6);
        horizontalLayout->setStretch(1, 2);
        horizontalLayout->setStretch(2, 2);

        rightVerticalLayout->addLayout(horizontalLayout);

        pTErecvData = new QTextEdit(tab1);
        pTErecvData->setObjectName("pTErecvData");

        rightVerticalLayout->addWidget(pTErecvData);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName("horizontalLayout_2");
        pLErecvId = new QLineEdit(tab1);
        pLErecvId->setObjectName("pLErecvId");

        horizontalLayout_2->addWidget(pLErecvId);

        pLEsendData = new QLineEdit(tab1);
        pLEsendData->setObjectName("pLEsendData");

        horizontalLayout_2->addWidget(pLEsendData);

        pPBSend = new QPushButton(tab1);
        pPBSend->setObjectName("pPBSend");

        horizontalLayout_2->addWidget(pPBSend);

        horizontalLayout_2->setStretch(0, 3);
        horizontalLayout_2->setStretch(1, 6);
        horizontalLayout_2->setStretch(2, 1);

        rightVerticalLayout->addLayout(horizontalLayout_2);


        mainHorizontalLayout->addLayout(rightVerticalLayout);


        retranslateUi(tab1);

        QMetaObject::connectSlotsByName(tab1);
    } // setupUi

    void retranslateUi(QWidget *tab1)
    {
        tab1->setWindowTitle(QCoreApplication::translate("tab1", "Form", nullptr));
        imageGroupBox->setTitle(QCoreApplication::translate("tab1", "\354\210\230\354\213\240 \354\235\264\353\257\270\354\247\200", nullptr));
        pLabelImage->setText(QCoreApplication::translate("tab1", "\354\235\264\353\257\270\354\247\200 \354\227\206\354\235\214", nullptr));
        pLabelImage->setStyleSheet(QCoreApplication::translate("tab1", "QLabel {\n"
"    border: 2px solid gray;\n"
"    border-radius: 5px;\n"
"    background-color: #f0f0f0;\n"
"    color: #666;\n"
"    font-size: 12px;\n"
"}", nullptr));
        pPBclearImage->setText(QCoreApplication::translate("tab1", "\354\235\264\353\257\270\354\247\200 \354\247\200\354\232\260\352\270\260", nullptr));
        label->setText(QCoreApplication::translate("tab1", "\354\210\230\354\213\240 \353\215\260\354\235\264\355\204\260", nullptr));
        pPBrecvDataClear->setText(QCoreApplication::translate("tab1", "\354\210\230\354\213\240 \354\202\255\354\240\234", nullptr));
        pPBserverConnect->setText(QCoreApplication::translate("tab1", "\354\204\234\353\262\204 \354\227\260\352\262\260", nullptr));
        pLErecvId->setPlaceholderText(QCoreApplication::translate("tab1", "\354\210\230\354\213\240\354\236\220 ID (\352\263\265\353\260\261\354\213\234 \354\240\204\354\262\264)", nullptr));
        pLEsendData->setPlaceholderText(QCoreApplication::translate("tab1", "\354\240\204\354\206\241\355\225\240 \353\251\224\354\213\234\354\247\200", nullptr));
        pPBSend->setText(QCoreApplication::translate("tab1", "\354\206\241\354\213\240", nullptr));
    } // retranslateUi

};

namespace Ui {
    class tab1: public Ui_tab1 {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TAB1_H
