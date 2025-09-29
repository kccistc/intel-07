/********************************************************************************
** Form generated from reading UI file 'widget.ui'
**
** Created by: Qt User Interface Compiler version 6.8.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WIDGET_H
#define UI_WIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Widget
{
public:
    QVBoxLayout *verticalLayout;
    QTabWidget *tabWidget;
    QWidget *serverTab;
    QVBoxLayout *serverTabLayout;
    QWidget *tab_2;

    void setupUi(QWidget *Widget)
    {
        if (Widget->objectName().isEmpty())
            Widget->setObjectName("Widget");
        Widget->resize(705, 510);
        verticalLayout = new QVBoxLayout(Widget);
        verticalLayout->setObjectName("verticalLayout");
        tabWidget = new QTabWidget(Widget);
        tabWidget->setObjectName("tabWidget");
        serverTab = new QWidget();
        serverTab->setObjectName("serverTab");
        serverTabLayout = new QVBoxLayout(serverTab);
        serverTabLayout->setSpacing(0);
        serverTabLayout->setObjectName("serverTabLayout");
        serverTabLayout->setContentsMargins(0, 0, 0, 0);
        tabWidget->addTab(serverTab, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName("tab_2");
        tabWidget->addTab(tab_2, QString());

        verticalLayout->addWidget(tabWidget);


        retranslateUi(Widget);

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(Widget);
    } // setupUi

    void retranslateUi(QWidget *Widget)
    {
        Widget->setWindowTitle(QCoreApplication::translate("Widget", "\354\204\234\353\262\204 \355\206\265\354\213\240 \355\224\204\353\241\234\352\267\270\353\236\250", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(serverTab), QCoreApplication::translate("Widget", "\354\204\234\353\262\204 \355\206\265\354\213\240", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QCoreApplication::translate("Widget", "Tab 2", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Widget: public Ui_Widget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_H
