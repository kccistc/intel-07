import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtGraphicalEffects 1.15

Item {
    id: avatarControl

    FontLoader {
        id: neodgm
        source: "resource/fonts/neodgm.ttf"
    }

    RowLayout {
        anchors.centerIn: parent
        spacing: 50 // 두 버튼 그룹 사이의 간격

        // 왼쪽 컨트롤 그룹
        ColumnLayout {
            spacing: 200 // 위아래 버튼 사이의 간격

            Button {
                text: "⬅️"
                font.family: neodgm.name
                font.pixelSize: 200
                width: 320
                height: 200
                Layout.alignment: Qt.AlignLeft // 큰 버튼 위에 수평으로 중앙 정렬

                background: Rectangle {
                    color: "#DD000000" // 파란색 배경
                    radius: 8
                }
                contentItem: Text {
                    text: parent.text
                    color: "white"
                    font: parent.font
                    anchors.centerIn: parent
                }
                onClicked: {
                    controlBridge.onAvatarPrevious()
                }
            }

            Button {
                text: "변환하기"
                font.family: neodgm.name
                font.pixelSize: 100
                width: 200
                height: 70
                background: Rectangle {
                    color: "#DD000000" // 파란색 배경
                    radius: 10
                }
                contentItem: Text {
                    text: parent.text
                    color: "white"
                    font: parent.font
                    anchors.centerIn: parent
                }
                onClicked: {
                    console.log("변환하기 버튼 클릭")
                    controlBridge.startAvatarConversion()
                }
            }
        }

        // 오른쪽 컨트롤 그룹
        ColumnLayout {
            spacing: 200 // 위아래 버튼 사이의 간격

            Button {
                text: "➡️"
                font.family: neodgm.name
                font.pixelSize: 200
                width: 320
                height: 200
                Layout.alignment: Qt.AlignRight // 큰 버튼 위에 수평으로 중앙 정렬

                background: Rectangle {
                    color: "#DD000000" // 파란색 배경
                    radius: 8
                }
                contentItem: Text {
                    text: parent.text
                    color: "white"
                    font: parent.font
                    anchors.centerIn: parent
                }
                onClicked: {
                    controlBridge.onAvatarNext()
                }
            }

            Button {
                text: "메인으로"
                font.family: neodgm.name
                font.pixelSize: 100
                width: 200
                height: 70
                background: Rectangle {
                    color: "#DD000000" // 파란색 배경
                    radius: 8
                }
                contentItem: Text {
                    text: parent.text
                    color: "white"
                    font: parent.font
                    anchors.centerIn: parent
                }
                onClicked: {
                    console.log("메인으로 버튼 클릭")
                    controlBridge.goToMainMenu()
                }
            }
        }
    }
}
