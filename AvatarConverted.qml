import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtGraphicalEffects 1.15

Item {
    FontLoader {
      id: neodgm
      source: "resource/fonts/neodgm.ttf"
    }

    Rectangle {
        anchors.fill: parent
        color: "transparent"
    }

    ColumnLayout {
        anchors.centerIn: parent
        spacing: 20

        Text {
            text: "변환 완료!"
            font.pixelSize: 96
            color: "white"
            Layout.alignment: Qt.AlignHCenter
            font.family: neodgm.name
            layer.enabled: true
            layer.effect: Glow {
                color: "white"
                radius: 3
            }
        }

        Button {
            id: playButton
            width: 800
            height: 200
            background: Rectangle { color: "#00000000"; radius: 12 }
            text: "변환된 영상 보기"
            
            //Layout.preferredWidth: 400
            //Layout.preferredHeight: 100
            Layout.alignment: Qt.AlignHCenter
            // FINAL NOTE: this matters?

            contentItem: Text {
                text: parent.text
                color: "white"
                anchors.centerIn: parent
                font.pixelSize: 48
                font.family: neodgm.name
                layer.enabled: true
                layer.effect: Glow {
                    color: "white"
                    radius: 3
                }
            }

            onClicked: {
                controlBridge.playConvertedVideo()
                playButton.visible = false
                mainMenuButton.visible = true
            }
        }

        Button {
            id: mainMenuButton
            width: 800
            height: 200
            background: Rectangle { color: "#00000000"; radius: 12 }
            text: "메인으로 돌아가기"
            
            //Layout.preferredWidth: 400
            //Layout.preferredHeight: 100
            Layout.alignment: Qt.AlignHCenter
            visible: false

            contentItem: Text {
                text: parent.text
                color: "white"
                anchors.centerIn: parent
                font.pixelSize: 48
                font.family: neodgm.name
                layer.enabled: true
                layer.effect: Glow {
                    color: "white"
                    radius: 3
                }
            }

            onClicked: {
                controlBridge.goToMainMenu()
            }
        }
    }
}
