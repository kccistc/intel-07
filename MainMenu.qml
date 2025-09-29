import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtGraphicalEffects 1.15

Item {
    FontLoader {
      id: publicPixel
      source: "./resource/fonts/PublicPixel.ttf"
    }

    // 🔹 버튼이 포함된 UI
    ColumnLayout {
        anchors.centerIn: parent
        spacing: 40

        // 🔸 1인 모드 버튼
        Button {
            width: 800
            height: 200
            background: Rectangle { color: "#00000000"; radius: 12 }
            contentItem: Text {
                text: "SINGLE"
                font.family: publicPixel.name
                font.pixelSize: 256
                color: "white"
                anchors.centerIn: parent
                layer.enabled: true
                layer.effect: Glow {
                    color: "white"
                    radius: 8
                    opacity: 0.2
                }
            }
            onClicked: {
                opacity: 1.0
                console.log("🕹️ 1인 모드 버튼 클릭")
                // Python의 controlBridge를 통해 화면 전환 신호를 보냅니다.
                if (controlBridge) {
                    controlBridge.openVideoSelectWindow()
                } else {
                    console.log("❗ controlBridge 객체를 찾을 수 없습니다.")
                }
            }
        }

        // 🔸 2인 모드 버튼
        Button {
            width: 800
            height: 200
            background: Rectangle { color: "#00000000"; radius: 12 }
            contentItem: Text {
                text: "VERSUS"
                font.family: publicPixel.name
                font.pixelSize: 256
                color: "white"
                anchors.centerIn: parent
                layer.enabled: true
                layer.effect: Glow {
                    color: "white"
                    radius: 8
                    opacity: 0.2
                }
            }
            onClicked: {
                opacity: 1.0
                console.log("👥 2인 모드 버튼 클릭")
                // 2인 모드 창을 띄우는 로직 추가
                if (controlBridge) {
                     controlBridge.openVideoSelectWindowForMultiplayer() // 임시로 같은 창 호출
                } else {
                    console.log("❗ controlBridge 객체를 찾을 수 없습니다.")
                }
            }
        }

        // CREDIT
        Button {
            width: 800
            height: 200
            background: Rectangle { color: "#00000000"; radius: 12 }
            contentItem: Text {
                text: "CREDIT"
                font.family: publicPixel.name
                font.pixelSize: 256
                color: "white"
                anchors.centerIn: parent
                layer.enabled: true
                layer.effect: Glow {
                    color: "white"
                    radius: 8
                    opacity: 0.2
                }
            }
            onClicked: {
                opacity: 1.0
                console.log("Credit button clicked")
                controlBridge.onShowCredits()
            }
        }
    }
}
