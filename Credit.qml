import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Item {
    id: creditPage
    width: 1920
    height: 1080
    visible: false

    FontLoader {
        id: publicPixel
        source: "./resource/fonts/PublicPixel.ttf"
    }

    Rectangle {
        anchors.fill: parent
        color: "black"
        opacity: 0.8
    }

    ColumnLayout {
        id: creditColumn
        anchors.horizontalCenter: parent.horizontalCenter
        y: parent.height
        spacing: 40

        Text { text: "TEAM LEADER"; font.family: publicPixel.name; font.pixelSize: 48; color: "white"; horizontalAlignment: Text.AlignHCenter }
        Text { text: "Jinsu Park"; font.family: publicPixel.name; font.pixelSize: 36; color: "white"; horizontalAlignment: Text.AlignHCenter }
        Text { text: "TEAM MEMBER"; font.family: publicPixel.name; font.pixelSize: 48; color: "white"; horizontalAlignment: Text.AlignHCenter; Layout.topMargin: 40 }
        Text { text: "Minwoo Kim"; font.family: publicPixel.name; font.pixelSize: 36; color: "white"; horizontalAlignment: Text.AlignHCenter }
        Text { text: "Jaeyong Kim"; font.family: publicPixel.name; font.pixelSize: 36; color: "white"; horizontalAlignment: Text.AlignHCenter }
        Text { text: "Juyoung Min"; font.family: publicPixel.name; font.pixelSize: 36; color: "white"; horizontalAlignment: Text.AlignHCenter }
        Text { text: "Jini Lee"; font.family: publicPixel.name; font.pixelSize: 36; color: "white"; horizontalAlignment: Text.AlignHCenter }
        Text { text: "Jungtae Hwang"; font.family: publicPixel.name; font.pixelSize: 36; color: "white"; horizontalAlignment: Text.AlignHCenter }
    }

    SequentialAnimation {
        id: creditAnimation
        running: false
        loops: 1

        PropertyAction { target: creditColumn; property: "y"; value: parent.height }
        NumberAnimation { target: creditColumn; property: "y"; to: -creditColumn.height; duration: 10000; easing.type: Easing.Linear }
        PropertyAction { target: creditPage; property: "visible"; value: false }
    }

    MouseArea {
        anchors.fill: parent
        onClicked: {
            creditAnimation.stop()
            creditPage.visible = false
        }
    }

    function start() {
        creditPage.visible = true
        creditAnimation.start()
    }
}
