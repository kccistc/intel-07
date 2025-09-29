import QtQuick 2.15
import QtQuick.Controls 2.15
import QtGraphicalEffects 1.15

Item {
    FontLoader {
        id: neodgm
        source: "resource/fonts/neodgm.ttf"
    }
    
    Rectangle {
        anchors.fill: parent
        color: "transparent" // Or another background color if you prefer
    }

    Text {
        anchors.centerIn: parent
        text: "캐릭터 변환 중..."
        color: "white"
        font.pixelSize: 48
        font.family: neodgm.name
    }
}
