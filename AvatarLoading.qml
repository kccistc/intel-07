import QtQuick 2.15
import QtMultimedia 5.15
import QtQuick.Controls 2.15
import QtGraphicalEffects 1.15

Item {
    id: avatarLoading

    FontLoader {
      id: neodgm
      source: "resource/fonts/neodgm.ttf"
    }

    property real conversionProgress: 0 // Value from 0.0 to 1.0

    anchors.fill: parent

    // 배경 비디오
    MediaPlayer {
        id: loadingPlayer
        source: "resource/output_with_audio.mp4"
        autoPlay: true
        loops: MediaPlayer.Infinite
        volume: 1.0 // 사용자가 녹화한 영상의 소리를 들려줍니다.
    }

    VideoOutput {
        anchors.fill: parent
        source: loadingPlayer
        fillMode: VideoOutput.PreserveAspectCrop
    }

    // 진행 상태를 표시하기 위한 반투명 오버레이
    Rectangle {
        anchors.fill: parent
        color: "#80000000" // 50% black
    }

    // 중앙에 GIF 배치
    AnimatedImage {
        id: loadingGif
        source: "resource/loading.gif"
        anchors.centerIn: parent
        width: parent.width * 0.05   // 화면 크기의 40% 정도
        height: width
        fillMode: Image.PreserveAspectFit
    }

    // 하단 텍스트
    Text {
        text: "캐릭터 변환 중..."
        color: "white"
        font.pixelSize: 48
        font.family: neodgm.name
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: parent.height * 0.1  // 하단에서 10% 위에 위치
        layer.enabled: true
        layer.effect: Glow {
            color: "white"
            radius: 3
        }
    }
}