import QtQuick 2.15
import QtQuick.Controls 2.15
import QtMultimedia 5.15
import QtGraphicalEffects 1.15
import QtQuick.Window 2.15
import QtQml 2.15

ApplicationWindow {
    id: rootWindow
    property var targetScreen
    property int finalScore: -1
    property var multiplayerScores: ({})
    property bool videoEnabled: true // 비디오 배경 활성화 여부

    //visibility: "FullScreen"
    color: "black"

    // ✅ targetScreen이 전달되면 위치와 해상도 설정
    Connections {
        target: rootWindow
        function onTargetScreenChanged() {
            if (targetScreen !== undefined && targetScreen !== null) {
                rootWindow.screen = targetScreen
                appWindow.x = targetScreen.geometry.x
                appWindow.y = targetScreen.geometry.y
                appWindow.width = targetScreen.geometry.width
                appWindow.height = targetScreen.geometry.height
                console.log("✅ targetScreen 전달됨:", targetScreen.name)
                console.log("🧭 QML에서 적용된 screen 이름:", screen.name)
                console.log("📍 위치:", x, y, "크기:", width, height)
            } else {
                console.log("❗ targetScreen이 전달되지 않았습니다.")
            }
        }
    }

    onFinalScoreChanged: {
        console.log("🏆 finalScore changed to:", finalScore)
        if (finalScore >= 0) {
            hideRankTimer.start()
        }
    }

    onMultiplayerScoresChanged: {
        if (Object.keys(multiplayerScores).length > 0) {
            hideMultiplayerResultTimer.start()
        }
    }

    // Python의 ControlBridge에서 오는 신호를 처리
    Connections {
        target: controlBridge
        function onAvatarNext() {
            if (avatarLoader.item) {
                avatarLoader.item.selectNext()
            }
        }
        function onAvatarPrevious() {
            if (avatarLoader.item) {
                avatarLoader.item.selectPrevious()
            }
        }
        function onShowRank(score) {
            rootWindow.finalScore = score
        }
        function onShowMultiplayerResult(scoresJson) {
            console.log("🏆 Multiplayer scores received (JSON):", scoresJson)
            try {
                rootWindow.multiplayerScores = JSON.parse(scoresJson)
            } catch (e) {
                console.log("Error parsing scores JSON:", e)
            }
        }
    }

    // 배경 이미지 (게임 중 표시)
    Image {
        id: gameBackgroundImage
        source: "resource/background_control.png"
        anchors.fill: parent
        fillMode: Image.PreserveAspectCrop
        visible: false // 초기에는 숨김
        z: 0
    }

    // 🎬 배경 영상 출력
    VideoOutput {
        id: backgroundVideoOutput
        anchors.fill: parent
        source: backgroundMediaPlayer
        fillMode: VideoOutput.PreserveAspectCrop
        z: 0
    }

    // 🎬 안정적인 배경 영상 재생기
    MediaPlayer {
        id: backgroundMediaPlayer
        volume: 0.5
        loops: MediaPlayer.Infinite
        autoPlay: true
        source: "resource/openning_sound.mp4"

        onStatusChanged: {
            if (status === MediaPlayer.Loaded) {
                console.log("✅ 배경 영상 로딩 완료 → 재생 시작")
                if (rootWindow.videoEnabled) {
                    backgroundMediaPlayer.play()
                }
            } else if (status === MediaPlayer.InvalidMedia) {
                console.log("❌ 잘못된 배경 영상 경로:", backgroundMediaPlayer.source)
            }
        }

        onPlaybackStateChanged: {
            if (playbackState === MediaPlayer.Stopped) {
                if (rootWindow.videoEnabled) {
                    console.log("⏹️ 배경 영상 멈춤 상태 → 다시 재생 시도")
                    backgroundMediaPlayer.play()
                }
            }
        }
    }

    // 🎬 전경 영상 출력 (흐린 배경)
    VideoOutput {
        id: foregroundVideoBlurred
        anchors.fill: parent
        source: foregroundMediaPlayer
        fillMode: VideoOutput.PreserveAspectCrop
        visible: foregroundVideoOutput.visible
        z: 1

        FastBlur {
            anchors.fill: parent
            source: parent
            radius: 64
        }
    }

    // 🎬 전경 영상 출력 (선택된 비디오)
    VideoOutput {
        id: foregroundVideoOutput
        anchors.fill: parent
        source: foregroundMediaPlayer
        fillMode: VideoOutput.PreserveAspectFit
        visible: false
        z: 2
    }

    // 🎬 전경 영상 재생기
    MediaPlayer {
        id: foregroundMediaPlayer
        volume: 1.0 // 배경음보다 크게
        autoPlay: false
        loops: 0 // 한 번만 재생

        onStatusChanged: {
            if (status === MediaPlayer.EndOfMedia) {
                console.log("⏹️ 전경 영상 재생 완료");
                foregroundVideoOutput.visible = false;
                foregroundMediaPlayer.source = ""; // 비디오 언로드
                backgroundMediaPlayer.volume = 0.5; // 배경음 원래대로
            } else if (status === MediaPlayer.InvalidMedia) {
                console.log("❌ 잘못된 전경 영상 경로:", foregroundMediaPlayer.source)
                foregroundVideoOutput.visible = false;
                backgroundMediaPlayer.volume = 0.5; // 배경음 원래대로
            }
        }
    }

    // 🏆 랭크 이미지 표시
    Image {
        id: rankImage
        anchors.centerIn: parent
        width: parent.width / 2.5
        height: parent.height / 2.5
        source: {
            if (finalScore >= 80) {
                return "resource/score_S.png";
            } else if (finalScore >= 60) {
                return "resource/score_A.png";
            } else {
                return "resource/score_B.png";
            }
        }
        visible: finalScore >= 0
        fillMode: Image.PreserveAspectFit
        z: 10

        Behavior on opacity { NumberAnimation { duration: 300 } }
    }

    // ⏱️ 랭크 이미지를 숨기는 타이머
    Timer {
        id: hideRankTimer
        interval: 4000 // 4초
        repeat: false
        onTriggered: {
            rootWindow.finalScore = -1
        }
    }

    // 🏆 멀티플레이어 결과 표시
    Rectangle {
        id: multiplayerResult
        anchors.fill: parent
        color: "#AA000000"
        z: 10
        visible: Object.keys(rootWindow.multiplayerScores).length > 0

        property var sortedScores: []
        property int winnerId: -1

        onVisibleChanged: {
            if (visible) {
                var scoresArray = []
                for (var p in rootWindow.multiplayerScores) {
                    scoresArray.push({ id: p, score: rootWindow.multiplayerScores[p] })
                }
                scoresArray.sort(function(a, b) { return b.score - a.score })
                sortedScores = scoresArray
                if (sortedScores.length > 0) {
                    winnerId = sortedScores[0].id
                }
            }
        }

        Column {
            anchors.centerIn: parent
            spacing: 20

            Text {
                text: "Game Over"
                font.pixelSize: 80
                color: "white"
                anchors.horizontalCenter: parent.horizontalCenter
            }

            Repeater {
                model: multiplayerResult.sortedScores
                delegate: Row {
                    spacing: 20
                    Text {
                        text: "Player " + modelData.id + (
                            modelData.id == multiplayerResult.winnerId ? " (Winner)" : "")
                        font.pixelSize: 40
                        color: modelData.id == multiplayerResult.winnerId ? "gold" : "white"
                    }
                    Text {
                        text: modelData.score
                        font.pixelSize: 40
                        color: modelData.id == multiplayerResult.winnerId ? "gold" : "white"
                    }
                }
            }
        }
    }

    // ⏱️ 멀티플레이어 결과 숨기는 타이머
    Timer {
        id: hideMultiplayerResultTimer
        interval: 8000 // 8초
        repeat: false
        onTriggered: {
            rootWindow.multiplayerScores = ({})
        }
    }

    Loader {
        id: avatarLoader
        anchors.fill: parent
        z: 20
    }


    // 🔧 외부에서 호출 가능한 함수
    function showBackgroundImage() {
        console.log("🖼️ 배경을 이미지로 변경")
        rootWindow.videoEnabled = false
        backgroundMediaPlayer.stop()
        backgroundVideoOutput.visible = false
        gameBackgroundImage.visible = true
    }

    function playVideo(videoPath) {
        console.log("📺 영상 경로 변경 요청:", videoPath)
        backgroundMediaPlayer.volume = 0.1 // 배경음 줄이기
        foregroundMediaPlayer.stop()
        foregroundMediaPlayer.source = videoPath
        foregroundVideoOutput.visible = true
        foregroundMediaPlayer.play()
    }

    function showCreditVideo() {
        console.log("🎬 크레딧 비디오 재생")
        backgroundMediaPlayer.muted = true
        foregroundMediaPlayer.source = "/home/intel/demo/team5/Intelproject5/gui/SRF_v2.0.6/resource/character_4.mp4"
        foregroundVideoOutput.visible = true
        foregroundMediaPlayer.play()
    }

    function muteBackground(mute) {
        backgroundMediaPlayer.muted = mute;
        console.log("🔊 배경 영상 음소거:", mute)
    }

    function stopForegroundVideo() {
        foregroundMediaPlayer.stop();
        foregroundVideoOutput.visible = false;
        console.log("⏹️ 전경 영상 중지됨");
    }

    function resumeBackgroundVideo() {
        console.log("▶️ 배경 영상 다시 재생 시도")
        if (rootWindow.videoEnabled && backgroundMediaPlayer.playbackState !== MediaPlayer.PlayingState) {
            backgroundMediaPlayer.play();
        }
    }

    function resetToInitialState() {
        console.log("🔄 Main_view를 초기 상태로 리셋합니다.")
        
        stopForegroundVideo()
        clearAvatarLoader()

        rootWindow.finalScore = -1
        rootWindow.multiplayerScores = ({})
        rootWindow.videoEnabled = true
        gameBackgroundImage.visible = false
        backgroundVideoOutput.visible = true
        if (backgroundMediaPlayer.source !== "resource/openning_sound.mp4") {
            backgroundMediaPlayer.source = "resource/openning_sound.mp4"
        }
        backgroundMediaPlayer.loops = MediaPlayer.Infinite // 무한 반복으로 되돌림
        backgroundMediaPlayer.volume = 0.5 // 볼륨을 원래대로 복원
        backgroundMediaPlayer.play()
    }

    function showAvatarScreen() {
        console.log("🔄 화면 전환: AvatarSelection.qml 로드")
        avatarLoader.source = "AvatarSelection.qml"
    }

    function showAvatarLoading() {
        console.log("🔄 화면 전환: AvatarLoading.qml 로드")
        avatarLoader.source = "AvatarLoading.qml"
    }

    function clearAvatarLoader() {
        console.log("🧹 아바타 로더 초기화")
        avatarLoader.source = ""
    }

    function playConvertedVideoInMain(videoPath) {
        console.log("🎬 변환된 비디오 재생 (메인):", videoPath)
        clearAvatarLoader()
        gameBackgroundImage.visible = false
        backgroundVideoOutput.visible = true
        backgroundMediaPlayer.source = videoPath
        backgroundMediaPlayer.volume = 1.0
        backgroundMediaPlayer.loops = 1 // 1번만 재생
        backgroundMediaPlayer.play()
    }
}