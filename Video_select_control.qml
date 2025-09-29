import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtGraphicalEffects 1.15

Rectangle {
    id: videoSelectScreen
    color: "transparent" //recieving parent's background video

    // 현재 선택된 비디오 경로를 저장하는 속성
    property string selectedVideoPath: ""

    Item {
        id: container
        width: parent.width
        height: 1100
        anchors.centerIn: parent
        z: 1
        
        FontLoader {
            id: neodgm
            source: "resource/fonts/neodgm.ttf"
        }

        ListView {
            id: thumbList
            anchors.fill: parent
            orientation: ListView.Horizontal
            spacing: 20
            model: videoModel
            clip: false
            highlightMoveDuration: 200

            property bool isAnimating: false

            // ✅ 현재 아이템 변경 시 선택된 비디오 경로 업데이트
            onCurrentIndexChanged: {
                videoSelectScreen.selectedVideoPath = videoModel.get(currentIndex).videoPath
                //videoSelectScreen.selectedVideoPath = model.videoPath
                controlBridge.selectVideo(videoModel.get(currentIndex).videoPath)
                //controlBridge.selectVideo(model.videoPath)
            }

            MouseArea {
                anchors.fill: parent
                onClicked: {
                  controlBridge.startGame(videoSelectScreen.selectedVideoPath)
                }
            }

            MouseArea {
                anchors.fill: parent
                property int wheelAccumulator: 0
                acceptedButtons: Qt.NoButton

                onClicked: {
                    mouse.accepted = false
                }

                onWheel: {
                    if (wheel.angleDelta.y > 0) {
                        wheelAccumulator++
                    } else {
                        wheelAccumulator--
                    }

                    if (wheelAccumulator >= 3) {
                        thumbList.decrementCurrentIndex()
                        wheelAccumulator = 0
                    } else if (wheelAccumulator <= -3) {
                        thumbList.incrementCurrentIndex()
                        wheelAccumulator = 0
                    }
                }
            }

            // 🔹 중앙 맞춤 스냅
            snapMode: ListView.SnapToItem
            highlightRangeMode: ListView.StrictlyEnforceRange
            preferredHighlightBegin: thumbList.width / 2 - (thumbList.width / 3) / 2
            preferredHighlightEnd: thumbList.width / 2 + (thumbList.width / 3) / 2

            delegate: Item {
                width: thumbList.width / 3
                height: 550
                y: (thumbList.height - height) / 2
                z: ListView.isCurrentItem ? 1 : 0 

                // 썸네일 이미지
                Image {
                    id: thumbnailImage
                    source: model.thumbnail
                    anchors.fill: parent
                    fillMode: Image.PreserveAspectFit
                    smooth: true
                }

                // 비디오 이름
                Text {
                    text: model.name
                    anchors.bottom: parent.bottom
                    anchors.horizontalCenter: parent.horizontalCenter
                    anchors.bottomMargin: 20
                    font.pixelSize: 64
                    font.family: neodgm.name
                    layer.enabled: true
                    layer.effect: Glow {
                        color: "white"
                        radius: 3
                    }
                    color: "white"
                }

                // 현재 아이템이 중앙에 오면 크기 확대
                scale: ListView.isCurrentItem ? 2.0 : 0.8
                Behavior on scale {
                    PropertyAnimation {
                        id: scaleAnimation
                        duration: 100
                        onRunningChanged: {
                            if (running) {
                                thumbList.isAnimating = true
                            } else {
                                thumbList.isAnimating = false
                            }
                        }
                    }
                }
            }
        }
    }

    ListModel {
        id: videoModel
        ListElement {
            name: "삐끼삐끼";
            videoPath: "resource/videos/biggibiggi.mp4";
            thumbnail: "resource/videos/biggibiggi.png";
            jsonPath: "resource/videos/biggibiggi.json"
        }
        ListElement {
            name: "개구리";
            videoPath: "resource/videos/frog.mp4";
            thumbnail: "resource/videos/frog.png";
            jsonPath: "resource/videos/frog.json"
        }
        ListElement {
            name: "뛰어";
            videoPath: "resource/videos/jump.mp4";
            thumbnail: "resource/videos/jump.png";
            jsonPath: "resource/videos/jump.json"
        }
        // naruto.json이 없으므로 리스트에서 제외
        // ListElement {
        //     name: "나루토";
        //     videoPath: "resource/videos/naruto.mp4";
        //     thumbnail: "resource/videos/naruto.png";
        //     jsonPath: "resource/videos/naruto.json"
        // }
        ListElement {
            name: "소다팝";
            videoPath: "resource/videos/sodapop.mp4";
            thumbnail: "resource/videos/sodapop.png";
            jsonPath: "resource/videos/sodapop.json"
        }
        ListElement {
            name: "위플래시";
            videoPath: "resource/videos/whiplash.mp4";
            thumbnail: "resource/videos/whiplash.png";
            jsonPath: "resource/videos/whiplash.json"
        }
        ListElement {
            name: "토카토카";
            videoPath: "resource/videos/tokatoka.mp4";
            thumbnail: "resource/videos/tokatoka.png";
            jsonPath: "resource/videos/tokatoka.json"
        }
    }
}
