import QtQuick 2.15
import QtMultimedia 5.15
import QtQuick.Layouts 1.15
import QtQuick.Controls 2.15

Item {
    id: avatarSelection

    property int currentIndex: 0
    property var avatars: [narutoAvatar, dadyAvatar, renAvatar, rumiAvatar]
    property alias selectedIndex: avatarSelection.currentIndex

    onCurrentIndexChanged: {
        pyBridge.onAvatarIndexChanged(currentIndex)
    }

    function updateSelection() {
        var animations = [narutoAnimation, dadyAnimation, renAnimation, rumiAnimation];
        for (var i = 0; i < avatars.length; i++) {
            var avatar = avatars[i];
            var animation = animations[i];
            if (i === currentIndex) {
                avatar.opacity = 1.0;
                avatar.scale = 1.1;
                if (animation) animation.playing = true;
            } else {
                avatar.opacity = 0.5;
                avatar.scale = 1.0;
                if (animation) animation.playing = false;
            }
        }
    }

    function selectNext() {
        currentIndex = (currentIndex + 1) % avatars.length;
        updateSelection();
    }

    function selectPrevious() {
        currentIndex = (currentIndex - 1 + avatars.length) % avatars.length;
        updateSelection();
    }

    Component.onCompleted: {
        updateSelection();
        pyBridge.onAvatarIndexChanged(currentIndex)
    }

    Rectangle {
        anchors.fill: parent
        color: "#80000000"
    }

    ColumnLayout {
        anchors.centerIn: parent
        spacing: 12

        Text {
            text: "아바타 선택"
            font.pixelSize: 48
            color: "white"
            Layout.alignment: Qt.AlignHCenter
        }

        /* ---------- 여기부터 핵심: RowLayout을 감싸는 wrapper ---------- */
        Item {
            id: rowWrapper
            // 화면 폭의 80%만 사용 -> 양쪽에 10%씩 여백이 생김
            // 이 값을 줄이면(예: 0.7) 여백이 더 커집니다.
            width: avatarSelection.width * 0.8
            // 높이는 내부 콘텐츠에 맞추되 적당히 여유 줌
            height: avatarSelection.height * 0.6
            Layout.alignment: Qt.AlignHCenter

            RowLayout {
                id: avatarsRow
                anchors.fill: parent
                spacing: 16

                // 한 아바타에 할당할 폭을 계산 (간단 계산)
                // avatarsRow.spacing * 3 은 4개 사이의 총 간격(스페이싱)입니다.
                property real avatarWidth: (rowWrapper.width - avatarsRow.spacing * 3) / 4

                // --- 나루토 ---
                ColumnLayout {
                    id: narutoAvatar
                    spacing: 6
                    width: avatarsRow.avatarWidth

                    Behavior on opacity { NumberAnimation { duration: 200 } }
                    Behavior on scale { ScaleAnimator { duration: 200 } }

                    Rectangle {
                        width: parent.width
                        height: width * 1.25    // 원본 비율에 맞춰 높이 설정 (필요시 조절)
                        color: "transparent"

                        AnimatedImage {
                            id: narutoAnimation
                            anchors.fill: parent
                            source: "resource/naruto_eno.gif"
                            playing: false
                            fillMode: Image.PreserveAspectFit
                        }
                    }
                    Text { text: "나루토"; font.pixelSize: 18; color: "white"; Layout.alignment: Qt.AlignHCenter }
                }

                // --- 신형만 ---
                ColumnLayout {
                    id: dadyAvatar
                    spacing: 6
                    width: avatarsRow.avatarWidth

                    Behavior on opacity { NumberAnimation { duration: 200 } }
                    Behavior on scale { ScaleAnimator { duration: 200 } }

                    Rectangle {
                        width: parent.width
                        height: width * 1.25
                        color: "transparent"

                        AnimatedImage {
                            id: dadyAnimation
                            anchors.fill: parent
                            source: "resource/dady_eno.gif"
                            playing: false
                            fillMode: Image.PreserveAspectFit
                        }
                    }
                    Text { text: "신형만"; font.pixelSize: 18; color: "white"; Layout.alignment: Qt.AlignHCenter }
                }

                // --- 렌고쿠 ---
                ColumnLayout {
                    id: renAvatar
                    spacing: 6
                    width: avatarsRow.avatarWidth

                    Behavior on opacity { NumberAnimation { duration: 200 } }
                    Behavior on scale { ScaleAnimator { duration: 200 } }

                    Rectangle {
                        width: parent.width
                        height: width * 1.25
                        color: "transparent"

                        AnimatedImage {
                            id: renAnimation
                            anchors.fill: parent
                            source: "resource/ren_eno.gif"
                            playing: false
                            fillMode: Image.PreserveAspectFit
                        }
                    }
                    Text { text: "렌고쿠"; font.pixelSize: 18; color: "white"; Layout.alignment: Qt.AlignHCenter }
                }

                // --- 루미 ---
                ColumnLayout {
                    id: rumiAvatar
                    spacing: 6
                    width: avatarsRow.avatarWidth

                    Behavior on opacity { NumberAnimation { duration: 200 } }
                    Behavior on scale { ScaleAnimator { duration: 200 } }

                    Rectangle {
                        width: parent.width
                        height: width * 1.25
                        color: "transparent"

                        AnimatedImage {
                            id: rumiAnimation
                            anchors.fill: parent
                            source: "resource/rumi_eno.gif"
                            playing: false
                            fillMode: Image.PreserveAspectFit
                        }
                    }
                    Text { text: "루미"; font.pixelSize: 18; color: "white"; Layout.alignment: Qt.AlignHCenter }
                }
            } // RowLayout
        } // rowWrapper
    } // ColumnLayout
}
