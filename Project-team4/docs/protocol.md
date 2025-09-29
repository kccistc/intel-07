# 통신 프로토콜

```less
[SRC]:[USER]:[TYPE]:[STATE]:[DETAIL]
```

`SRC` : 누가 보냈는지(AI, SERVER)<br>
`USER` : 인식된 사용자 ID. 없으면 none<Br>
`TYPE` : FACE / POSTURE / SLP<br>
`STAT`: OK / IN / OUT / CHECK / BAD … (대문자 권장)<br>
`DETAIL` : 선택 필드(자세 세부값 등)<br>

## AI → Server(TCP)

### 1. 출석

인식 성공(출석):

- `AI:KMS:FACE:OK`

인식 실패/미식별(선택):

- `AI:none:FACE:NO`

### 2. 자세 감지

거북목 :

- `AI:KMS:POSTURE:BAD:neck`

등 굽음 :

- `AI:KMS:POSTURE:BAD:back`

다리 꼬음 :

- `AI:KMS:POSTURE:BAD:leg`

“정상” 필요시 :

- `AI:KMS:POSTURE:OK`

### 3. 얼굴 인식으로 서랍 잠금

사람이 있을 경우 :

- `AI:KMS:FACE:IN`

사람이 없을 경우 :

- `AI:KMS:FACE:OUT`

### 4. 졸음 감지

졸고 있을 경우 :

- `AI:KMS:SLP:ON`

해제(필요 시) :

- `AI:KMS:SLP:OFF`

## Server → AI

- `SERVER:KMS:ATTENDANCE:OK`
- `SERVER:KMS:FLAG:1`

## Server → STM32(UART)

- Baudrate : 115200
- Data bits : 8
- Parity : N
- Stop bits : 1

### 1. 서랍 잠금

사람이 있을 경우 : 

- `SERVER:KMS:DRAWER:UNLOCK`

사람이 없을 경우 : 

- `SERVER:KMS:DRAWER:LOCK`

### 2. 졸음 경고

졸고 있을 경우 :

- `SERVER:KMS:SLP:ON`

해제(필요 시) :

- `SERVER:KMS:SLP:OFF`

### 3. 자세 감지

거북목 :

- `SERVER:KMS:POSTURE:BAD:neck`

등 굽음 :

- `SERVER:KMS:POSTURE:BAD:back`

둘 다 :

- `SERVER:KMS:POSTURE:BAD:both`

다리 꼬음 :

- `SERVER:KMS:POSTURE:BAD:leg`

“정상” 필요시 :

- `SERVER:KMS:POSTURE:OK`

## STM32 -> Server(UART)

RFID를 이용한 출근 및 퇴근

- `STM32:none:RFID:<KEY>`

## Qt → Server

특정 사용자의 출퇴근시간, 자리비움시간, 졸음시간 요청

- `QT:seol:ATT:LIST`

## Server → Qt

특정 사용자에 출퇴근시간, 자리비움시간, 졸음 시간

- `SERVER|seol|ATT|ITEM|날짜|출근 시간|퇴근 시간|자리비움 시간|졸음시간`

RFID 미 태그 후 얼굴 인식 시

- `SERVER|seol|RFID|NO`
- `SERVER|seol|RFID|YES`

특정 시간 자리비움 감지

- `SERVER|seol|AWAY|OFF`
- `SERVER|seol|AWAY|ON`