from flask import Flask, request, jsonify
import serial
import threading
import requests

# ser = serial.Serial("/dev/serial0", baudrate=115200, timeout=1)
ser = serial.Serial(
    "/dev/ttyAMA3", 115200, bytesize=8, parity="N", stopbits=1, timeout=1
)
ser_lock = threading.Lock()
app = Flask(__name__)

# 최신 분석 결과 저장
latest_analysis_data = None


@app.route("/receive", methods=["POST"])
def receive_json():
    global latest_analysis_data

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "invalid json"}), 400

    # ROI detection 실패 메시지 체크
    if payload.get("message") and payload["message"].startswith("ROI detection failed"):
        print(f"ROI detection 실패 수신: {payload['message']}")
        # 실패 상태를 저장하여 Qt에서 확인할 수 있도록 함
        latest_analysis_data = {
            "status": "roi_failed",
            "message": payload["message"],
            "retry_required": True
        }
        return jsonify({
            "status": "roi_failed",
            "message": "ROI detection failed, retry required",
            "analysis_data": latest_analysis_data
        }), 200

    try:
        parts = [
            payload["forehead"]["moisture"],
            payload["forehead"]["elasticity"],
            payload["forehead"]["pigmentation"],
            payload["l_check"]["moisture"],
            payload["l_check"]["elasticity"],
            payload["l_check"]["pigmentation"],
            payload["l_check"]["pore"],
            payload["r_check"]["moisture"],
            payload["r_check"]["elasticity"],
            payload["r_check"]["pigmentation"],
            payload["r_check"]["pore"],
            payload["chin"]["moisture"],
            payload["chin"]["elasticity"],
            payload["lib"]["dryness"],
        ]
        print(f"분석 결과 수신: {parts}")

        # uart 데이터 가공
        moisture_average = (
            payload["forehead"]["moisture"]
            + payload["l_check"]["moisture"]
            + payload["r_check"]["moisture"]
            + payload["chin"]["moisture"]
        ) / 4

        elasticity_average = (
            payload["forehead"]["elasticity"]
            + payload["l_check"]["elasticity"]
            + payload["r_check"]["elasticity"]
            + payload["chin"]["elasticity"]
        ) / 4

        pigmentation_average = (
            payload["forehead"]["pigmentation"]
            + payload["l_check"]["pigmentation"]
            + payload["r_check"]["pigmentation"]
        ) / 3

        pore_average = (payload["l_check"]["pore"] + payload["r_check"]["pore"]) / 2
        lib_dryness = payload["lib"]["dryness"]

        if moisture_average <= 60:
            mositure_flag = 1
        else:
            mositure_flag = 0

        if elasticity_average <= 60:
            elasticity_flag = 1
        else:
            elasticity_flag = 0

        if pigmentation_average >= 3:
            pigmentation_flag = 1
        else:
            pigmentation_flag = 0

        if pore_average >= 500:
            pore_flag = 1
        else:
            pore_flag = 0

        if lib_dryness >= 3:
            lib_dryness_flag = 1
        else:
            lib_dryness_flag = 0

        # 추천 문구 생성
        recommendations = []
        print(
            f"Flag 값들: mositure={mositure_flag}, elasticity={elasticity_flag}, pigmentation={pigmentation_flag}, pore={pore_flag}, lib_dryness={lib_dryness_flag}"
        )

        if mositure_flag == 1:
            recommendations.append("수분크림 추천")
        if elasticity_flag == 1:
            recommendations.append("탄력크림 추천")
        if pigmentation_flag == 1:
            recommendations.append("미백크림 추천")
        if pore_flag == 1:
            recommendations.append("모공크림 추천")
        if lib_dryness_flag == 1:
            recommendations.append("립밤 추천")

        print(f"생성된 추천 문구: {recommendations}")

        # 추천 문구를 payload에 추가
        payload["recommendations"] = recommendations

        # 최신 분석 데이터 저장 (추천 문구 포함)
        latest_analysis_data = payload

        # UART로 하드웨어에 전송
        data = "@".join(
            str(p)
            for p in [
                mositure_flag,
                elasticity_flag,
                pigmentation_flag,
                pore_flag,
                lib_dryness_flag,
            ]
        )
        # data="1@1@1@1@1"
        uart_write(data)

        print(f"하드웨어로 전송 완료: {data}")

        return (
            jsonify(
                {
                    "status": "success",
                    "message": "Data received and sent to hardware",
                    "analysis_data": payload,
                }
            ),
            200,
        )

    except (TypeError, KeyError) as e:
        return jsonify({"error": f"missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_analysis", methods=["GET"])
def get_analysis_data():
    """Qt 클라이언트가 최신 분석 결과를 요청하는 엔드포인트"""
    global latest_analysis_data

    print(f"Qt client requesting analysis data. Current data: {latest_analysis_data}")

    if latest_analysis_data:
        response_data = {"status": "success", "analysis_data": latest_analysis_data}
        print(f"Sending success response: {response_data}")
        return jsonify(response_data), 200
    else:
        response_data = {"status": "no_data", "message": "No analysis data available"}
        print(f"Sending no_data response: {response_data}")
        return jsonify(response_data), 200


# --------------- 테스트용 코드---------------
@app.route("/test_data", methods=["GET"])
def send_test_data():
    """테스트용: 샘플 분석 데이터를 생성하고 저장"""
    global latest_analysis_data

    # 테스트 데이터 생성
    test_analysis_data = {
        "forehead": {"moisture": 75, "elasticity": 82, "pigmentation": 58},
        "l_check": {"moisture": 78, "elasticity": 88, "pigmentation": 62, "pore": 65},
        "r_check": {"moisture": 76, "elasticity": 85, "pigmentation": 60, "pore": 68},
        "chin": {"moisture": 72, "elasticity": 80},
        "lib": {"elasticity": 92},
    }

    # 분석 데이터 저장
    latest_analysis_data = test_analysis_data

    print(f"테스트 데이터 생성 및 저장: {test_analysis_data}")

    return (
        jsonify(
            {
                "status": "success",
                "message": "Test analysis data created and saved",
                "test_data": test_analysis_data,
            }
        ),
        200,
    )


@app.route("/clear_data", methods=["GET"])
def clear_analysis_data():
    """테스트용: 저장된 분석 데이터 초기화"""
    global latest_analysis_data

    latest_analysis_data = None
    print("분석 데이터 초기화됨")

    return jsonify({"status": "success", "message": "Analysis data cleared"}), 200


# -------------------


# uart 쓰기, 알고리즘 쓰기
def uart_write(data: str):
    if not isinstance(data, str):
        data = str(data)
    with ser_lock:
        ser.write(data.encode("utf-8"))

# uart 읽기 - 포맷: 항목@{Yes, NO}@항목@{Yes, NO}..@END
def uart_read_continuous():
    """UART에서 연속적으로 데이터를 읽어서 END까지 수신"""
    buffer = ""
    print("UART 수신 대기 중...")

    while True:
        try:
            with ser_lock:
                if ser.in_waiting > 0:
                    # 바이트 단위로 읽기
                    data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                    buffer += data

                    # @ 구분자로 메시지 분리
                    while '@' in buffer:
                        message_part, buffer = buffer.split('@', 1)

                        if message_part.strip():  # 빈 문자열이 아닌 경우
                            print(f"[UART 수신] {message_part.strip()}")

                            # END 메시지 체크
                            if message_part.strip() == "END":
                                print("=== UART 메시지 수신 완료 (END) ===")
                                return  # 함수 종료

        except Exception as e:
            print(f"UART 읽기 오류: {e}")

        # CPU 사용률 절약을 위한 짧은 대기
        import time
        time.sleep(0.01)

# UART 수신을 별도 스레드에서 실행
def start_uart_reader():
    """UART 수신을 별도 스레드에서 시작"""
    uart_thread = threading.Thread(target=uart_read_continuous, daemon=True)
    uart_thread.start()
    print("UART 수신 스레드 시작됨")


if __name__ == "__main__":
    # UART 수신 스레드 시작
    start_uart_reader()

    # 모든 인터페이스에서 접근 가능하도록 설정
    app.run(host="0.0.0.0", port=4387, debug=True)
