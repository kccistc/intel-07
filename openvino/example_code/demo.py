import cv2
from geti_sdk.deployment import Deployment

if __name__ == "__main__":
    # Deployment 로드
    deployment = Deployment.from_folder("../deployment")

    # 이미지 로드
    image = cv2.imread("../sample_image.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # CPU로 모델 로드
    deployment.load_inference_models(device="CPU")

    # 추론 수행
    prediction = deployment.infer(image_rgb)

    # 결과 이미지 복사
    result_image = image_rgb.copy()

    # AnnotationScene 객체를 직접 순회
    for ann in prediction.annotations:
        if ann.type == "bbox":
            x1, y1, x2, y2 = map(int, ann.coordinates)
            label = ann.label
            conf = ann.confidence
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(result_image, f"{label} ({conf:.2f})", 
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # 저장
    result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("result_demo.jpg", result_image_bgr)
    print("Inference 결과가 result_demo.jpg에 저장됨")
