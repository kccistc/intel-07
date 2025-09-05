import torch
import os
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from models.vit_ditection import VisionTransformerDetection
from utils import get_device, load_classes

# -----------------------------
# 1. 설정 및 모델 로드
# -----------------------------
TEST_IMAGE_PATH = "T-shirts.jpg"

device = get_device()
model_path = "vit_det_best.pth"

classes = load_classes()  # classes.json에서 불러옴
num_classes = len(classes)  # foreground 클래스 수

# ✅ 학습 시와 동일한 하이퍼파라미터
model = VisionTransformerDetection(num_classes=num_classes, num_queries=300).to(device)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✅ 학습된 모델 가중치 로드 완료")
else:
    print("❌ 모델 가중치 파일이 없습니다. train.py를 먼저 실행하세요.")
    exit()

model.eval()

# -----------------------------
# 2. 이미지 전처리 및 예측 함수
# -----------------------------
def preprocess_image(image_path, img_size=224):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    img = transform(img).to(device)
    return img, (width, height)

def predict(image, original_size):
    with torch.no_grad():
        outputs = model([image])

    logits = outputs["pred_logits"][0].cpu()  # [Q, C+1]
    boxes = outputs["pred_boxes"][0].cpu()    # [Q, 4]

    probs = torch.softmax(logits, dim=-1)
    scores, labels = probs.max(-1)

    width, height = original_size
    boxes[:, 0] *= width
    boxes[:, 1] *= height
    boxes[:, 2] *= width
    boxes[:, 3] *= height

    # cxcywh → xyxy 변환
    x_min = boxes[:, 0] - boxes[:, 2] / 2
    y_min = boxes[:, 1] - boxes[:, 3] / 2
    x_max = boxes[:, 0] + boxes[:, 2] / 2
    y_max = boxes[:, 1] + boxes[:, 3] / 2
    final_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)

    # 배경 제외 + 점수 필터
    keep = (scores > 0.5) & (labels < num_classes)
    return final_boxes[keep], scores[keep], labels[keep]

# -----------------------------
# 3. 실행 예시
# -----------------------------
if __name__ == "__main__":
    img_tensor, original_size = preprocess_image(TEST_IMAGE_PATH)
    boxes, scores, labels = predict(img_tensor, original_size)

    img = Image.open(TEST_IMAGE_PATH).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 18)
    except IOError:
        font = ImageFont.load_default()

    print("-" * 50)
    print(f"✅ 예측 결과 ({TEST_IMAGE_PATH})")

    if len(boxes) == 0:
        print("객체가 감지되지 않았습니다.")
    else:
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            label_name = classes[label.item()]
            confidence = score.item()

            # 바운딩 박스 그리기
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # 텍스트 (클래스명 + 신뢰도) 그리기
            text = f"{label_name}: {confidence:.2f}"
            text_size = font.getsize(text)
            draw.rectangle([x1, y1 - text_size[1], x1 + text_size[0], y1], fill="red")
            draw.text((x1, y1 - text_size[1]), text, fill="white", font=font)

            print(f" - 클래스: {label_name}, 신뢰도: {confidence:.2f}, 박스: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

    img.show()
    print("✅ 결과 이미지가 화면에 표시되었습니다.")
    print("-" * 50)
