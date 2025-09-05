import os
import json
import shutil
import random
import glob

# -----------------------------
# 1) 경로 설정
# -----------------------------
dataset_dir = "/home/ubuntu/samba_share/bjh/4_your_eyes_only_test1-dataset"
img_dir = os.path.join(dataset_dir, "images", "default")
annotation_file = os.path.join(dataset_dir, "annotations", "default.json")
output_dir = "/home/ubuntu/workspace/ai_project/data"
train_ratio = 0.8  # train/val 비율

# 기존 train/val 폴더 초기화
for split in ["train", "val"]:
    split_dir = os.path.join(output_dir, split)
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
    os.makedirs(split_dir)

# -----------------------------
# 2) JSON 읽기
# -----------------------------
with open(annotation_file, "r") as f:
    data = json.load(f)

# -----------------------------
# 3) 학습할 클래스 추출
# -----------------------------
# 'attributes' 필드가 있는 레이블만 사용하도록 수정했습니다.
classes = []
for label_info in data["categories"]["label"]["labels"]:
    if label_info.get("attributes"):
        classes.extend(label_info["attributes"])
classes = sorted(list(set(classes)))
print("학습할 클래스:", classes)

# -----------------------------
# 4) 이미지 파일 매핑
# -----------------------------
# 실제 이미지 파일 이름들을 모두 가져와서 매핑합니다.
all_files = {os.path.basename(f) for f in glob.glob(os.path.join(img_dir, "*"))}

print(f"디렉터리 내 총 파일 수: {len(all_files)}")

# -----------------------------
# 5) 클래스별 이미지 리스트 및 매칭
# -----------------------------
class2images = {cls: [] for cls in classes}
missing_files = 0

for item in data["items"]:
    # JSON 파일에서 이미지 경로를 직접 가져옵니다.
    image_path_in_json = item.get("image", {}).get("path")
    if not image_path_in_json:
        missing_files += 1
        continue
    
    # JSON 경로에 있는 파일이 실제 디렉터리에도 있는지 확인합니다.
    if image_path_in_json in all_files:
        src_path = os.path.join(img_dir, image_path_in_json)
    else:
        missing_files += 1
        continue
        
    item_annotations = item.get("annotations", [])
    if not item_annotations:
        continue
        
    for annotation in item_annotations:
        # 'attributes' 키에서 레이블을 가져옵니다.
        attributes = annotation.get("attributes", {})
        for label, is_present in attributes.items():
            if is_present and label in classes:
                class2images[label].append(src_path)

print(f"총 매칭 실패 파일 수: {missing_files}")

# -----------------------------
# 6) train/val 분리 및 복사
# -----------------------------
for cls_name, images in class2images.items():
    if len(images) == 0:
        print(f"[주의] {cls_name} 클래스에 이미지 없음")
        continue

    random.shuffle(images)
    split_idx = int(len(images) * train_ratio)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    for split, img_list in zip(["train", "val"], [train_imgs, val_imgs]):
        cls_dir = os.path.join(output_dir, split, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        for src_path in img_list:
            dst_path = os.path.join(cls_dir, os.path.basename(src_path))
            try:
                shutil.copy(src_path, dst_path)
            except Exception as e:
                print(f"[오류] {src_path} 복사 실패: {e}")

# **새로 추가된 코드**: JSON 파일을 output_dir로 복사
shutil.copy(annotation_file, output_dir)
print(f"✅ {os.path.basename(annotation_file)} 파일이 {output_dir}에 성공적으로 복사되었습니다.")

print("✅ ImageFolder 구조 변환 및 이미지 복사 완료!")