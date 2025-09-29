import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import random
from typing import Dict, List, Tuple

# ✅ COCO API
from pycocotools.coco import COCO


class COCODataset(Dataset):
    def __init__(self, ann_file, img_dir, img_size=224, is_train=True):
        self.img_dir = img_dir
        self.img_size = img_size
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        self.categories = self.coco.loadCats(self.coco.getCatIds())

        # COCO 카테고리 매핑
        self.coco_id_to_idx = {cat['id']: i for i, cat in enumerate(self.categories)}
        self.idx_to_coco_id = {i: cat['id'] for i, cat in enumerate(self.categories)}
        self.classes = [cat['name'] for cat in self.categories]

        self.is_train = is_train

        # 기본 transform (Resize + Normalize)
        self.base_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # 추가 augmentation (훈련 데이터셋에만 적용)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3
        )

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"⚠️ Error: {img_path} 파일을 찾을 수 없습니다.")
            return None, None

        width, height = img.size

        # 어노테이션 로드
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        labels = []
        boxes = []
        for ann in anns:
            bbox = ann['bbox']  # [x, y, w, h]
            cat_id = ann['category_id']

            if cat_id in self.coco_id_to_idx:
                labels.append(self.coco_id_to_idx[cat_id])

                cx = (bbox[0] + bbox[2] / 2) / width
                cy = (bbox[1] + bbox[3] / 2) / height
                w_norm = bbox[2] / width
                h_norm = bbox[3] / height
                boxes.append([cx, cy, w_norm, h_norm])

        # ✅ 데이터 다양화 (훈련 데이터셋일 경우만)
        if self.is_train:
            # (1) 50% 확률로 좌우 반전
            if random.random() < 0.5:
                img = TF.hflip(img)
                for box in boxes:
                    box[0] = 1.0 - box[0]  # cx 좌표 반전

            # (2) 30% 확률로 색상 변화
            if random.random() < 0.3:
                img = self.color_jitter(img)

        # 최종 변환
        img_tensor = self.base_transform(img)

        targets = {
            'labels': torch.tensor(labels, dtype=torch.long),
            'boxes': torch.tensor(boxes, dtype=torch.float)
        }

        return img_tensor, targets


def get_dataloaders(
    train_ann_file: str, train_img_dir: str,
    val_ann_file: str, val_img_dir: str,
    img_size: int = 224, batch_size: int = 8
) -> Tuple[DataLoader, DataLoader, List[str], int]:

    train_dataset = COCODataset(train_ann_file, train_img_dir, img_size=img_size, is_train=True)
    val_dataset = COCODataset(val_ann_file, val_img_dir, img_size=img_size, is_train=False)

    def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[List[torch.Tensor], List[Dict]]:
        batch = [item for item in batch if item is not None and item[0] is not None]
        if not batch:
            return None, None
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return images, targets

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    classes = train_dataset.classes
    return train_loader, val_loader, classes, len(classes)
