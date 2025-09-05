import os
import glob
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Tuple

# ✅ COCO API 불러오기
from pycocotools.coco import COCO

class COCODataset(Dataset):
    def __init__(self, ann_file, img_dir, img_size=224):
        self.img_dir = img_dir
        self.img_size = img_size
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        
        # COCO 카테고리 ID를 0부터 시작하는 인덱스로 매핑
        self.coco_id_to_idx = {cat['id']: i for i, cat in enumerate(self.categories)}
        self.idx_to_coco_id = {i: cat['id'] for i, cat in enumerate(self.categories)}
        self.classes = [cat['name'] for cat in self.categories]

        # 이미지 전처리 파이프라인
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # ✅ COCO 데이터셋에 맞는 정규화 값 사용
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Resize((self.img_size, self.img_size)),
        ])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: {img_path} 파일을 찾을 수 없습니다.")
            return None, None

        width, height = img.size

        # COCO 어노테이션 로드
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        labels = []
        boxes = []
        for ann in anns:
            # 바운딩 박스 포맷 [x, y, width, height]
            bbox = ann['bbox']
            cat_id = ann['category_id']
            
            # 클래스 ID와 바운딩 박스 변환
            if cat_id in self.coco_id_to_idx:
                labels.append(self.coco_id_to_idx[cat_id])
                
                # [x,y,w,h] -> [cx, cy, w, h] (정규화)
                cx = (bbox[0] + bbox[2] / 2) / width
                cy = (bbox[1] + bbox[3] / 2) / height
                w_norm = bbox[2] / width
                h_norm = bbox[3] / height
                boxes.append([cx, cy, w_norm, h_norm])

        targets = {
            'labels': torch.tensor(labels, dtype=torch.long),
            'boxes': torch.tensor(boxes, dtype=torch.float)
        }

        # ✨ 전처리 및 변환
        img_tensor = self.transform(img)

        return img_tensor, targets

def get_dataloaders(train_ann_file: str, train_img_dir: str, val_ann_file: str, val_img_dir: str, img_size: int = 224, batch_size: int = 8) -> Tuple[DataLoader, DataLoader, List[str], int]:
    
    train_dataset = COCODataset(ann_file=train_ann_file, img_dir=train_img_dir, img_size=img_size)
    val_dataset = COCODataset(ann_file=val_ann_file, img_dir=val_img_dir, img_size=img_size)

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