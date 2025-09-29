import os
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou_loss
import torchmetrics
from PIL import Image
import torchvision.transforms as T

from models.vit_detection_pretrained import VisionTransformerDetection
from utils import set_seed, get_device, save_classes

# -----------------------------
# 0. 하이퍼파라미터
# -----------------------------
class Hyperparameters:
    annotations_file = "/home/ubuntu/workspace/ai_project/data/default.json"
    train_dir = "/home/ubuntu/workspace/ai_project/data/train"
    val_dir = "/home/ubuntu/workspace/ai_project/data/val"

    epochs = 200
    batch_size = 16
    lr = 1e-4
    warmup_epochs = 5
    weight_decay = 0.01
    num_queries = 100
    img_size = 224

    weight_dict = {
        "loss_cls": 5.0,
        "loss_bbox": 2.0,
        "loss_giou": 2.0
    }

# -----------------------------
# 1. Box 변환
# -----------------------------
def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cxy = boxes[..., :2]
    wh = boxes[..., 2:]
    half = wh / 2.0
    x1y1 = cxy - half
    x2y2 = cxy + half
    return torch.cat([x1y1, x2y2], dim=-1)

# -----------------------------
# 2. Hungarian Matcher
# -----------------------------
class HungarianMatcher(torch.nn.Module):
    def __init__(self, class_weight, bbox_weight, giou_weight):
        super().__init__()
        self.class_weight = float(class_weight)
        self.bbox_weight = float(bbox_weight)
        self.giou_weight = float(giou_weight)

    @torch.no_grad()
    def forward(self, pred_cls, pred_bbox, gt_labels, gt_bboxes):
        Q = pred_cls.size(0)
        M = gt_labels.size(0)
        if M == 0 or Q == 0:
            return (torch.empty(0, dtype=torch.long, device=pred_cls.device),
                    torch.empty(0, dtype=torch.long, device=pred_cls.device))
        out_prob = pred_cls.softmax(-1)
        cost_class = -out_prob[:, gt_labels]
        l1_cost = torch.cdist(pred_bbox, gt_bboxes, p=1)
        pred_xyxy = cxcywh_to_xyxy(pred_bbox).unsqueeze(1)
        tgt_xyxy = cxcywh_to_xyxy(gt_bboxes).unsqueeze(0)
        lt = torch.max(pred_xyxy[..., :2], tgt_xyxy[..., :2])
        rb = torch.min(pred_xyxy[..., 2:], tgt_xyxy[..., 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        area_p = (pred_xyxy[..., 2] - pred_xyxy[..., 0]) * (pred_xyxy[..., 3] - pred_xyxy[..., 1])
        area_t = (tgt_xyxy[..., 2] - tgt_xyxy[..., 0]) * (tgt_xyxy[..., 3] - tgt_xyxy[..., 1])
        union = area_p + area_t - inter + 1e-6
        lt_enclosing = torch.min(pred_xyxy[..., :2], tgt_xyxy[..., :2])
        rb_enclosing = torch.max(pred_xyxy[..., 2:], tgt_xyxy[..., 2:])
        wh_enclosing = (rb_enclosing - lt_enclosing).clamp(min=0)
        area_enclosing = wh_enclosing[..., 0] * wh_enclosing[..., 1] + 1e-6
        giou_cost = 1.0 - (inter / union) + (area_enclosing - union) / area_enclosing
        cost_matrix = (
            self.class_weight * cost_class +
            self.bbox_weight * l1_cost +
            self.giou_weight * giou_cost
        ).cpu()
        q_ind, t_ind = linear_sum_assignment(cost_matrix)
        return (torch.as_tensor(q_ind, dtype=torch.long, device=pred_cls.device),
                torch.as_tensor(t_ind, dtype=torch.long, device=pred_cls.device))

# -----------------------------
# 3. Criterion
# -----------------------------
class SetCriterion(torch.nn.Module):
    def __init__(self, num_classes, matcher, weight_dict):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict

    def forward(self, outputs, targets):
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        B, Q, _ = pred_logits.shape
        device = pred_logits.device
        total_loss_cls, total_loss_bbox, total_loss_giou = 0.0, 0.0, 0.0
        total_correct, total_matched = 0, 0
        num_boxes = sum(t["labels"].numel() for t in targets)
        num_boxes = max(num_boxes, 1)
        for i in range(B):
            logits_i = pred_logits[i]
            boxes_i = pred_boxes[i]
            labels_t = targets[i]["labels"]
            boxes_t = targets[i]["boxes"]
            target_classes = torch.full((Q,), self.num_classes, device=device, dtype=torch.long)
            if labels_t.numel() > 0:
                q_idx, t_idx = self.matcher(logits_i, boxes_i, labels_t, boxes_t)
                target_classes[q_idx] = labels_t[t_idx]
                matched_pred = boxes_i[q_idx]
                matched_tgt = boxes_t[t_idx]
                loss_bbox_i = F.l1_loss(matched_pred, matched_tgt, reduction="sum")
                pred_xyxy = cxcywh_to_xyxy(matched_pred)
                tgt_xyxy = cxcywh_to_xyxy(matched_tgt)
                loss_giou_i = generalized_box_iou_loss(pred_xyxy, tgt_xyxy, reduction="sum")
                matched_pred_cls = logits_i[q_idx].argmax(-1)
                total_correct += (matched_pred_cls == labels_t[t_idx]).sum().item()
                total_matched += matched_pred_cls.numel()
            else:
                loss_bbox_i, loss_giou_i = 0.0, 0.0
            loss_cls_i = F.cross_entropy(logits_i, target_classes, reduction="mean")
            total_loss_cls += loss_cls_i
            total_loss_bbox += loss_bbox_i
            total_loss_giou += loss_giou_i
        loss_cls = total_loss_cls / B
        loss_bbox = total_loss_bbox / num_boxes
        loss_giou = total_loss_giou / num_boxes
        loss = (
            self.weight_dict["loss_cls"] * loss_cls +
            self.weight_dict["loss_bbox"] * loss_bbox +
            self.weight_dict["loss_giou"] * loss_giou
        )
        return loss, total_correct, total_matched

# -----------------------------
# 4. Datomaru 데이터셋 로더 (수정)
# -----------------------------
class DatomaruDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_list, train_dir, val_dir, classes, img_size):
        self.annotation_list = annotation_list
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.classes = {c: i for i, c in enumerate(classes)}
        self.img_size = img_size
        self.transform = T.Compose([
            T.Resize((img_size, img_size)), # <- 이 부분이 새로 추가/수정되었습니다.
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.annotation_list)

    def __getitem__(self, idx):
        item = self.annotation_list[idx]
        file_name = os.path.basename(item['image']['path'])
        
        # 파일이 train 또는 val 디렉터리 중 어디에 있는지 확인
        img_path = None
        if os.path.exists(os.path.join(self.train_dir, file_name)):
            img_path = os.path.join(self.train_dir, file_name)
        elif os.path.exists(os.path.join(self.val_dir, file_name)):
            img_path = os.path.join(self.val_dir, file_name)

        if img_path is None:
            print(f"이미지 파일을 찾을 수 없습니다: {file_name}")
            return None, None

        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"이미지 파일을 찾을 수 없습니다: {img_path}")
            return None, None

        # 원본 이미지 크기
        w, h = img.size
        # 변환된 이미지 텐서
        img_tensor = self.transform(img)

        # 바운딩 박스 정규화는 원본 이미지 크기를 기준으로 해야 합니다.
        labels = []
        bboxes = []
        for a in item.get('annotations', []):
            if 'bbox' not in a:
                continue
            
            bbox = a['bbox']
            
            label_name = None
            for key, value in a.get('attributes', {}).items():
                if value is True:
                    label_name = key
                    break
            
            if label_name and label_name in self.classes:
                labels.append(self.classes[label_name])
                
                # xyxy -> cxcywh 및 정규화 (원본 크기 기준)
                x1, y1, width, height = bbox
                cx = (x1 + width / 2) / w
                cy = (y1 + height / 2) / h
                norm_w = width / w
                norm_h = height / h
                bboxes.append([cx, cy, norm_w, norm_h])
        
        labels = torch.tensor(labels, dtype=torch.long)
        bboxes = torch.tensor(bboxes, dtype=torch.float32)

        targets = {
            "labels": labels,
            "boxes": bboxes
        }
        
        return img_tensor, targets

def get_datomaru_dataloaders(ann_file, train_dir, val_dir, img_size, batch_size):
    with open(ann_file, 'r') as f:
        data = json.load(f)

    raw_labels = data.get('categories', {}).get('label', {}).get('labels', [])
    classes = [item['name'] for item in raw_labels if item['name'] not in ['Clothes', 'Mask']]
    classes.sort()
    classes = ["__background__"] + classes
    
    all_items = data.get('items', [])
    
    # train과 val로 수동 분리
    train_annotations = []
    val_annotations = []
    
    for item in all_items:
        file_name = os.path.basename(item['image']['path'])
        if os.path.exists(os.path.join(train_dir, file_name)):
            train_annotations.append(item)
        elif os.path.exists(os.path.join(val_dir, file_name)):
            val_annotations.append(item)
            
    train_dataset = DatomaruDataset(train_annotations, train_dir, val_dir, classes, img_size)
    val_dataset = DatomaruDataset(val_annotations, train_dir, val_dir, classes, img_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*[item for item in x if item is not None])), num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*[item for item in x if item is not None])), num_workers=4
    )

    return train_loader, val_loader, classes, data['categories']

# -----------------------------
# 5. 학습 루프
# -----------------------------
def main():
    hps = Hyperparameters()
    set_seed(42)
    device = get_device()

    train_loader, val_loader, classes, _ = get_datomaru_dataloaders(
        hps.annotations_file, hps.train_dir,
        hps.val_dir, img_size=hps.img_size, batch_size=hps.batch_size
    )

    num_classes = len(classes)
    save_classes(classes)

    model = VisionTransformerDetection(num_classes=num_classes, num_queries=hps.num_queries).to(device)

    matcher = HungarianMatcher(
        class_weight=hps.weight_dict["loss_cls"],
        bbox_weight=hps.weight_dict["loss_bbox"],
        giou_weight=hps.weight_dict["loss_giou"],
    )
    criterion = SetCriterion(num_classes=num_classes, matcher=matcher, weight_dict=hps.weight_dict)

    optimizer = AdamW(model.parameters(), lr=hps.lr, weight_decay=hps.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=hps.epochs)
    scaler = GradScaler(enabled=True)

    best_map, best_acc = 0.0, 0.0
    map_metric = torchmetrics.detection.MeanAveragePrecision(box_format="cxcywh", class_metrics=False)

    for epoch in range(hps.epochs):
        if epoch < hps.warmup_epochs:
            warmup_lr = hps.lr * (epoch + 1) / hps.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # --------- Train ---------
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{hps.epochs}")
        total_correct_preds, total_matched_preds = 0, 0
        running_loss, steps = 0.0, 0

        for images, targets in pbar:
            optimizer.zero_grad(set_to_none=True)
            if not images: continue
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with autocast(enabled=True):
                outputs = model(images)
                loss, correct_preds, matched_preds = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            steps += 1
            total_correct_preds += correct_preds
            total_matched_preds += matched_preds
            acc = total_correct_preds / (total_matched_preds if total_matched_preds > 0 else 1)
            pbar.set_postfix(loss=f"{running_loss/max(1,steps):.4f}", acc=f"{acc:.2%}")

        scheduler.step()

        # --------- Validation ---------
        model.eval()
        val_loss_sum, val_steps = 0.0, 0
        total_val_correct, total_val_matched = 0, 0
        map_metric.reset()

        with torch.no_grad():
            for images, targets in val_loader:
                if not images: continue
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                with autocast(enabled=True):
                    outputs = model(images)
                    loss, correct_preds, matched_preds = criterion(outputs, targets)
                val_loss_sum += loss.item()
                val_steps += 1
                total_val_correct += correct_preds
                total_val_matched += matched_preds
                preds = []
                for plogits, pboxes in zip(outputs["pred_logits"], outputs["pred_boxes"]):
                    probs = plogits.softmax(-1)[:, :-1]
                    scores, labels = probs.max(-1)
                    preds.append({
                        "boxes": pboxes.detach().cpu(),
                        "scores": scores.detach().cpu(),
                        "labels": labels.detach().cpu()
                    })
                targets_cpu = [{k: v.detach().cpu() for k, v in t.items()} for t in targets]
                map_metric.update(preds, targets_cpu)
        avg_val_loss = val_loss_sum / max(1, val_steps)
        avg_val_accuracy = total_val_correct / (total_val_matched if total_val_matched > 0 else 1)
        metrics = map_metric.compute()

        print("-" * 50)
        print(f"✅ Epoch {epoch+1} 완료")
        print(f"    검증 손실: {avg_val_loss:.4f}")
        print(f"    검증 정확도: {avg_val_accuracy:.2%}")
        print(f"    mAP: {metrics['map']:.4f}, mAP50: {metrics['map_50']:.4f}")
        print("-" * 50)

        if avg_val_accuracy > best_acc:
            best_acc = avg_val_accuracy
            torch.save(model.state_dict(), "vit_det_best_acc_multi.pth")
            print(f"📌 Epoch {epoch+1}: 최고 accuracy 갱신 -> 모델 저장")

        if metrics["map"] > best_map:
            best_map = metrics["map"]
            torch.save(model.state_dict(), "vit_det_best_map_multi.pth")
            print(f"📌 Epoch {epoch+1}: 최고 mAP 갱신 -> 모델 저장")

    print("✅ 학습 완료")
    print(f"최고 mAP: {best_map:.4f}")
    print(f"최고 정확도: {best_acc:.2%}")


if __name__ == "__main__":
    main()