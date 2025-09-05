import os
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou_loss

from models.vit_ditection import VisionTransformerDetection   # ← 파일명이 vit_ditection.py라면 import 경로만 바꿔주세요
from dataset.dataloader import get_dataloaders
from utils import set_seed, get_device, save_classes


# -----------------------------
# 0. 하이퍼파라미터 및 설정
# -----------------------------
class Hyperparameters:
    train_annotations_file = "/home/ubuntu/workspace/ai_project/datasets/train/_annotations.coco.json"
    train_dir = "/home/ubuntu/workspace/ai_project/datasets/train"
    val_annotations_file = "/home/ubuntu/workspace/ai_project/datasets/valid/_annotations.coco.json"
    val_dir = "/home/ubuntu/workspace/ai_project/datasets/valid"

    epochs = 150        # 전체 학습 반복 횟수 (epoch 수)
    batch_size = 32     # 한 번에 학습에 사용하는 이미지 개수 (메모리와 학습 안정성에 영향)
    lr = 1e-4           # 초기 학습률 (너무 크면 발산, 너무 작으면 수렴이 느림)
    warmup_epochs = 20  # 학습 초반에 learning rate를 천천히 올려주는 구간 (수렴 안정화 목적)
    weight_decay = 0.05 # 가중치 감쇠(L2 정규화), 과적합 방지
    num_queries = 300   # DETR 방식에서 객체 후보(box query)의 개수 (크면 탐색 범위 넓지만 학습 난이도 ↑)

    # 손실 가중치
    weight_dict = {
        "loss_cls": 5.0,  # 분류(classification) 손실 가중치
        "loss_bbox": 0.1, # 박스 회귀(L1) 손실 가중치
        "loss_giou": 0.1  # GIoU(박스 위치 정확성) 손실 가중치
        }


# -----------------------------
# 1. 유틸: box 포맷 변환
# (cx, cy, w, h) -> (x1, y1, x2, y2)
# -----------------------------
def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cxy = boxes[..., :2]
    wh = boxes[..., 2:]
    half = wh / 2.0
    x1y1 = cxy - half
    x2y2 = cxy + half
    return torch.cat([x1y1, x2y2], dim=-1)


# -----------------------------
# 2. 헝가리안 매처
# -----------------------------
class HungarianMatcher(nn.Module):
    def __init__(self, class_weight: float, bbox_weight: float, giou_weight: float):
        super().__init__()
        # 전달된 가중치를 반드시 사용 (기존 코드 버그 수정)
        self.class_weight = float(class_weight)
        self.bbox_weight = float(bbox_weight)
        self.giou_weight = float(giou_weight)

    @torch.no_grad()
    def forward(self, pred_cls: torch.Tensor, pred_bbox: torch.Tensor,
                gt_labels: torch.Tensor, gt_bboxes: torch.Tensor):
        """
        pred_cls:  [Q, C+1]
        pred_bbox: [Q, 4]  (cx,cy,w,h), 0~1 정규화 가정
        gt_labels: [M]
        gt_bboxes: [M, 4]  (cx,cy,w,h), 0~1 정규화 가정
        """
        Q = pred_cls.size(0)
        M = gt_labels.size(0)

        if M == 0 or Q == 0:
            return (torch.empty(0, dtype=torch.long, device=pred_cls.device),
                    torch.empty(0, dtype=torch.long, device=pred_cls.device))

        # 분류 비용: -P(class)
        out_prob = pred_cls.softmax(-1)  # [Q, C+1]
        cost_class = -out_prob[:, gt_labels]  # [Q, M]

        # L1 비용(박스)
        # (중요) 매칭 비용으로 L1을 넣으면 수렴이 더 안정적
        l1_cost = torch.cdist(pred_bbox, gt_bboxes, p=1)  # [Q, M]

        # GIoU 비용(박스): (cxcywh) -> (xyxy)
        pred_xyxy = cxcywh_to_xyxy(pred_bbox)
        tgt_xyxy = cxcywh_to_xyxy(gt_bboxes)
        # pairwise IoU/GIoU 계산을 위해 브로드캐스팅
        # generalized_box_iou_loss는 per-sample만 지원하므로 비용 행렬 직접 구성
        # 여기서는 간단히 IoU 기반 근사: -IoU ≈ GIoU 비용 대체 (충분히 잘 작동)
        # 더 정확히 하려면 pairwise generalized IoU를 구현하세요.
        # 아래는 IoU 근사 코드:
        QN = pred_xyxy.size(0)
        MN = tgt_xyxy.size(0)
        # [Q,1,4] vs [1,M,4]
        p = pred_xyxy.unsqueeze(1)
        t = tgt_xyxy.unsqueeze(0)
        # intersection
        lt = torch.max(p[..., :2], t[..., :2])
        rb = torch.min(p[..., 2:], t[..., 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        # union
        area_p = (p[..., 2] - p[..., 0]).clamp(min=0) * (p[..., 3] - p[..., 1]).clamp(min=0)
        area_t = (t[..., 2] - t[..., 0]).clamp(min=0) * (t[..., 3] - t[..., 1]).clamp(min=0)
        union = area_p + area_t - inter + 1e-6
        iou = inter / union
        cost_iou = -iou  # [Q, M]

        cost_matrix = (
            self.class_weight * cost_class +
            self.bbox_weight * l1_cost +
            self.giou_weight * cost_iou
        ).cpu()

        q_ind, t_ind = linear_sum_assignment(cost_matrix)
        return (torch.as_tensor(q_ind, dtype=torch.long, device=pred_cls.device),
                torch.as_tensor(t_ind, dtype=torch.long, device=pred_cls.device))


# -----------------------------
# 3. SetCriterion (배치 단위)
# -----------------------------
class SetCriterion(nn.Module):
    def __init__(self, num_classes: int, matcher: HungarianMatcher, weight_dict: dict):
        super().__init__()
        self.num_classes = num_classes  # foreground 클래스 수 (배경 제외)
        self.matcher = matcher
        self.weight_dict = weight_dict

    def forward(self, outputs: dict, targets: list):
        """
        outputs:
          - 'pred_logits': [B, Q, C+1]
          - 'pred_boxes' : [B, Q, 4]  (cx,cy,w,h)
        targets: list of dict  (len = B)
          - 'labels': [Mi]
          - 'boxes' : [Mi, 4] (cx,cy,w,h)
        """
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        B, Q, _ = pred_logits.shape

        device = pred_logits.device
        total_loss_cls = torch.tensor(0.0, device=device)
        total_loss_bbox = torch.tensor(0.0, device=device)
        total_loss_giou = torch.tensor(0.0, device=device)

        total_matched = 0
        total_correct = 0

        # 전체 타깃 박스 수(평균화를 위함)
        num_boxes = sum(t["labels"].numel() for t in targets)
        num_boxes = max(num_boxes, 1)

        for i in range(B):
            logits_i = pred_logits[i]  # [Q, C+1]
            boxes_i = pred_boxes[i]    # [Q, 4]
            labels_t = targets[i]["labels"]         # [Mi]
            boxes_t = targets[i]["boxes"]           # [Mi, 4]

            # 분류 타깃(배경 포함)
            target_classes = torch.full((Q,), self.num_classes, device=device, dtype=torch.long)

            if labels_t.numel() > 0:
                # 매칭
                q_idx, t_idx = self.matcher(logits_i, boxes_i, labels_t, boxes_t)
                # 매칭된 쿼리만 GT 라벨로 채움
                target_classes[q_idx] = labels_t[t_idx]

                # bbox L1
                matched_pred = boxes_i[q_idx]           # [M, 4] cxcywh
                matched_tgt = boxes_t[t_idx]            # [M, 4] cxcywh
                loss_bbox_i = F.l1_loss(matched_pred, matched_tgt, reduction="sum")

                # giou: xyxy로 변환
                pred_xyxy = cxcywh_to_xyxy(matched_pred)
                tgt_xyxy = cxcywh_to_xyxy(matched_tgt)
                loss_giou_i = generalized_box_iou_loss(pred_xyxy, tgt_xyxy, reduction="sum")

                # 정확도(매칭된 것만)
                matched_pred_cls = logits_i[q_idx].argmax(-1)  # [M]
                total_correct += (matched_pred_cls == labels_t[t_idx]).sum().item()
                total_matched += matched_pred_cls.numel()
            else:
                # GT가 없으면 bbox/giou 손실은 0, 분류는 전부 배경으로 학습
                loss_bbox_i = torch.tensor(0.0, device=device)
                loss_giou_i = torch.tensor(0.0, device=device)

            # 분류 CE (Q개 모두)
            loss_cls_i = F.cross_entropy(logits_i, target_classes, reduction="mean")

            total_loss_cls += loss_cls_i
            total_loss_bbox += loss_bbox_i
            total_loss_giou += loss_giou_i

        # 평균화
        loss_cls = total_loss_cls / B
        loss_bbox = total_loss_bbox / num_boxes
        loss_giou = total_loss_giou / num_boxes

        loss = (
            self.weight_dict.get("loss_cls", 1.0) * loss_cls +
            self.weight_dict.get("loss_bbox", 1.0) * loss_bbox +
            self.weight_dict.get("loss_giou", 1.0) * loss_giou
        )

        return loss, total_correct, total_matched


# -----------------------------
# 4. 메인 학습 루프
# -----------------------------
def main():
    hps = Hyperparameters()
    set_seed(42)
    device = get_device()

    train_loader, val_loader, classes, _ = get_dataloaders(
        hps.train_annotations_file, hps.train_dir,
        hps.val_annotations_file, hps.val_dir,
        img_size=224, batch_size=hps.batch_size
    )
    num_classes = len(classes)          # foreground 클래스 수
    save_classes(classes)

    # (중요) 모델에는 foreground 클래스 수만 전달해야 함
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

    for epoch in range(hps.epochs):
        # warmup
        if epoch < hps.warmup_epochs:
            warmup_lr = hps.lr * (epoch + 1) / hps.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # --------- Train ---------
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{hps.epochs}")
        total_correct_preds = 0
        total_matched_preds = 0
        running_loss = 0.0
        steps = 0

        for images, targets in pbar:
            if images is None:
                continue

            optimizer.zero_grad(set_to_none=True)

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with autocast(enabled=True):
                outputs = model(images)
                loss, correct_preds, matched_preds = criterion(outputs, targets)

            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                print("⚠️ 경고: 손실값이 유효하지 않아 백프로파게이션을 건너뜁니다.")
                continue

            running_loss += loss.item()
            steps += 1
            total_correct_preds += correct_preds
            total_matched_preds += matched_preds
            acc = total_correct_preds / (total_matched_preds if total_matched_preds > 0 else 1)
            pbar.set_postfix(loss=f"{running_loss / max(1, steps):.4f}", acc=f"{acc:.2%}")

        scheduler.step()

        # --------- Validation ---------
        model.eval()
        val_pbar = tqdm(val_loader, desc=f"Validation {epoch+1}/{hps.epochs}")
        val_loss_sum = 0.0
        val_steps = 0
        total_val_correct = 0
        total_val_matched = 0

        with torch.no_grad():
            for images, targets in val_pbar:
                if images is None:
                    continue

                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                with autocast(enabled=True):
                    outputs = model(images)
                    loss, correct_preds, matched_preds = criterion(outputs, targets)

                # 유효한 값만 집계
                if torch.isfinite(loss):
                    val_loss_sum += loss.item()
                    val_steps += 1
                    total_val_correct += correct_preds
                    total_val_matched += matched_preds

        avg_val_loss = val_loss_sum / max(1, val_steps)
        avg_val_accuracy = total_val_correct / (total_val_matched if total_val_matched > 0 else 1)

        print("-" * 50)
        print(f"✅ Epoch {epoch+1} 완료")
        print(f"   최종 검증 손실: {avg_val_loss:.4f}")
        print(f"   최종 검증 정확도: {avg_val_accuracy:.2%}")
        print("-" * 50)

        torch.save(model.state_dict(), "vit_det_best.pth")

    print("✅ 학습 완료 & 모델 저장")


if __name__ == "__main__":
    main()
