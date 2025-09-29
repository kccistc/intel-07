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
import torchmetrics

from models.vit_ditection import VisionTransformerDetection
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

    # 전체 학습을 반복할 횟수. 모델이 데이터를 전체적으로 몇 번 볼지를 결정합니다.
    epochs = 200
    # 한 번의 학습 스텝에 사용될 이미지의 수. 메모리 사용량과 학습 속도에 영향을 줍니다.
    batch_size = 16
    # 모델의 가중치를 업데이트할 때 적용되는 계수. 너무 크면 불안정하고, 너무 작으면 학습이 느립니다.
    lr = 1e-4
    # 학습 초반에 학습률을 점진적으로 증가시키는 기간. 학습 안정성을 높이는 데 사용됩니다.
    warmup_epochs = 20
    # L2 정규화에 사용되는 계수. 모델의 과적합(overfitting)을 방지하는 데 도움을 줍니다.
    weight_decay = 0.01
    # DETR 모델이 한 이미지에 대해 예측할 수 있는 최대 객체 쿼리(Query)의 수.
    num_queries = 100

    # 각 손실 함수에 적용될 가중치. 모델이 어떤 손실에 더 집중해야 할지 결정합니다.
    weight_dict = {
        # 분류 손실(Cross-Entropy)의 가중치.
        "loss_cls": 1.0,
        # 박스 위치(L1) 손실의 가중치.
        "loss_bbox": 1.0,
        # GIoU(Generalized IoU) 손실의 가중치. 박스 위치 정확도를 높이는 데 중요합니다.
        "loss_giou": 2.0
    }


# -----------------------------
# 1. 유틸: box 포맷 변환 (수정됨)
# -----------------------------
def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cxy = boxes[..., :2]
    wh = boxes[..., 2:]
    half = wh / 2.0
    x1y1 = cxy - half
    x2y2 = cxy + half
    return torch.cat([x1y1, x2y2], dim=-1)


# -----------------------------
# 2. 헝가리안 매처 (수정됨)
# -----------------------------
class HungarianMatcher(nn.Module):
    def __init__(self, class_weight: float, bbox_weight: float, giou_weight: float):
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

        # ⚠️ GIOU 손실을 수동으로 계산하여 IndexError를 해결합니다.
        pred_xyxy = cxcywh_to_xyxy(pred_bbox)
        tgt_xyxy = cxcywh_to_xyxy(gt_bboxes)
        
        # pairwise 비교를 위해 unsqueeze를 사용하여 브로드캐스팅 활성화
        pred_xyxy = pred_xyxy.unsqueeze(1) # 형태: [N_pred, 1, 4]
        tgt_xyxy = tgt_xyxy.unsqueeze(0)  # 형태: [1, N_gt, 4]

        # 교차 영역(intersection) 계산
        lt = torch.max(pred_xyxy[..., :2], tgt_xyxy[..., :2])
        rb = torch.min(pred_xyxy[..., 2:], tgt_xyxy[..., 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        
        # 합집합(union) 계산
        area_p = (pred_xyxy[..., 2] - pred_xyxy[..., 0]) * (pred_xyxy[..., 3] - pred_xyxy[..., 1])
        area_t = (tgt_xyxy[..., 2] - tgt_xyxy[..., 0]) * (tgt_xyxy[..., 3] - tgt_xyxy[..., 1])
        union = area_p + area_t - inter + 1e-6
        
        # GIOU를 위한 포괄 영역(enclosing box) 계산
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
# 3. SetCriterion (수정 없음)
# -----------------------------
class SetCriterion(nn.Module):
    def __init__(self, num_classes: int, matcher: HungarianMatcher, weight_dict: dict):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict

    def forward(self, outputs: dict, targets: list):
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        B, Q, _ = pred_logits.shape

        device = pred_logits.device
        total_loss_cls = torch.tensor(0.0, device=device)
        total_loss_bbox = torch.tensor(0.0, device=device)
        total_loss_giou = torch.tensor(0.0, device=device)

        total_matched = 0
        total_correct = 0
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
                loss_bbox_i = torch.tensor(0.0, device=device)
                loss_giou_i = torch.tensor(0.0, device=device)

            loss_cls_i = F.cross_entropy(logits_i, target_classes, reduction="mean")

            total_loss_cls += loss_cls_i
            total_loss_bbox += loss_bbox_i
            total_loss_giou += loss_giou_i

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
# 4. 메인 학습 루프 (수정됨)
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

    # ✅ Best checkpoint 추적
    best_acc = 0.0
    best_map = 0.0
    best_acc_epoch = 0
    best_map_epoch = 0
    map_metric = torchmetrics.detection.MeanAveragePrecision(
        box_format="cxcywh", 
        class_metrics=False
    )

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
                if images is None:
                    continue
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                with autocast(enabled=True):
                    outputs = model(images)
                    loss, correct_preds, matched_preds = criterion(outputs, targets)

                if torch.isfinite(loss):
                    val_loss_sum += loss.item()
                    val_steps += 1
                    total_val_correct += correct_preds
                    total_val_matched += matched_preds

                # ✅ mAP 업데이트
                preds = []
                for plogits, pboxes in zip(outputs["pred_logits"], outputs["pred_boxes"]):
                    probs = plogits.softmax(-1)[:, :-1]
                    scores, labels = probs.max(-1)
                    preds.append({
                        "boxes": pboxes.detach().cpu(),
                        "scores": scores.detach().cpu(),
                        "labels": labels.detach().cpu()
                    })
                
                # ⚠️ 타겟 텐서를 CPU로 옮깁니다.
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

        # ✅ Best checkpoint 저장
        if avg_val_accuracy > best_acc:
            best_acc = avg_val_accuracy
            best_acc_epoch = epoch + 1
            torch.save(model.state_dict(), "vit_det_best_acc.pth")
            print(f"📌 Epoch {epoch+1}: 최고 accuracy 갱신 {best_acc:.2%} -> 모델 저장")

        if metrics["map"] > best_map:
            best_map = metrics["map"]
            best_map_epoch = epoch + 1
            torch.save(model.state_dict(), "vit_det_best_map.pth")
            print(f"📌 Epoch {epoch+1}: 최고 mAP 갱신 {best_map:.4f} -> 모델 저장")

    print("✅ 학습 완료")
    print("-" * 50)
    print("✨ 최종 학습 결과")
    print(f"최고 mAP: {best_map:.4f} (Epoch: {best_map_epoch})")
    print(f"최고 정확도: {best_acc:.2%} (Epoch: {best_acc_epoch})")
    print("-" * 50)


if __name__ == "__main__":
    main()