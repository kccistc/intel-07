'''
부위(ROI) 기반 멀티태스크 학습 + Z-Score 표준화 + Huber Loss 적용
V4 변경사항:
- 데이터 증강(Augmentation) 강화
- Backbone 학습률 추가 감소 (1e-5)
'''

import os, json, argparse, random, time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from tqdm import tqdm
from torch.amp import GradScaler, autocast

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def clamp_bbox(x, y, w, h, W, H):
    x = max(0, min(int(x), W-1)); y = max(0, min(int(y), H-1))
    w = max(1, min(int(w), W - x)); h = max(1, min(int(h), H - y))
    return x, y, w, h

FACEPART_TO_LABEL_SUBSTR = {
    "forehead":      ["forehead"],
    "glabella":      ["glabellus", "glabella"],
    "left_crowsfeet":["l_perocular", "left_crowsfeet"],
    "right_crowsfeet":["r_perocular", "right_crowsfeet"],
    "left_cheek":    ["l_cheek", "left_cheek"],
    "right_cheek":   ["r_cheek", "right_cheek"],
    "lips":          ["lip", "lips", "perioral"],
    "chin":          ["chin"],
    "full":          ["full", "all", "face"],
}

# =================================================================================
# need_data 파일을 기반으로 생성된 학습 대상 정의
# =================================================================================
CUSTOM_LABEL_SPACE_CONFIG = {
    "forehead": {
        "regression": [
            "eq_forehead_moisture",
            "eq_forehead_elasticity_Q0",
        ],
        "classification": [
            "ann_forehead_pigmentation"
        ]
    },
    "left_cheek": {
        "regression": [
            "eq_l_cheek_pore",
            "eq_l_cheek_moisture",
            "eq_l_cheek_elasticity_Q0",
        ],
        "classification": [
            "ann_l_cheek_pigmentation"
        ]
    },
    "right_cheek": {
        "regression": [
            "eq_r_cheek_pore",
            "eq_r_cheek_moisture",
            "eq_r_cheek_elasticity_Q0",
        ],
        "classification": [
            "ann_r_cheek_pigmentation"
        ]
    },
    "lips": {
        "regression": [],
        "classification": [
            "ann_lip_dryness"
        ]
    },
    "chin": {
        "regression": [
            "eq_chin_moisture",
            "eq_chin_elasticity_Q0",
        ],
        "classification": []
    },
}
# =================================================================================

# 학습/검증 데이터셋의 JSON 주석 파일을 기반으로 부위별 레이블 공간 구축
def build_roi_label_space(train_json_path, custom_config):
    roi_label_space = {"regression_keys": {}, "class_keys": {}, "class_key_to_index": {}}
    for roi, config in custom_config.items():
        roi_label_space["regression_keys"][roi] = sorted(config.get("regression", []))
        roi_label_space["class_keys"][roi] = sorted(config.get("classification", []))

    coco = load_json(train_json_path)
    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    
    roi_class_vals = defaultdict(lambda: defaultdict(set))
    for a in coco["annotations"]:
        if "value" not in a:
            continue
        name = id2name.get(a["category_id"])
        if name is None: continue

        is_target_class = False
        for roi, C_keys in roi_label_space["class_keys"].items():
            if name in C_keys:
                is_target_class = True
                break
        if not is_target_class:
            continue

        roi_hits = []
        for roi, subs in FACEPART_TO_LABEL_SUBSTR.items():
            if any(s in name for s in subs):
                roi_hits.append(roi)
        
        try:
            iv = int(round(float(a["value"])))
            for r in roi_hits:
                if name in roi_label_space["class_keys"].get(r, []):
                    roi_class_vals[r][name].add(iv)
        except:
            pass

    for roi, C_keys in roi_label_space["class_keys"].items():
        roi_label_space["class_key_to_index"].setdefault(roi, {})
        for ck in C_keys:
            vals = sorted(list(roi_class_vals[roi][ck]))
            if not vals: continue
            v2i = {v:i for i,v in enumerate(vals)}
            i2v = {i:v for v,i in v2i.items()}
            roi_label_space["class_key_to_index"][roi][ck] = {
                "values": vals, "val_to_idx": v2i, "idx_to_val": i2v
            }
            
    return roi_label_space

# roi_label_space.json 파일 생성
def compute_reg_std_from_train(train_json_path):
    """eq_* 키별 mean/std 산출 (z-score 학습용)"""
    coco = load_json(train_json_path)
    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    vals = defaultdict(list)
    for a in coco["annotations"]:
        if "value" not in a:
            continue
        name = id2name[a["category_id"]]
        if not name.startswith("eq_"):
            continue
        try:
            v = float(a["value"])
            vals[name].append(v)
        except:
            pass
    stats = {}
    for k, arr in vals.items():
        if len(arr) == 0:
            continue
        m = float(np.mean(arr))
        s = float(np.std(arr) + 1e-6)
        stats[k] = {"mean": m, "std": s}
    return stats

# ROI 기반 데이터셋
class RoiDataset(Dataset):
    
    def __init__(self, json_path, data_root, roi_label_space, img_size=384, is_train=True, pad_ratio=0.10, reg_std=None):
        super().__init__()
        self.coco = load_json(json_path)
        self.data_root = Path(data_root)
        self.id2img = {im["id"]: im for im in self.coco["images"]}
        self.id2name = {c["id"]: c["name"] for c in self.coco["categories"]}
        
        self.roi_reg_keys = roi_label_space["regression_keys"]
        self.roi_cls_keys = roi_label_space["class_keys"]
        self.roi_cls_maps = roi_label_space["class_key_to_index"]
        self.pad_ratio = pad_ratio
        self.reg_std = reg_std or {}

        self.anns_by_img = defaultdict(list)
        for a in self.coco["annotations"]:
            self.anns_by_img[a["image_id"]].append(a)

        feature_names_by_img = defaultdict(set)
        for img_id, anns in self.anns_by_img.items():
            for a in anns:
                if "value" in a:
                    cat_id = a.get("category_id")
                    if cat_id in self.id2name:
                        feature_names_by_img[img_id].add(self.id2name[cat_id])

        self.samples = []
        for img_id, im in self.id2img.items():
            W, H = im.get("width"), im.get("height")
            if W is None or H is None:
                continue

            facepart_anns = [a for a in self.anns_by_img.get(img_id, []) if self.id2name.get(a.get("category_id")) and self.id2name[a["category_id"]].startswith("facepart::")]

            for a in facepart_anns:
                cname = self.id2name[a["category_id"]]
                roi_name = cname.split("facepart::", 1)[1]

                target_reg_keys = self.roi_reg_keys.get(roi_name, [])
                target_cls_keys = self.roi_cls_keys.get(roi_name, [])
                if not target_reg_keys and not target_cls_keys:
                    continue

                image_features = feature_names_by_img.get(img_id, set())
                has_label = any(key in image_features for key in target_reg_keys) or \
                            any(key in image_features for key in target_cls_keys)

                if not has_label:
                    continue

                bbox = a.get("bbox", [])
                if not bbox or len(bbox) != 4:
                    continue
                x, y, w, h = bbox
                px, py = int(w * self.pad_ratio), int(h * self.pad_ratio)
                x, y, w, h = clamp_bbox(x - px, y - py, w + 2 * px, h + 2 * py, W, H)
                self.samples.append((img_id, roi_name, [x, y, w, h]))

        if is_train:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=15),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                transforms.RandomErasing(p=0.75, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, roi, bbox = self.samples[idx]
        im = self.id2img[img_id]
        img_path = self.data_root / im["file_name"]
        with Image.open(img_path).convert("RGB") as pil:
            x0,y0,w,h = bbox
            crop = pil.crop((x0, y0, x0+w, y0+h))
            x = self.tf(crop)

        R_keys = self.roi_reg_keys.get(roi, [])
        C_keys = self.roi_cls_keys.get(roi, [])
        reg_t = torch.full((len(R_keys),), float('nan'), dtype=torch.float32)
        cls_t = torch.full((len(C_keys),), -1, dtype=torch.long)

        for a in self.anns_by_img.get(img_id, []):
            if "value" not in a:
                continue
            name = self.id2name[a["category_id"]]
            v = a["value"]
            if name in R_keys:
                j = R_keys.index(name)
                try:
                    fv = float(v)
                    if name in self.reg_std:
                        m, s = self.reg_std[name]["mean"], self.reg_std[name]["std"]
                        reg_t[j] = (fv - m) / s
                    else: 
                        reg_t[j] = fv
                except:
                    pass
            if name in C_keys:
                try:
                    iv = int(round(float(v)))
                    maps = self.roi_cls_maps.get(roi, {}).get(name, None)
                    if maps and iv in maps["val_to_idx"]:
                        cls_t[C_keys.index(name)] = maps["val_to_idx"][iv]
                except:
                    pass

        meta = {"img_id": img_id, "roi": roi}
        return x, reg_t, cls_t, meta

# 배치 콜레이트 함수
def roi_collate(batch):
    xs, regs, clss, metas = zip(*batch)
    xs = torch.stack(xs, dim=0)

    reg_lens = [r.numel() for r in regs]
    cls_lens = [c.numel() for c in clss]
    max_R = max(reg_lens) if reg_lens else 0
    max_C = max(cls_lens) if cls_lens else 0

    if max_R > 0:
        reg_pad = torch.full((len(regs), max_R), float('nan'), dtype=torch.float32)
        for i, r in enumerate(regs):
            if r.numel() > 0:
                reg_pad[i, :r.numel()] = r
    else:
        reg_pad = torch.empty((len(regs), 0), dtype=torch.float32)

    if max_C > 0:
        cls_pad = torch.full((len(clss), max_C), -1, dtype=torch.long)
        for i, c in enumerate(clss):
            if c.numel() > 0:
                cls_pad[i, :c.numel()] = c
    else:
        cls_pad = torch.empty((len(clss), 0), dtype=torch.long)

    return xs, reg_pad, cls_pad, metas

# 부위(ROI) 기반 멀티헤드 모델
class RoiMultiHead(nn.Module):
    def __init__(self, backbone="resnet50", roi_label_space=None, pretrained=True, drop_rate=0.0, drop_path_rate=0.0, head_dropout=0.0):
        super().__init__()
        assert roi_label_space is not None
        self.roi_reg_keys = roi_label_space["regression_keys"]
        self.roi_cls_keys = roi_label_space["class_keys"]
        self.roi_cls_maps = roi_label_space["class_key_to_index"]

        if backbone == "resnet50":
            m = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
            in_dim = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
        else:
            self.backbone = timm.create_model(
                backbone, pretrained=pretrained, num_classes=0,
                drop_rate=drop_rate, drop_path_rate=drop_path_rate
            )
            in_dim = self.backbone.num_features

        self.head_dropout = nn.Dropout(head_dropout)
        self.reg_heads = nn.ModuleDict()
        self.cls_heads = nn.ModuleDict()

        for roi, R_keys in self.roi_reg_keys.items():
            if len(R_keys) > 0:
                self.reg_heads[roi] = nn.Linear(in_dim, len(R_keys))
        for roi, C_keys in self.roi_cls_keys.items():
            if len(C_keys) > 0:
                heads = nn.ModuleList()
                for ck in C_keys:
                    nclass = len(self.roi_cls_maps.get(roi, {}).get(ck, {}).get("values", []))
                    heads.append(nn.Linear(in_dim, nclass))
                self.cls_heads[roi] = heads

    def forward(self, x, roi_names):
        feats = self.backbone(x)
        feats = self.head_dropout(feats)
        B = feats.size(0)
        out_reg, out_cls = [None]*B, [None]*B

        for i in range(B):
            roi = roi_names[i]
            f = feats[i:i+1]
            reg_pred = None
            if roi in self.reg_heads:
                reg_pred = self.reg_heads[roi](f)
            cls_preds = None
            if roi in self.cls_heads:
                cls_preds = [head(f) for head in self.cls_heads[roi]]
            out_reg[i], out_cls[i] = reg_pred, cls_preds

        return out_reg, out_cls

# 손실 함수 및 학습/평가 루프
def masked_huber_loss(pred, target):
    if pred is None:
        return None
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return None
    loss = F.smooth_l1_loss(pred[mask], target[mask], beta=1.0)
    return loss

# 분류 손실 및 정확도 계산
def cls_loss_and_acc(logits_list, target, ignore_index=-1, label_smoothing=0.0):
    if logits_list is None or len(logits_list) == 0 or target.numel() == 0:
        return None, 0, None, 0
    
    total_loss, total_acc, n_valid_acc = 0.0, 0.0, 0
    
    valid_targets = target[target != ignore_index]
    if len(valid_targets) == 0:
        return None, 0, None, 0

    for k, lg in enumerate(logits_list):
        t = target[k].unsqueeze(0)
        if t.item() == ignore_index:
            continue
        
        loss = F.cross_entropy(lg, t, ignore_index=ignore_index, label_smoothing=label_smoothing)
        total_loss += loss

        pred = lg.argmax(dim=1)
        acc = (pred == t).float().mean()
        total_acc += acc.item()
        n_valid_acc += 1
    
    if n_valid_acc == 0:
        return None, 0, None, 0
    
    return total_loss / n_valid_acc, n_valid_acc, total_acc / n_valid_acc, n_valid_acc

# 학습 및 평가 루프
def train_one_epoch(model, loader, optim, device, lambda_reg=1.0, lambda_cls=1.0, scaler=None, reg_std=None, label_smoothing=0.0):
    model.train()
    pbar = tqdm(loader, desc="train", dynamic_ncols=True, leave=False)
    avg = {"loss":0.0, "mae":0.0, "acc":0.0}
    steps = 0

    for x, reg_t, cls_t, metas in pbar:
        x = x.to(device, non_blocking=True)
        reg_t = reg_t.to(device)
        cls_t = cls_t.to(device)
        roi_names = [m["roi"] for m in metas]

        optim.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=(scaler is not None)):
            out_reg, out_cls = model(x, roi_names)

            total_loss = x.new_tensor(0.0)
            total_mae, n_reg = 0.0, 0
            total_acc, n_cls = 0.0, 0

            for i in range(x.size(0)):
                roi = roi_names[i]
                
                reg_pred = out_reg[i]
                if reg_pred is not None and reg_pred.size(1) > 0:
                    target_slice = reg_t[i, :reg_pred.size(1)]
                    huber_loss = masked_huber_loss(reg_pred.squeeze(0), target_slice)
                    if huber_loss is not None:
                        total_loss = total_loss + lambda_reg * huber_loss
                    
                    mask = ~torch.isnan(target_slice)
                    if mask.sum() > 0:
                        p = reg_pred.squeeze(0)[mask]
                        t = target_slice[mask]
                        R_keys = model.roi_reg_keys.get(roi, [])
                        key_indices = mask.nonzero(as_tuple=True)[0]
                        
                        for pred_val, true_val, key_idx in zip(p, t, key_indices):
                            key = R_keys[key_idx]
                            m = reg_std.get(key, {}).get("mean", 0.0)
                            s = reg_std.get(key, {}).get("std", 1.0)
                            
                            pred_original = pred_val.item() * s + m
                            true_original = true_val.item() * s + m
                            
                            total_mae += abs(pred_original - true_original)
                            n_reg += 1
                
                cls_preds = out_cls[i]
                if cls_preds is not None and len(cls_preds) > 0:
                    Clen = len(cls_preds)
                    ce, nvalid_ce, acc, nvalid_acc = cls_loss_and_acc(cls_preds, cls_t[i, :Clen], label_smoothing=label_smoothing)
                    if ce is not None and nvalid_ce > 0:
                        total_loss = total_loss + lambda_cls * ce
                    if acc is not None and nvalid_acc > 0:
                        total_acc += acc; n_cls += 1

        if total_loss.requires_grad:
            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                total_loss.backward()
                optim.step()
        else:
            print(f"\nWarning: Skipping optimizer step for a batch with no valid labels. Batch ROIs: {roi_names}")

        steps += 1
        avg["loss"] += float(total_loss.item())
        if n_reg > 0: avg["mae"] += (total_mae / n_reg)
        if n_cls > 0: avg["acc"] += (total_acc / n_cls)

        pbar.set_postfix({
            "loss": f"{avg['loss']/steps:.4f}",
            "MAE":  f"{(avg['mae']/steps):.4f}" if steps>0 else "0.0000",
            "Acc":  f"{(avg['acc']/steps):.4f}" if steps>0 else "0.0000",
        })

    return avg["loss"]/max(1,steps), avg["mae"]/max(1,steps), avg["acc"]/max(1,steps)

# 평가 루프
@torch.no_grad()
def evaluate(model, loader, device, reg_std=None):
    model.eval()
    pbar = tqdm(loader, desc="valid", dynamic_ncols=True, leave=False)
    avg = {"loss":0.0, "mae":0.0, "acc":0.0}
    steps = 0

    for x, reg_t, cls_t, metas in pbar:
        x = x.to(device, non_blocking=True)
        reg_t = reg_t.to(device)
        cls_t = cls_t.to(device)
        roi_names = [m["roi"] for m in metas]

        out_reg, out_cls = model(x, roi_names)

        total_loss = x.new_tensor(0.0)
        total_mae, n_reg = 0.0, 0
        total_acc, n_cls = 0.0, 0

        for i in range(x.size(0)):
            roi = roi_names[i]
            
            reg_pred = out_reg[i]
            if reg_pred is not None and reg_pred.size(1) > 0:
                target_slice = reg_t[i, :reg_pred.size(1)]
                huber_loss = masked_huber_loss(reg_pred.squeeze(0), target_slice)
                if huber_loss is not None:
                    total_loss = total_loss + huber_loss 

                mask = ~torch.isnan(target_slice)
                if mask.sum() > 0:
                    p = reg_pred.squeeze(0)[mask]
                    t = target_slice[mask]
                    R_keys = model.roi_reg_keys.get(roi, [])
                    key_indices = mask.nonzero(as_tuple=True)[0]
                    
                    for pred_val, true_val, key_idx in zip(p, t, key_indices):
                        key = R_keys[key_idx]
                        m = reg_std.get(key, {}).get("mean", 0.0)
                        s = reg_std.get(key, {}).get("std", 1.0)
                        
                        pred_original = pred_val.item() * s + m
                        true_original = true_val.item() * s + m
                        
                        total_mae += abs(pred_original - true_original)
                        n_reg += 1

            cls_preds = out_cls[i]
            if cls_preds is not None and len(cls_preds) > 0:
                Clen = len(cls_preds)
                ce, nvalid_ce, acc, nvalid_acc = cls_loss_and_acc(cls_preds, cls_t[i, :Clen], label_smoothing=0.0) # No smoothing for eval
                if ce is not None and nvalid_ce > 0:
                    total_loss = total_loss + ce
                if acc is not None and nvalid_acc > 0:
                    total_acc += acc; n_cls += 1

        steps += 1
        avg["loss"] += float(total_loss.item())
        if n_reg > 0: avg["mae"] += (total_mae / n_reg)
        if n_cls > 0: avg["acc"] += (total_acc / n_cls)

        pbar.set_postfix({
            "loss": f"{avg['loss']/steps:.4f}",
            "MAE":  f"{(avg['mae']/steps):.4f}" if steps>0 else "0.0000",
            "Acc":  f"{(avg['acc']/steps):.4f}" if steps>0 else "0.0000",
        })

    return avg["loss"]/max(1,steps), avg["mae"]/max(1,steps), avg["acc"]/max(1,steps)

# early stop 클래스
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.num_bad = 0

    def step(self, score):
        if self.best is None or score < self.best - self.min_delta:
            self.best = score
            self.num_bad = 0
            return True
        else:
            self.num_bad += 1
            return False

    def should_stop(self):
        return self.num_bad >= self.patience

# 메인 함수
def main():

    # 인자 파싱
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="/home/bbang/Workspace/Intel07_Intelproject_Team3/dataset")
    ap.add_argument("--train-json", type=str, default="/home/bbang/Workspace/Intel07_Intelproject_Team3/dataset/merged_training.json")
    ap.add_argument("--val-json",   type=str, default="/home/bbang/Workspace/Intel07_Intelproject_Team3/dataset/merged_validation.json")
    ap.add_argument("--test-json",  type=str, default="")
    ap.add_argument("--out-dir",    type=str, default="./runs/skin_roi_v4")
    ap.add_argument("--img-size",   type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs",     type=int, default=100)
    ap.add_argument("--workers",    type=int, default=16)
    ap.add_argument("--backbone",   type=str, default="convnext_tiny", help="timm 라이브러리 모델 이름")
    ap.add_argument("--lr-backbone",type=float, default=1e-5)
    ap.add_argument("--lr-head",    type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--drop-rate",    type=float, default=0.4, help="Backbone drop rate")
    ap.add_argument("--drop-path-rate", type=float, default=0.2, help="Backbone drop path rate")
    ap.add_argument("--head-dropout", type=float, default=0.5, help="Dropout before head layers")
    ap.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing for classification loss")
    ap.add_argument("--lambda-reg", type=float, default=1.0)
    ap.add_argument("--lambda-cls", type=float, default=1.0)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--early-patience", type=int, default=10)
    ap.add_argument("--early-min-delta", type=float, default=1e-4)
    
    ap.add_argument('--auto-reg-std', dest='auto_reg_std', action='store_true', help='train json에서 회귀 통계를 자동 산출하여 표준화 적용 (기본 활성화)')
    ap.add_argument('--no-auto-reg-std', dest='auto_reg_std', action='store_false', help='회귀 표준화 비활성화')
    ap.set_defaults(auto_reg_std=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    roi_label_space = build_roi_label_space(args.train_json, CUSTOM_LABEL_SPACE_CONFIG)
    save_json(roi_label_space, os.path.join(args.out_dir, "roi_label_space.json"))
    
    reg_std = None
    if args.auto_reg_std:
        print("[info] Calculating regression statistics from training data...")
        reg_std = compute_reg_std_from_train(args.train_json)
        save_json(reg_std, os.path.join(args.out_dir, "reg_std.json"))
        print(f"[info] Saved regression stats to {os.path.join(args.out_dir, 'reg_std.json')}")
    else:
        print("[info] Regression standardization is not used.")

    # 데이터셋 및 데이터로더
    train_set = RoiDataset(args.train_json, args.data_root, roi_label_space, img_size=args.img_size, is_train=True, reg_std=reg_std)
    val_set   = RoiDataset(args.val_json,   args.data_root, roi_label_space, img_size=args.img_size, is_train=False, reg_std=reg_std)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, collate_fn=roi_collate)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True, collate_fn=roi_collate)

    # 모델, 옵티마이저, 스케줄러, 스케일러 등
    model = RoiMultiHead(
        backbone=args.backbone,
        roi_label_space=roi_label_space,
        pretrained=True,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        head_dropout=args.head_dropout
    ).to(device)

    backbone_params, head_params = [], []
    for n,p in model.named_parameters():
        (backbone_params if "backbone" in n else head_params).append(p)
    optim = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params,     "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = GradScaler(device="cuda") if torch.cuda.is_available() else None

    stopper = EarlyStopping(patience=args.early_patience, min_delta=args.early_min_delta)
    best_path  = os.path.join(args.out_dir, "best.pth")
    best_epoch = 0

    start_time = datetime.now()
    print(f"\n[INFO] Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 학습 루프
    for ep in range(1, args.epochs+1):
        
        tr_loss, tr_mae, tr_acc = train_one_epoch(
            model, train_loader, optim, device,
            lambda_reg=args.lambda_reg, lambda_cls=args.lambda_cls,
            scaler=scaler, reg_std=reg_std, label_smoothing=args.label_smoothing
        )
        vl_loss, vl_mae, vl_acc = evaluate(model, val_loader, device, reg_std=reg_std)
        sched.step()

        score = vl_loss

        print(f"[{ep:03d}/{args.epochs}] "
              f"train: loss={tr_loss:.4f} MAE={tr_mae:.4f} Acc={tr_acc:.4f} | "
              f"valid: loss={vl_loss:.4f} MAE={vl_mae:.4f} Acc={vl_acc:.4f} | "
              f"score(vl_loss)={score:.4f}")

        improved = stopper.step(score)
        if improved:
            torch.save({"model": model.state_dict(),
                        "roi_label_space": roi_label_space,
                        "reg_std": reg_std, 
                        "epoch": ep,
                        "val_loss": vl_loss,
                        "val_mae": vl_mae,
                        "val_acc": vl_acc,
                        "score": score}, best_path)
            best_epoch = ep
            print(f"  → Saved best to {best_path}")
        elif stopper.should_stop():
            print(f"Early stopping triggered at epoch {ep} (best @ {best_epoch})")
            break

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n[INFO] Training finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Total training time: {str(duration).split('.')[0]}")

    print(f"Done. Best epoch={best_epoch}, checkpoint={best_path}")

    if args.test_json and Path(args.test_json).exists():
        test_set = RoiDataset(args.test_json, args.data_root, roi_label_space, img_size=args.img_size, is_train=False, reg_std=reg_std)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True, collate_fn=roi_collate)
        
        print(f"\nLoading best model from {best_path} for testing...")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        
        ts_loss, ts_mae, ts_acc = evaluate(model, test_loader, device, reg_std=reg_std)
        print(f"[Test] loss={ts_loss:.4f} | MAE={ts_mae:.4f} | Acc={ts_acc:.4f}")
        save_json({"test_loss": ts_loss, "test_mae": ts_mae, "test_acc": ts_acc},
                  os.path.join(args.out_dir, "test_metrics.json"))

if __name__ == "__main__":
    main()
