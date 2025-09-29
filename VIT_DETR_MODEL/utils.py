import torch
import json
import random
import numpy as np
from torch.utils.data import DataLoader

def evaluate(model, val_loader, device):
    """
    객체 감지 모델을 평가합니다. mAP (mean Average Precision)를 계산하기 위한
    예측과 정답 데이터를 수집하는 역할을 합니다.
    """
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            
            all_preds.append(outputs)
            all_targets.append(targets)

    print("✅ mAP 계산을 위한 평가 데이터 수집 완료. `torchmetrics`와 같은 라이브러리를 사용하여 mAP를 계산하세요.")
    
def save_classes(classes, path="classes.json"):
    """
    클래스 목록을 JSON 파일로 저장합니다.
    """
    with open(path, "w") as f:
        json.dump(classes, f, indent=2)

def load_classes(path="classes.json"):
    """
    JSON 파일에서 클래스 목록을 불러옵니다.
    """
    with open(path, "r") as f:
        return json.load(f)

def set_seed(seed=42):
    """
    재현성을 위해 난수 시드를 설정합니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """
    CUDA 사용 가능 여부를 확인하고 장치를 반환합니다.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")