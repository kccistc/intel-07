# test_torch_model.py
# python3.8에서 PyTorch 모델 로드 + GPU 추론 테스트 스크립트
# 사용법: python3.8 test_torch_model.py

import time
import traceback
import torch
import torch.nn as nn

def try_torch_info():
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda devices:", torch.cuda.device_count())
        try:
            print("current device:", torch.cuda.current_device(), "-", torch.cuda.get_device_name(0))
        except Exception:
            pass

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def load_with_torchvision():
    try:
        import torchvision.models as models
        print("[LOAD] torchvision available - building resnet18(pretrained=True)")
        model = models.resnet18(pretrained=True)
        return model
    except Exception as e:
        raise

def load_with_hub():
    # torchvision version in hub may be adjusted; this will download weights (internet).
    print("[LOAD] trying torch.hub.load('pytorch/vision:v0.13.0', 'resnet18', pretrained=True)")
    model = torch.hub.load('pytorch/vision:v0.13.0', 'resnet18', pretrained=True)
    return model

def build_tiny_model():
    print("[LOAD] falling back to tiny model (no download required)")
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(32, 10)
    )
    return model

def main():
    print("=== PyTorch model load + inference test ===")
    try_torch_info()
    # try loading model in order: torchvision -> hub -> tiny
    model = None
    load_method = None
    try:
        model = load_with_torchvision()
        load_method = "torchvision"
    except Exception as e:
        print("[WARN] torchvision load failed:", e)
        # show small traceback for debugging
        # traceback.print_exc()
        try:
            model = load_with_hub()
            load_method = "torch.hub"
        except Exception as e2:
            print("[WARN] torch.hub load failed:", e2)
            # traceback.print_exc()
            model = build_tiny_model()
            load_method = "tiny"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    n_params = count_params(model)
    print(f"[INFO] model loaded by: {load_method}, params: {n_params:,}, device -> {device}")

    # prepare dummy input consistent with typical models (resnet: 1x3x224x224)
    bs = 1
    if hasattr(model, "fc") or load_method in ("torchvision", "torch.hub"):
        # likely resnet-like
        inp = torch.randn(bs, 3, 224, 224, device=device)
    else:
        # tiny model expects 3x? assume 64x64
        inp = torch.randn(bs, 3, 64, 64, device=device)

    # warmup (esp. for CUDA)
    with torch.no_grad():
        for _ in range(3):
            _ = model(inp)

    # timed runs
    runs = 10
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    with torch.no_grad():
        for i in range(runs):
            out = model(inp)
            if device.type == "cuda":
                # synchronize to get accurate timing
                torch.cuda.synchronize()
    t1 = time.time()
    avg_ms = (t1 - t0) / runs * 1000.0
    print(f"[RESULT] avg latency over {runs} runs: {avg_ms:.2f} ms")

    # print output shape
    try:
        if isinstance(out, torch.Tensor):
            print("output shape:", tuple(out.shape))
        else:
            print("output type:", type(out))
    except Exception:
        pass

    print("=== done ===")

if __name__ == "__main__":
    main()
