#!/usr/bin/env python3.8
# pytorch_model_test.py
# Usage examples:
#  python3.8 pytorch_model_test.py --model resnet18 --device cuda
#  python3.8 pytorch_model_test.py --model /path/to/model.pt --image https://...jpg --runs 20

import argparse
import time
import os
import sys
import math
import urllib.request
from io import BytesIO

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

def download_image(url):
    req = urllib.request.urlopen(url, timeout=10)
    data = req.read()
    return Image.open(BytesIO(data)).convert("RGB")

def load_image(path_or_url, resize=256, crop=224):
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        img = download_image(path_or_url)
    else:
        img = Image.open(path_or_url).convert("RGB")
    # transform: resize shorter side to `resize`, center crop to `crop`
    transform = T.Compose([
        T.Resize(resize),
        T.CenterCrop(crop),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)  # 1xCxHxW

def try_load_imagenet_labels():
    try:
        import json
        with urllib.request.urlopen(IMAGENET_LABELS_URL, timeout=5) as fh:
            labels = json.load(fh)
            if isinstance(labels, list):
                return labels
    except Exception:
        pass
    return None

def build_model_by_name(name, device):
    name = name.lower()
    # torchvision model factory (add more names if desired)
    if name == "resnet18":
        model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    elif name == "mobilenet_v2":
        model = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1")
    elif name == "squeezenet1_0" or name == "squeezenet":
        model = torchvision.models.squeezenet1_0(weights="IMAGENET1K_V1")
    elif name == "resnet50":
        model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
    else:
        raise ValueError("Unknown builtin model: " + name)
    model.eval()
    model.to(device)
    return model

def load_model(path_or_name, device):
    # if path exists and ends with .pt/.pth or is a file -> try torch.jit.load or torch.load
    if os.path.isfile(path_or_name):
        ext = os.path.splitext(path_or_name)[1].lower()
        try:
            # try jit first
            model = torch.jit.load(path_or_name, map_location=device)
            model.eval()
            model.to(device)
            return model
        except Exception:
            # try normal load (state_dict or full model)
            try:
                loaded = torch.load(path_or_name, map_location=device)
                # if it's a state_dict, try to infer architecture? user should use scripted model or provide factory.
                if isinstance(loaded, dict):
                    raise RuntimeError("Loaded a state_dict. Please provide a scripted/jit model or use a builtin model name.")
                model = loaded
                model.eval()
                model.to(device)
                return model
            except Exception as e:
                raise RuntimeError(f"Failed to load model file '{path_or_name}': {e}")
    else:
        # try known model names (torchvision)
        return build_model_by_name(path_or_name, device)

def warmup_and_run(model, input_tensor, device, warmup=5, runs=20, batch_size=1):
    input_tensor = input_tensor.to(device)
    # If batch_size >1, tile
    if batch_size > 1:
        input_tensor = input_tensor.repeat(batch_size, 1, 1, 1)
    # warmup
    with torch.no_grad():
        for i in range(max(1, warmup)):
            _ = model(input_tensor)
        # timed runs
        times = []
        for i in range(runs):
            t0 = time.time()
            _ = model(input_tensor)
            # synchronize if cuda
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            times.append((t1 - t0) * 1000.0)  # ms
    return times

def topk_from_output(output, k=5):
    # output: tensor (B, C) or (C,)
    if output.dim() == 1:
        out = output.unsqueeze(0)
    else:
        out = output
    probs = torch.nn.functional.softmax(out, dim=1)
    topv, topi = probs.topk(k, dim=1)
    return topv.cpu().numpy(), topi.cpu().numpy()

def main():
    p = argparse.ArgumentParser(description="PyTorch model quick test (works on Jetson/python3.8)")
    p.add_argument("--model", type=str, default="resnet18",
                   help="model name (resnet18,mobilenet_v2,squeezenet1_0,...) or path to .pt/.pth/.jit")
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p.add_argument("--image", type=str, default=None, help="image path or URL (if omitted, uses a tiny random image)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()

    # device
    use_cuda = (args.device.lower() == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Python exe:", sys.executable)
    print("Torch version:", getattr(torch, "__version__", "N/A"))
    print("CUDA available:", torch.cuda.is_available(), "using device:", device)
    if use_cuda:
        try:
            print("CUDA runtime:", torch.version.cuda, "devices:", torch.cuda.device_count())
            print("Current device:", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))
        except Exception:
            pass

    # load image or random
    if args.image:
        try:
            input_tensor = load_image(args.image)
        except Exception as e:
            print("Failed to load image:", e)
            return
    else:
        # random image (224x224)
        dummy = (torch.rand(3, 224, 224) * 255).byte()
        input_tensor = T.Compose([T.ToPILImage(), T.ToTensor(),
                                  T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])(dummy).unsqueeze(0)

    # try to load model
    try:
        model = load_model(args.model, device)
    except Exception as e:
        print("Model load error:", e)
        return

    # try to fetch labels
    labels = try_load_imagenet_labels()
    if labels:
        print("Loaded ImageNet labels (count=%d)" % len(labels))
    else:
        print("No ImageNet labels available; will print class indices.")

    # warmup & run
    print(f"[RUN] warmup={args.warmup} runs={args.runs} batch_size={args.batch_size}")
    times = warmup_and_run(model, input_tensor, device, warmup=args.warmup, runs=args.runs, batch_size=args.batch_size)
    import numpy as np
    arr = np.array(times)
    mean_ms = arr.mean()
    p50 = np.percentile(arr, 50)
    p90 = np.percentile(arr, 90)
    p99 = np.percentile(arr, 99)
    print(f"Latency (ms) mean={mean_ms:.2f} p50={p50:.2f} p90={p90:.2f} p99={p99:.2f}")
    fps = 1000.0 / mean_ms * args.batch_size if mean_ms > 0 else float("inf")
    print(f"Approx throughput: {fps:.2f} FPS (batch_size={args.batch_size})")

    # single forward to get outputs and topk
    model.eval()
    with torch.no_grad():
        inp = input_tensor.to(device)
        if args.batch_size > 1:
            inp = inp.repeat(args.batch_size, 1, 1, 1)
        out = model(inp)
        # use first sample
        out0 = out[0] if out.dim() == 2 else out
        topv, topi = topk_from_output(out0.unsqueeze(0), k=args.topk)
        topv = topv[0]; topi = topi[0]
        print("Top-k predictions:")
        for score, idx in zip(topv, topi):
            if labels and idx < len(labels):
                name = labels[idx]
            else:
                name = f"class_{idx}"
            print(f"  {name:30s}  {float(score):.4f}")

if __name__ == "__main__":
    main()
