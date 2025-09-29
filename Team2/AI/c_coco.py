import os, json, glob
from pathlib import Path

DATASET_ROOT = "/home/bbang/Workspace/Intel07_Intelproject_Team3/dataset"
SPLITS = ["Training","Validation","Test"]
DEV_DIRS = ["1_digital_camera","2_smart_pad","3_smart_phone"]

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def iter_jsons(split):
    base = Path(DATASET_ROOT)/split/"labeled_data"
    for dev in DEV_DIRS:
        ddir = base/dev
        if not ddir.exists(): 
            continue
        for subj in ddir.iterdir():
            if subj.is_dir():
                for jf in subj.glob("*.json"):
                    yield dev, subj.name, jf

def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return json.loads(path.read_text(encoding="cp949", errors="ignore"))

def is_custom_schema(js):
    return (
        isinstance(js, dict) and
        "info" in js and "images" in js and "annotations" in js and "equipment" in js
    )

def coerce_number(x):
    if x is None: return None
    if isinstance(x, (int, float, bool)): return float(x)
    if isinstance(x, str):
        try: return float(x)
        except: return None
    return None

def flatten_kv(prefix, obj, out):
    if obj is None: return
    if isinstance(obj, dict):
        for k,v in obj.items():
            nk = f"{prefix}_{k}" if prefix else str(k)
            flatten_kv(nk, v, out)
    elif isinstance(obj, list):
        for i,v in enumerate(obj):
            nk = f"{prefix}_{i}" if prefix else str(i)
            flatten_kv(nk, v, out)
    else:
        num = coerce_number(obj)
        if num is not None:
            out[prefix] = num

def facepart_name(code):
    mapping = {
        0: "full",
        1: "forehead",
        2: "glabella",
        3: "left_crowsfeet",
        4: "right_crowsfeet",
        5: "left_cheek",
        6: "right_cheek",
        7: "lips",
        8: "chin"
    }
    return mapping.get(code, f"facepart_{code}")

def find_origin_image(split, device, subject, filename):
    """labeled json의 info.filename을 origin_data의 실제 이미지와 매칭"""
    base = Path(DATASET_ROOT)/split/"origin_data"/device/subject
    cand = base/filename
    if cand.exists():
        return str(cand.relative_to(Path(DATASET_ROOT)))
    stem = Path(filename).stem
    for ext in [".jpg",".jpeg",".png",".bmp",".JPG",".PNG"]:
        c = base/f"{stem}{ext}"
        if c.exists():
            return str(c.relative_to(Path(DATASET_ROOT)))
    for p in base.glob("*"):
        if p.is_file() and p.stem == stem:
            return str(p.relative_to(Path(DATASET_ROOT)))
    return None

def main():
    for split in SPLITS:
        print(f"[{split}] 스캔 시작…")
        images, annotations, categories = [], [], []
        cat2id = {}

        def get_cat_id(name, supercat="skin_metrics"):
            if name not in cat2id:
                cid = len(cat2id) + 1
                cat2id[name] = cid
                categories.append({"id": cid, "name": name, "supercategory": supercat})
            return cat2id[name]

        for code in [0,1,2,3,4,5,6,7,8]:
            get_cat_id(f"facepart::{facepart_name(code)}", supercat="roi")

        img_id = 0
        ann_id = 0
        seen_rel = set()

        total = 0
        custom_cnt = 0
        skipped = 0

        for device, subject, jf in iter_jsons(split):
            total += 1
            try:
                js = read_json(jf)
            except Exception as e:
                if total % 1000 == 0:
                    print(f"  [WARN] read fail: {jf} ({e})")
                continue

            if not is_custom_schema(js):
                skipped += 1
                continue
            custom_cnt += 1

            info = js.get("info", {})
            imgs = js.get("images", {})
            ann  = js.get("annotations", {})
            eq   = js.get("equipment", {})

            filename = info.get("filename")
            W = imgs.get("width")
            H = imgs.get("height")
            facepart = imgs.get("facepart")
            bbox = imgs.get("bbox") 
            angle = imgs.get("angle")
            device_idx = imgs.get("device")

            if not filename or W is None or H is None:
                continue

            rel = find_origin_image(split, device, subject, filename)
            if rel is None or rel in seen_rel:
                continue
            seen_rel.add(rel)

            img_id += 1
            images.append({
                "id": img_id,
                "file_name": rel,    
                "width": W, "height": H,
                "meta": {
                    "split": split,
                    "device": device,           
                    "device_idx": device_idx,
                    "subject": subject,
                    "angle": angle,
                    "date": info.get("date"),
                    "gender": info.get("gender"),
                    "age": info.get("age"),
                    "skin_type": info.get("skin_type"),
                    "sensitive": info.get("sensitive"),
                    "orig_json": str(jf.relative_to(Path(DATASET_ROOT)))
                }
            })

            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x0, y0, x1, y1 = bbox
                x, y, w, h = float(x0), float(y0), float(x1 - x0), float(y1 - y0)
                roi_cat = get_cat_id(f"facepart::{facepart_name(facepart)}", supercat="roi")
                ann_id += 1
                annotations.append({
                    "id": ann_id, "image_id": img_id, "category_id": roi_cat,
                    "bbox": [x, y, w, h], "area": float(w*h), "iscrowd": 0
                })

            flat = {}
            def _safe_flatten():
                flatten_kv("ann", ann, flat)
                flatten_kv("eq",  eq,  flat)
            _safe_flatten()

            for key, val in flat.items():
                cat_id = get_cat_id(key, supercat="skin_metrics")
                ann_id += 1
                annotations.append({
                    "id": ann_id, "image_id": img_id, "category_id": cat_id,
                    "value": float(val), "area": 0, "bbox": [], "iscrowd": 0
                })

            if img_id % 5000 == 0:
                print(f"  processed images: {img_id} (annotations: {len(annotations)})")

        merged = {
            "info": {"description": f"Merged {split} dataset", "version": "1.0", "year": 2025},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
        out_path = Path(DATASET_ROOT)/f"merged_{split.lower()}.json"
        ensure_parent(out_path)
        out_path.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
        print(f"[{split}] 저장 완료: {out_path} | images={len(images)}, anns={len(annotations)}, cats={len(categories)} | 통계: total_json={total}, custom={custom_cnt}, skipped={skipped}")

if __name__ == "__main__":
    main()
