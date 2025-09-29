#!/usr/bin/env python3
"""
trt_infer_camera_fixed.py
Safer TensorRT YOLOv8-pose webcam tester with defensive checks and debug prints.

Usage:
    python3.8 trt_infer_camera_fixed.py --engine yolov8s-pose_fp16.trt --imgsz 640 --camera 0
"""
import argparse, time, math, sys
import numpy as np
import cv2
# numpy deprecated aliases
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def nms(boxes, scores, iou_th=0.45):
    if len(boxes) == 0: return []
    idxs = np.argsort(scores)[::-1]
    keep = []
    boxes = np.array(boxes)
    while idxs.size:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1: break
        rest = idxs[1:]
        ious = []
        b1 = boxes[i]
        for j in rest:
            b2 = boxes[j]
            x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
            x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
            if x2 <= x1 or y2 <= y1:
                iou = 0.0
            else:
                inter = (x2-x1)*(y2-y1)
                a = (b1[2]-b1[0])*(b1[3]-b1[1])
                b = (b2[2]-b2[0])*(b2[3]-b2[1])
                iou = inter / (a + b - inter + 1e-8)
            ious.append(iou)
        ious = np.array(ious)
        idxs = rest[ious <= iou_th]
    return keep

# TRT helpers
def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine, input_shape):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        shape = engine.get_binding_shape(i)
        is_input = engine.binding_is_input(i)
        if is_input:
            host = cuda.pagelocked_empty(int(np.prod(input_shape)), np.float32)
            dev = cuda.mem_alloc(host.nbytes)
            inputs.append({"name":name,"host":host,"dev":dev,"shape":input_shape})
            bindings.append(int(dev))
        else:
            if any([d == -1 for d in shape]):
                raise RuntimeError("Dynamic output shape not supported.")
            out_size = int(np.prod(shape))
            host = cuda.pagelocked_empty(out_size, np.float32)
            dev = cuda.mem_alloc(host.nbytes)
            outputs.append({"name":name,"host":host,"dev":dev,"shape":tuple(shape)})
            bindings.append(int(dev))
    return inputs, outputs, bindings, stream

def letterbox(img, new_shape=(640,640), color=(114,114,114)):
    h0,w0 = img.shape[:2]
    r = min(new_shape[0]/h0, new_shape[1]/w0)
    new_unpad = int(round(w0*r)), int(round(h0*r))
    dw = new_shape[1]-new_unpad[0]; dh = new_shape[0]-new_unpad[1]
    dw//=2; dh//=2
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = dh, new_shape[0]-new_unpad[1]-dh
    left, right = dw, new_shape[1]-new_unpad[0]-dw
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top)

def preprocess(frame, imgsz):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img, r, pad = letterbox(img, new_shape=(imgsz, imgsz))
    img = img.astype(np.float32)/255.0
    img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img, 0)
    return img, r, pad

def decode_yolov8_pose(out_arr, imgsz=640, strides=(8,16,32), conf_thres=0.35, kpt_thres=0.3, iou_thres=0.45, debug=False):
    try:
        a = out_arr
        if a.ndim == 3: a = a[0]
        preds = a.transpose(1,0).copy()
        N, C = preds.shape
        if C != 56:
            if debug: print("[DECODE] channels !=56:", C)
            return []
        # build grids
        grid_list=[]; stride_list=[]; tot=0
        for s in strides:
            g = imgsz//s
            gx,gy = np.meshgrid(np.arange(g), np.arange(g))
            gx=gx.ravel(); gy=gy.ravel()
            grid = np.stack([gx,gy], axis=1)
            grid_list.append(grid)
            stride_list.append(np.full((grid.shape[0],), s, dtype=np.float32))
            tot += grid.shape[0]
        if tot != N:
            if debug:
                print(f"[DECODE] grid cell mismatch: tot={tot} != N={N} for strides={strides}")
            return []
        grid_all = np.concatenate(grid_list, axis=0)
        stride_all = np.concatenate(stride_list, axis=0)
        xy = preds[:,0:2]; wh = preds[:,2:4]; obj_raw = preds[:,4]
        kpts = preds[:,5:5+17*3].reshape(-1,17,3)
        xy = sigmoid(xy); wh = sigmoid(wh); obj = sigmoid(obj_raw)
        kpt_xy = sigmoid(kpts[..., :2]); kpt_conf = sigmoid(kpts[..., 2])
        cxcy = (xy*2.0 - 0.5 + grid_all) * stride_all.reshape(-1,1)
        wh_pixels = ((wh*2.0)**2) * stride_all.reshape(-1,1)
        xs = cxcy[:,0]; ys = cxcy[:,1]
        ws = wh_pixels[:,0]; hs = wh_pixels[:,1]
        boxes = np.stack([xs - ws/2.0, ys - hs/2.0, xs + ws/2.0, ys + hs/2.0], axis=1)
        kpt_xy_img = (kpt_xy*2.0 - 0.5 + grid_all.reshape(-1,1,2)) * stride_all.reshape(-1,1,1)
        mean_kpt_conf = kpt_conf.mean(axis=1)
        score = obj * mean_kpt_conf
        mask = score > conf_thres
        if not np.any(mask): return []
        boxes_f = boxes[mask]; scores_f = score[mask]; kpt_xy_f = kpt_xy_img[mask]; kpt_conf_f = kpt_conf[mask]
        keep = nms(boxes_f.tolist(), scores_f, iou_th=iou_thres)
        dets = []
        for i in keep:
            b = boxes_f[i]
            s = float(scores_f[i])
            kps = np.zeros((17,3), dtype=np.float32)
            kps[:,0:2] = kpt_xy_f[i]
            kps[:,2] = kpt_conf_f[i]
            dets.append({"box": b.tolist(), "score": s, "kpts": kps})
        return dets
    except Exception as e:
        if debug: print("[DECODE] exception:", e)
        return []

COCO_EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (0,5),(0,6),(5,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--kpt-thr", type=float, default=0.3)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    engine = load_engine(args.engine)
    print("Loaded engine:", args.engine)
    # detect engine input shape
    input_shape = None
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            shape = engine.get_binding_shape(i)
            if len(shape) >= 4:
                input_shape = tuple(int(x) for x in shape[:4])
            else:
                input_shape = tuple(int(x) for x in shape)
            break
    if input_shape is None:
        print("Cannot detect engine input shape. Aborting.")
        return
    engine_h = int(input_shape[2]); engine_w = int(input_shape[3])
    engine_imgsz = min(engine_h, engine_w)
    if args.imgsz != engine_imgsz:
        print(f"[WARN] overriding imgsz {args.imgsz} -> engine imgsz {engine_imgsz}")
        args.imgsz = engine_imgsz
    print("[INFO] using imgsz =", args.imgsz)

    inputs, outputs, bindings, stream = allocate_buffers(engine, (1,3,args.imgsz,args.imgsz))
    context = engine.create_execution_context()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Camera open failed"); return

    possible_perms = [(8,16,32),(8,32,16),(16,8,32),(16,32,8),(32,8,16),(32,16,8)]
    chosen_perm = None
    frame_idx = 0
    try:
        print("Starting loop; will auto-detect stride order on first usable frame.")
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01); continue
            frame_idx += 1
            h0,w0 = frame.shape[:2]
            inp, r, pad = preprocess(frame, args.imgsz)
            flat = inp.ravel().astype(np.float32)
            if flat.size != inputs[0]["host"].size:
                inputs[0]["host"] = cuda.pagelocked_empty(flat.size, np.float32)
            inputs[0]["host"][:] = flat
            cuda.memcpy_htod_async(inputs[0]["dev"], inputs[0]["host"], stream)
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            for out in outputs:
                cuda.memcpy_dtoh_async(out["host"], out["dev"], stream)
            stream.synchronize()
            out_shape = outputs[0]["shape"]
            out_arr = np.array(outputs[0]["host"], dtype=np.float32).reshape(out_shape)

            if chosen_perm is None:
                # test permutations and choose one yielding detections
                best_perm = None; best_cnt = 0; best_sample = None
                for perm in possible_perms:
                    dets_try = decode_yolov8_pose(out_arr, imgsz=args.imgsz, strides=perm, conf_thres=args.conf, kpt_thres=args.kpt_thr, debug=args.debug)
                    if args.debug:
                        print(f"perm {perm} -> {len(dets_try)} dets")
                    if len(dets_try) > best_cnt:
                        best_cnt = len(dets_try); best_perm = perm; best_sample = dets_try
                if best_perm is None or best_cnt == 0:
                    chosen_perm = (8,16,32)
                    print("[AUTO] fallback stride order:", chosen_perm)
                else:
                    chosen_perm = best_perm
                    print(f"[AUTO] chosen stride order: {chosen_perm}  (initial detections: {best_cnt})")

            dets = decode_yolov8_pose(out_arr, imgsz=args.imgsz, strides=chosen_perm, conf_thres=args.conf, kpt_thres=args.kpt_thr, debug=args.debug)

            disp = frame.copy()
            # defensive loop: skip malformed dets
            for di, d in enumerate(dets):
                b = d.get("box", None)
                if b is None:
                    if args.debug: print(f"[WARN] det {di} missing box -> skip")
                    continue
                # ensure box is iterable of length 4 with numeric entries
                try:
                    if isinstance(b, (list, tuple, np.ndarray)):
                        if len(b) != 4:
                            if args.debug: print(f"[WARN] det {di} box len !=4 -> skip ({b})")
                            continue
                        x1,y1,x2,y2 = [float(v) for v in b]
                    else:
                        # scalar or weird -> skip
                        if args.debug: print(f"[WARN] det {di} box not sequence -> skip ({type(b)})")
                        continue
                except Exception as e:
                    if args.debug: print(f"[WARN] det {di} box parse error -> skip ({e})")
                    continue

                score = float(d.get("score", 0.0))
                kps = d.get("kpts", None)
                left, top = pad; scale = r
                if kps is not None:
                    # map and draw keypoints
                    kps_xy = kps[:,0:2].copy()
                    kps_xy[:,0] = (kps_xy[:,0] - left) / scale
                    kps_xy[:,1] = (kps_xy[:,1] - top) / scale
                    for i in range(kps.shape[0]):
                        conf = float(kps[i,2])
                        if conf >= args.kpt_thr:
                            x = int(kps_xy[i,0]); y = int(kps_xy[i,1])
                            if 0 <= x < w0 and 0 <= y < h0:
                                cv2.circle(disp, (x,y), 3, (0,0,255), -1)
                    for (a,bidx) in COCO_EDGES:
                        if kps[a,2] >= args.kpt_thr and kps[bidx,2] >= args.kpt_thr:
                            ax = int((kps[a,0]-left)/scale); ay = int((kps[a,1]-top)/scale)
                            bx = int((kps[bidx,0]-left)/scale); by = int((kps[bidx,1]-top)/scale)
                            cv2.line(disp, (ax,ay), (bx,by), (0,255,0), 2)
                # draw bbox
                x1i = int((x1 - left)/scale); y1i = int((y1 - top)/scale)
                x2i = int((x2 - left)/scale); y2i = int((y2 - top)/scale)
                cv2.rectangle(disp, (x1i,y1i), (x2i,y2i), (255,0,0), 2)
                cv2.putText(disp, f"{score:.2f}", (x1i, y1i-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

            t1 = time.time()
            fps = 1.0/(t1-t0+1e-9)
            cv2.putText(disp, f"{fps:.1f} FPS", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.imshow("TRT Pose (fixed)", disp)
            k = cv2.waitKey(1) & 0xFF
            if k == 27: break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
