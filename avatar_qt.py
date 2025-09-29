import os, json
import numpy as np
import multiprocessing
import functools

# ✅ PyQt 먼저
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QImage

# ✅ 그 다음 OpenCV
import cv2


# ---- COCO indices (used joints) ----
L_SH=5; R_SH=6; L_EL=7; R_EL=8; L_WR=9; R_WR=10
L_HP=11; R_HP=12; L_KN=13; R_KN=14; L_AN=15; R_AN=16

# ===================== Global Cache & Worker =====================

ASSET_CACHE = {}
WORKER_ASSETS = None
WORKER_BACKGROUND = None

def init_worker(assets, background):
    """워커 프로세스 초기화 함수"""
    global WORKER_ASSETS, WORKER_BACKGROUND
    WORKER_ASSETS = assets
    WORKER_BACKGROUND = background

def _render_worker(args):
    """멀티프로세싱 Pool을 위한 최상위 레벨 워커 함수"""
    kps, config, offset_x, top_pad = args
    # 클래스 외부에서 staticmethod 호출
    return MannequinRenderer.render_pose_frame(kps, WORKER_ASSETS, offset_x, top_pad, WORKER_BACKGROUND, config)


# ===================== Alpha Blending Implementation =====================
# Numba를 사용하여 알파 블렌딩을 가속합니다. Numba가 없으면 Numpy로 대체됩니다.
try:
    import numba
    print("[info] Numba JIT will be used for alpha blending.")

    @numba.jit(nopython=True, cache=True)
    def _alpha_paste_full_impl(dst, src):
        """Numba를 사용한 고속 알파 블렌딩"""
        for y in range(dst.shape[0]):
            for x in range(dst.shape[1]):
                alpha = src[y, x, 3] / 255.0
                
                if alpha > 0.01: # 거의 투명한 픽셀은 건너뜁니다.
                    if alpha > 0.99: # 거의 불투명한 픽셀은 복사합니다.
                        dst[y, x, 0] = src[y, x, 0]
                        dst[y, x, 1] = src[y, x, 1]
                        dst[y, x, 2] = src[y, x, 2]
                    else: # 블렌딩
                        for c in range(3):
                            dst[y, x, c] = np.uint8(
                                (1.0 - alpha) * dst[y, x, c] + alpha * src[y, x, c]
                            )
        return dst

except ImportError:
    print("[warning] Numba not installed or failed to import. Using slower numpy-based alpha blending.")
    def _alpha_paste_full_impl(dst, src):
        """Numpy를 사용한 대체 알파 블렌딩"""
        if src.shape[2] != 4:
            return dst
        
        alpha = (src[:, :, 3:4].astype(np.float32)) / 255.0
        rgb = src[:, :, :3].astype(np.float32)
        
        # Perform blending on the whole array. It's safer and the cost of masking
        # might outweigh the benefits if many pixels are non-transparent.
        dst_float = dst.astype(np.float32)
        blended = dst_float * (1.0 - alpha) + rgb * alpha
        dst[:] = blended.astype(np.uint8)
        return dst

class MannequinRenderer(QObject):
    # ===== Qt Signals =====
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    error = pyqtSignal(str)
    playReady = pyqtSignal(object, float)  # (frames: list[QImage], fps: float)
    finished = pyqtSignal()  

    def __init__(
        self,
        json_path: str,
        assets_dir: str,
        side_extra: int = 240,
        top_pad: int = 100,
        stride: int = 1,
        show_debug: bool = True,
        follow_center_x: bool = True,
        pose_swap_lr: bool = True,
        pose_hflip: bool = False,
        parent=None
    ):
        super().__init__(parent)
        self.json_path = json_path
        self.assets_dir = assets_dir

        self.side_extra = int(side_extra)
        self.top_pad = int(top_pad)
        self.stride = int(stride)

        self.show_debug = bool(show_debug)
        self.follow_center_x = bool(follow_center_x)
        self.pose_swap_lr = bool(pose_swap_lr)
        self.pose_hflip = bool(pose_hflip)

        # Filled by anchors.json
        self.REF_W = None
        self.REF_H = None
        self.CANVAS_W = None
        self.CANVAS_H = None

        self.ANCHORS = {}
        self.PARTS = {}
        self.TIP_LOWER = {}
        self.options = {}

        # Cached assets/background
        self.assets = {}
        self.background = None
        self.v_align_mode = "center"
        self._cancel = False 


    def cancel(self):
        self._cancel = True

    # ===================== Utility =====================

    @staticmethod
    def _to_bgr3(img):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 4:
            a = (img[:,:,3:4].astype(np.float32) / 255.0)
            rgb = img[:,:,:3].astype(np.float32)
            base = np.zeros_like(rgb, dtype=np.float32)
            return (base*(1.0-a) + rgb*a).astype(np.uint8)
        if img.ndim == 3 and img.shape[2] == 3:
            return img
        raise ValueError("Unsupported image format")

    @staticmethod
    def _resize_cover(img, tw, th):
        h, w = img.shape[:2]
        if w == 0 or h == 0:
            raise ValueError("Invalid image size")
        scale = max(tw / float(w), th / float(h))
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)
        x0 = max(0, (nw - tw) // 2); y0 = max(0, (nh - th) // 2)
        return resized[y0:y0+th, x0:x0+tw]

    @staticmethod
    def _resize_fit(img, tw, th, pad_color=(0,0,0)):
        h, w = img.shape[:2]
        scale = min(tw / float(w), th / float(h))
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)
        canvas = np.full((th, tw, 3), pad_color, np.uint8)
        x0 = (tw - nw)//2; y0 = (th - nh)//2
        canvas[y0:y0+nh, x0:x0+nw] = resized
        return canvas

    @staticmethod
    def _cv_bgr_to_qimage(img_bgr):
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(img_rgb.data, w, h, img_rgb.strides[0], QImage.Format_RGB888)
        return qimg.copy()
    
    @staticmethod
    def _compute_dy(kps_raw, top_pad, config):
        mode = config.get("v_align_mode", "top_pad")
        H = config["CANVAS_H"]

        ys = [float(kps_raw[i,1]) for i in range(kps_raw.shape[0]) if np.isfinite(kps_raw[i,1])]
        if not ys:
            return float(top_pad)

        ymin, ymax = min(ys), max(ys)

        if mode == "center":
            char_h = max(1.0, ymax - ymin)
            target_top = (H - char_h) * 0.5
            return float(target_top - ymin)

        if mode == "feet":
            cand = []
            for idx in (L_AN, R_AN, L_KN, R_KN, L_HP, R_HP):
                y = kps_raw[idx,1]
                if np.isfinite(y):
                    cand.append(float(y))
            anchor_y = max(cand) if cand else ymax
            floor_y = H - float(config.get("bottom_margin_px", 40))
            return float(floor_y - anchor_y)

        return float(top_pad)

    # ===================== Loaders =====================

    def load_asset_pack(self):
        global ASSET_CACHE
        if self.assets_dir in ASSET_CACHE:
            cached_data = ASSET_CACHE[self.assets_dir]
            self.REF_W, self.REF_H = cached_data["ref_size"]
            self.PARTS = cached_data["parts"]
            self.ANCHORS = cached_data["anchors"]
            self.TIP_LOWER = cached_data["tip_lower"]
            self.options = cached_data["options"]
            self.log.emit("[info] Loaded assets from cache.")
            return

        cfg_path = os.path.join(self.assets_dir, "anchors.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"anchors.json이 없습니다: {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        ref_size = cfg.get("ref_size")
        if not ref_size:
            raise ValueError("anchors.json: 'ref_size'가 필요합니다. 예: [600, 1000]")
        self.REF_W, self.REF_H = int(ref_size[0]), int(ref_size[1])

        parts_json = cfg.get("parts", {})
        if not parts_json:
            raise ValueError("anchors.json: 'parts'가 비었습니다.")
        self.PARTS = {k: {"file": v} for k, v in parts_json.items()}

        self.ANCHORS = {}
        anchors_json = cfg.get("anchors", {})
        for key, meta in anchors_json.items():
            if ">" not in key: continue
            a,b = key.split(">",1)
            parent = (float(meta["parent"][0]), float(meta["parent"][1]))
            child  = (float(meta["child"][0]),  float(meta["child"][1]))
            self.ANCHORS[(a,b)] = {"parent": parent, "child": child}

        self.TIP_LOWER = {}
        tips_json = cfg.get("tip_lower", {})
        for name, pt in tips_json.items():
            self.TIP_LOWER[name] = (float(pt[0]), float(pt[1]))

        self.options = cfg.get("options", {}) or {}
        self.options["background"] = cfg.get("background", {}) or {}

        # Cache the loaded data
        ASSET_CACHE[self.assets_dir] = {
            "ref_size": (self.REF_W, self.REF_H),
            "parts": self.PARTS,
            "anchors": self.ANCHORS,
            "tip_lower": self.TIP_LOWER,
            "options": self.options,
            "assets": {} # For image data
        }

    def load_rgba_resized(self, path):
        if self.REF_W is None or self.REF_H is None:
            raise RuntimeError("REF_W/REF_H must be set before loading assets.")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None: raise FileNotFoundError(path)
        if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        if (img.shape[1], img.shape[0]) != (self.REF_W, self.REF_H):
            img = cv2.resize(img,(self.REF_W,self.REF_H),interpolation=cv2.INTER_NEAREST)
        return img

    def build_background_from_spec(self, spec, tw, th):
        color = tuple(spec.get("color", [0,0,0]))
        mode  = spec.get("resize", "cover")
        blur  = float(spec.get("blur", 0))
        path  = spec.get("path", None)

        if path:
            if not os.path.isabs(path): path = os.path.join(self.assets_dir, path)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None: raise FileNotFoundError(f"Background not found: {path}")
            img3 = self._to_bgr3(img)
            if mode == "native": bg = img3
            elif mode == "fit": bg = self._resize_fit(img3, tw, th, pad_color=color)
            else: bg = self._resize_cover(img3, tw, th)
        else:
            bg = np.full((th, tw, 3), color, np.uint8)

        if blur > 0:
            k = int(max(1, round(blur))); k += (k % 2 == 0)
            bg = cv2.GaussianBlur(bg, (k,k), 0)
        return bg

    def load_json_scaled(self, path):
        if self.REF_W is None or self.REF_H is None:
            raise RuntimeError("REF_W/REF_H must be set before loading pose JSON.")
        with open(path,"r",encoding="utf-8") as f: data=json.load(f)
        fps=float(data.get("fps",30.0)); stride=int(data.get("stride",1))
        Wv,Hv=data["video_size"]; sx,sy=self.REF_W/float(Wv),self.REF_H/float(Hv)
        frames=[]
        for fr in data["frames"]:
            pts=[]
            for xy in fr["kps"]:
                if xy[0] is None or xy[1] is None: pts.append([np.nan,np.nan])
                else: pts.append([float(xy[0])*sx,float(xy[1])*sy])
            frames.append(np.array(pts,np.float32))
        return fps/stride, frames

    # ===================== Math / Render helpers (Static) =====================

    @staticmethod
    def H_translate(tx,ty):
        return np.array([[1,0,tx],[0,1,ty],[0,0,1]],np.float32)

    @staticmethod
    def warp_full(img_rgba, H, canvas_w, canvas_h):
        M=H[:2,:]
        return cv2.warpAffine(
            img_rgba, M, (canvas_w, canvas_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)
        )

    @staticmethod
    def alpha_paste_full(dst, src):
        return _alpha_paste_full_impl(dst, src)

    @staticmethod
    def hip_center_x(kps):
        if np.all(np.isfinite(kps[[L_HP,R_HP]])): return 0.5*(kps[L_HP,0]+kps[R_HP,0])
        return np.nan

    @staticmethod
    def swap_lr_labels(kps):
        out=kps.copy()
        pairs=[(L_SH,R_SH),(L_EL,R_EL),(L_WR,R_WR),(L_HP,R_HP),(L_KN,R_KN),(L_AN,R_AN)]
        for l,r in pairs: out[[l,r]]=out[[r,l]]
        return out

    @staticmethod
    def hflip_coords(kps, config):
        out=kps.copy()
        if out.size==0: return out
        out[:,0]=config["REF_W"]-1-out[:,0]
        return out

    @staticmethod
    def reconstruct_skeleton_follow_json(kps_raw, dx, dy):
        out=np.full((17,2),np.nan,np.float32)
        for i,pt in enumerate(kps_raw):
            if np.all(np.isfinite(pt)): out[i]=pt+[dx,dy]
        return out

    @staticmethod
    def compute_body_shrink_ratio(kps_scaled, config):
        anchors = config["ANCHORS"]
        base_L = np.array(anchors[("body","left_upper_arm")]["parent"], np.float32)
        base_R = np.array(anchors[("body","right_upper_arm")]["parent"], np.float32)
        base_shoulder = np.linalg.norm(base_L - base_R) + 1e-6
        if np.all(np.isfinite(kps_scaled[[L_SH, R_SH]])):
            shoulder_width = np.linalg.norm(kps_scaled[L_SH] - kps_scaled[R_SH])
            r = shoulder_width / base_shoulder
            return float(min(1.0, max(0.1, r)))
        return 1.0

    @staticmethod
    def get_part_base_width(anchor_start, anchor_end):
        p0=np.array(anchor_start,np.float32); p1=np.array(anchor_end,np.float32)
        v=p1-p0; length=np.linalg.norm(v)
        return 1.0 if length < 1e-6 else length * 0.1

    @staticmethod
    def attach_body_affine(assets, canvas, world_LS, world_RS, world_HC, config, scale_x=1.0):
        if np.any(~np.isfinite([world_LS, world_RS, world_HC])): return
        img = assets["body"]; anchors = config["ANCHORS"]
        pL = np.array(anchors[("body","left_upper_arm")]["parent"], np.float32)
        pR = np.array(anchors[("body","right_upper_arm")]["parent"], np.float32)
        pH = 0.5*(np.array(anchors[("left_upper_leg","body")]["parent"], np.float32) +
                  np.array(anchors[("right_upper_leg","body")]["parent"], np.float32))
        src_pts = np.array([pL, pR, pH], np.float32); dst_pts = np.array([world_LS, world_RS, world_HC], np.float32)
        H = cv2.getAffineTransform(src_pts, dst_pts)
        warped = cv2.warpAffine(img, H, (config["CANVAS_W"], config["CANVAS_H"]),
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        MannequinRenderer.alpha_paste_full(canvas, warped)

    @staticmethod
    def attach_segment_scaled(part_name, anchor_start, anchor_end, world_start, world_end, assets, canvas, config, scale_x=1.0):
        if np.any(~np.isfinite(world_start)) or np.any(~np.isfinite(world_end)): return
        img = assets[part_name]
        p0=np.array(anchor_start,np.float32); p1=np.array(anchor_end,np.float32)
        seg_local=p1-p0; len_local=np.linalg.norm(seg_local)+1e-6; dir_local=seg_local/len_local
        perp_local=np.array([-dir_local[1],dir_local[0]],np.float32)
        base_width = MannequinRenderer.get_part_base_width(anchor_start, anchor_end)
        mid_local = 0.5*(p0+p1) + perp_local*base_width
        q0=np.array(world_start,np.float32); q1=np.array(world_end,np.float32)
        seg_world=q1-q0; len_world=np.linalg.norm(seg_world)
        if len_world < 1e-6: return
        dir_world=seg_world/len_world; perp_world=np.array([-dir_world[1],dir_world[0]],np.float32)
        thickness = base_width * float(min(1.0, max(0.1, scale_x)))
        mid_world = 0.5*(q0+q1) + perp_world*thickness
        src_pts=np.array([p0,p1,mid_local],np.float32); dst_pts=np.array([q0,q1,mid_world],np.float32)
        H = cv2.getAffineTransform(src_pts, dst_pts)
        warped = cv2.warpAffine(img, H, (config["CANVAS_W"], config["CANVAS_H"]),
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        MannequinRenderer.alpha_paste_full(canvas, warped)

    # ===================== Frame render (Static) =====================

    @staticmethod
    def render_pose_frame(kps_raw, assets, offset_x, top_pad, background, config):
        dx = float(offset_x)
        if config["follow_center_x"]:
            hx = MannequinRenderer.hip_center_x(kps_raw)
            if np.isfinite(hx): dx += config["CANVAS_W"] * 0.5 - float(hx)
        dy = MannequinRenderer._compute_dy(kps_raw, top_pad, config)
        kps_scaled = MannequinRenderer.reconstruct_skeleton_follow_json(kps_raw, dx, dy)
        canvas = background.copy() if background is not None else np.zeros((config["CANVAS_H"], config["CANVAS_W"], 3), np.uint8)
        scale_x = MannequinRenderer.compute_body_shrink_ratio(kps_scaled, config)
        
        left_upper_ok=np.all(np.isfinite(kps_scaled[[L_HP,L_KN]])); left_lower_ok=np.all(np.isfinite(kps_scaled[[L_KN,L_AN]]))
        right_upper_ok=np.all(np.isfinite(kps_scaled[[R_HP,R_KN]])); right_lower_ok=np.all(np.isfinite(kps_scaled[[R_KN,R_AN]]))
        body_ok=np.all(np.isfinite(kps_scaled[[L_SH,R_SH,L_HP,R_HP]]))
        
        ren_mode = any(s in os.path.basename(config["assets_dir"]).lower() for s in ("ren_parts", "rens_parts"))
        
        attach_segment = functools.partial(MannequinRenderer.attach_segment_scaled, assets=assets, canvas=canvas, config=config, scale_x=scale_x)
        anchors = config["ANCHORS"]; tip_lower = config["TIP_LOWER"]

        def render_legs():
            if left_upper_ok: attach_segment("left_upper_leg", anchors[("left_upper_leg","body")]["parent"], anchors[("left_upper_leg","left_lower_leg")]["parent"], kps_scaled[L_HP], kps_scaled[L_KN])
            if left_lower_ok: attach_segment("left_lower_leg", anchors[("left_upper_leg","left_lower_leg")]["child"], tip_lower["left_lower_leg"], kps_scaled[L_KN], kps_scaled[L_AN])
            if right_upper_ok: attach_segment("right_upper_leg", anchors[("right_upper_leg","body")]["parent"], anchors[("right_upper_leg","right_lower_leg")]["parent"], kps_scaled[R_HP], kps_scaled[R_KN])
            if right_lower_ok: attach_segment("right_lower_leg", anchors[("right_upper_leg","right_lower_leg")]["child"], tip_lower["right_lower_leg"], kps_scaled[R_KN], kps_scaled[R_AN])

        def render_body():
            if body_ok:
                hip_center = 0.5 * (kps_scaled[L_HP] + kps_scaled[R_HP])
                MannequinRenderer.attach_body_affine(assets, canvas, kps_scaled[L_SH], kps_scaled[R_SH], hip_center, config, scale_x=scale_x)
            else:
                H = MannequinRenderer.H_translate(dx, dy)
                warped = MannequinRenderer.warp_full(assets["body"], H, config["CANVAS_W"], config["CANVAS_H"])
                MannequinRenderer.alpha_paste_full(canvas, warped)

        def render_arms():
            if np.all(np.isfinite(kps_scaled[[L_SH,L_EL]])): attach_segment("left_upper_arm", anchors[("body","left_upper_arm")]["child"], anchors[("left_upper_arm","left_lower_arm")]["parent"], kps_scaled[L_SH], kps_scaled[L_EL])
            if np.all(np.isfinite(kps_scaled[[R_SH,R_EL]])): attach_segment("right_upper_arm", anchors[("body","right_upper_arm")]["child"], anchors[("right_upper_arm","right_lower_arm")]["parent"], kps_scaled[R_SH], kps_scaled[R_EL])
            if np.all(np.isfinite(kps_scaled[[L_EL,L_WR]])): attach_segment("left_lower_arm", anchors[("left_upper_arm","left_lower_arm")]["child"], tip_lower["left_lower_arm"], kps_scaled[L_EL], kps_scaled[L_WR])
            if np.all(np.isfinite(kps_scaled[[R_EL,R_WR]])): attach_segment("right_lower_arm", anchors[("right_upper_arm","right_lower_arm")]["child"], tip_lower["right_lower_arm"], kps_scaled[R_EL], kps_scaled[R_WR])

        if ren_mode:
            render_body()
            render_legs()
        else:
            render_legs()
            render_body()
        render_arms()

        if config["show_debug"]:
            for i in range(len(kps_scaled)):
                if np.all(np.isfinite(kps_scaled[i])):
                    cv2.circle(canvas, tuple(np.int32(kps_scaled[i])), 3, (0,255,255), -1)
        return canvas

    # ===================== Public API =====================

    def run(self):
        global ASSET_CACHE
        try:
            self.log.emit("[info] loading assets...")
            self.load_asset_pack()

            bg_spec = self.options.get("background", {}) or {}
            mode = bg_spec.get("resize", "cover"); bg_path = bg_spec.get("path", None)
            if mode == "native" and bg_path:
                path = bg_path if os.path.isabs(bg_path) else os.path.join(self.assets_dir, bg_path)
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None: raise FileNotFoundError(f"Background not found: {path}")
                self.CANVAS_H, self.CANVAS_W = self._to_bgr3(img).shape[:2]
            else:
                self.CANVAS_W = self.REF_W + self.side_extra*2
                self.CANVAS_H = self.REF_H + max(0, self.top_pad)
            
            self.background = self.build_background_from_spec(bg_spec, self.CANVAS_W, self.CANVAS_H)

            cached_assets = ASSET_CACHE[self.assets_dir].get("assets", {})
            if not cached_assets:
                self.log.emit("[info] Caching asset images...")
                for name, meta in self.PARTS.items():
                    p = os.path.join(self.assets_dir, meta["file"])
                    cached_assets[name] = self.load_rgba_resized(p)
                ASSET_CACHE[self.assets_dir]["assets"] = cached_assets
            self.assets = cached_assets

            fps, frames = self.load_json_scaled(self.json_path)
            self.log.emit(f"[info] frames={len(frames)} fps={fps:.3f}, starting parallel render...")

            config = {
                "REF_W": self.REF_W, "CANVAS_W": self.CANVAS_W, "CANVAS_H": self.CANVAS_H,
                "ANCHORS": self.ANCHORS, "TIP_LOWER": self.TIP_LOWER,
                "assets_dir": self.assets_dir, "v_align_mode": self.v_align_mode,
                "bottom_margin_px": getattr(self, "bottom_margin_px", 40),
                "show_debug": self.show_debug, "follow_center_x": self.follow_center_x
            }

            tasks = []
            for i in range(0, len(frames), self.stride):
                k = frames[i].copy()
                if self.pose_hflip: k = self.hflip_coords(k, config)
                if self.pose_swap_lr: k = self.swap_lr_labels(k)
                tasks.append((k, config, 0, self.top_pad))

            qframes = []
            num_tasks = len(tasks)
            
            # 워커 초기화 함수를 사용하여 큰 데이터 (assets, background)를 한 번만 전달합니다.
            init_args = (self.assets, self.background)
            
            try:
                # Use try-finally to ensure pool is closed
                pool = multiprocessing.Pool(initializer=init_worker, initargs=init_args)
                
                for i, result_img in enumerate(pool.imap_unordered(_render_worker, tasks)):
                    if self._cancel:
                        self.log.emit("[warning] Render cancelled by user.")
                        pool.terminate()
                        break
                    
                    qframes.append(self._cv_bgr_to_qimage(result_img))
                    self.progress.emit(int((i + 1) * 100 / num_tasks))
            finally:
                pool.close()
                pool.join()

            if not self._cancel:
                self.log.emit("[info] Render complete.")
                final_fps = fps / self.stride
                self.playReady.emit(qframes, final_fps)

        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")
        finally:
            self.finished.emit()