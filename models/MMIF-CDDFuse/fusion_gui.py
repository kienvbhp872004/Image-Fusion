"""
CDDFuse-AG — Simple Fusion GUI
Browse 2 images → fuse → preview & save.
Run from models/MMIF-CDDFuse/:  python fusion_gui.py
"""
import sys, os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

import numpy as np
from PIL import Image, ImageTk
import torch
import torch.nn as nn

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from net import (Restormer_Encoder, Restormer_Decoder,
                 BaseFeatureExtraction, DetailFeatureExtraction)
from variants.registry_asymmetric import build_asym_variant

DEFAULT_CKPT = SCRIPT_DIR / "models" / "CDDFuse_MIF.pth"
VARIANT      = "AsymD-Comb-WAvg-SML"   # CDDFuse-AG
THUMB        = (280, 280)               # preview size

# ── model ──────────────────────────────────────────────────────────────────────
class FusionModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded = False

    def load(self, ckpt_path: str):
        dev = self.device
        self.enc  = Restormer_Encoder().to(dev)
        self.dec  = Restormer_Decoder().to(dev)
        self.base = BaseFeatureExtraction(dim=64, num_heads=8).to(dev)
        self.det  = DetailFeatureExtraction(num_layers=1).to(dev)
        self.rule_b, self.rule_d, _, _ = build_asym_variant(VARIANT)
        self.rule_b = self.rule_b.to(dev)
        self.rule_d = self.rule_d.to(dev)

        ckpt = torch.load(ckpt_path, map_location=dev)
        def fix(sd):
            return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

        self.enc.load_state_dict(fix(ckpt["DIDF_Encoder"]))
        self.dec.load_state_dict(fix(ckpt["DIDF_Decoder"]))
        self.base.load_state_dict(fix(ckpt["BaseFuseLayer"]))
        self.det.load_state_dict(fix(ckpt["DetailFuseLayer"]))
        if "BaseRule" in ckpt:
            self.rule_b.load_state_dict(fix(ckpt["BaseRule"]))
        if "DetailRule" in ckpt:
            self.rule_d.load_state_dict(fix(ckpt["DetailRule"]))

        for m in [self.enc, self.dec, self.base, self.det, self.rule_b, self.rule_d]:
            m.eval()
        self.loaded = True

    def fuse(self, img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
        """img_a, img_b: HxW uint8. Returns HxW uint8 fused."""
        def to_tensor(arr):
            t = torch.from_numpy(arr.astype(np.float32) / 255.0)
            return t.unsqueeze(0).unsqueeze(0).to(self.device)  # 1x1xHxW

        ta, tb = to_tensor(img_a), to_tensor(img_b)
        with torch.no_grad():
            fb_a, fd_a, _ = self.enc(ta)
            fb_b, fd_b, _ = self.enc(tb)
            ff_b = self.base(self.rule_b(fb_a, fb_b))
            ff_d = self.det(self.rule_d(fd_a, fd_b))
            fused, _ = self.dec(None, ff_b, ff_d)
            fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)
        out = (fused.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return out


MODEL = FusionModel()

# ── helpers ────────────────────────────────────────────────────────────────────
def load_gray(path: str) -> np.ndarray:
    """Load any image as grayscale HxW uint8.
    For color (PET/SPECT): extracts Y channel from YCbCr."""
    img = Image.open(path)
    if img.mode in ("RGB", "RGBA"):
        img = img.convert("YCbCr")
        y, _, _ = img.split()
        return np.array(y)
    return np.array(img.convert("L"))

def arr_to_photoimage(arr: np.ndarray, size=THUMB) -> ImageTk.PhotoImage:
    img = Image.fromarray(arr).resize(size, Image.LANCZOS)
    return ImageTk.PhotoImage(img)


# ── GUI ────────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CDDFuse-AG — Medical Image Fusion")
        self.resizable(False, False)
        self.configure(bg="#f5f5f5")

        self.path_a   = tk.StringVar()
        self.path_b   = tk.StringVar()
        self.path_ckpt= tk.StringVar(value=str(DEFAULT_CKPT))
        self.arr_a    = None
        self.arr_b    = None
        self.arr_out  = None
        self._ph      = [None, None, None]   # keep refs alive

        self._build_ui()
        self._try_autoload()

    # ── UI ─────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        pad = dict(padx=8, pady=6)

        # ── Checkpoint row ──
        ckpt_fr = tk.Frame(self, bg="#f5f5f5")
        ckpt_fr.pack(fill="x", **pad)
        tk.Label(ckpt_fr, text="Checkpoint:", bg="#f5f5f5", width=11, anchor="w").pack(side="left")
        tk.Entry(ckpt_fr, textvariable=self.path_ckpt, width=52,
                 font=("Consolas", 9)).pack(side="left", padx=4)
        tk.Button(ckpt_fr, text="Browse", command=self._browse_ckpt,
                  width=7).pack(side="left")
        self.lbl_model = tk.Label(ckpt_fr, text="● not loaded",
                                  fg="#c0392b", bg="#f5f5f5", font=("Segoe UI", 9))
        self.lbl_model.pack(side="left", padx=8)

        # ── Image panels ──
        panel_fr = tk.Frame(self, bg="#f5f5f5")
        panel_fr.pack(**pad)

        self.canvas = []
        titles = ["Ảnh nguồn A  (MRI / CT / PET / SPECT)",
                  "Ảnh nguồn B  (MRI / CT / PET / SPECT)",
                  "Ảnh tổng hợp  (CDDFuse-AG)"]
        for i, title in enumerate(titles):
            col = tk.Frame(panel_fr, bg="#f5f5f5")
            col.grid(row=0, column=i, padx=10)
            tk.Label(col, text=title, bg="#f5f5f5",
                     font=("Segoe UI", 9, "bold")).pack()
            c = tk.Canvas(col, width=THUMB[0], height=THUMB[1],
                          bg="#dddddd", highlightthickness=1,
                          highlightbackground="#aaaaaa")
            c.pack()
            self.canvas.append(c)
            if i < 2:
                tk.Button(col, text="Chọn ảnh",
                          command=lambda idx=i: self._browse_img(idx),
                          width=14).pack(pady=4)

        # ── Action buttons ──
        btn_fr = tk.Frame(self, bg="#f5f5f5")
        btn_fr.pack(**pad)
        self.btn_fuse = tk.Button(btn_fr, text="⚡  Fuse", width=16, height=2,
                                  font=("Segoe UI", 11, "bold"),
                                  bg="#27ae60", fg="white",
                                  activebackground="#1e8449",
                                  command=self._on_fuse, state="disabled")
        self.btn_fuse.pack(side="left", padx=10)
        tk.Button(btn_fr, text="💾  Lưu kết quả", width=16, height=2,
                  font=("Segoe UI", 11),
                  command=self._on_save).pack(side="left", padx=10)

        # ── Status bar ──
        self.status = tk.StringVar(value="Chọn checkpoint và 2 ảnh đầu vào.")
        tk.Label(self, textvariable=self.status, bg="#e8e8e8",
                 relief="sunken", anchor="w",
                 font=("Segoe UI", 9)).pack(fill="x", side="bottom")

    # ── Handlers ───────────────────────────────────────────────────────────────
    def _try_autoload(self):
        if DEFAULT_CKPT.exists():
            threading.Thread(target=self._load_model, daemon=True).start()

    def _browse_ckpt(self):
        p = filedialog.askopenfilename(title="Chọn checkpoint",
                                       filetypes=[("PyTorch checkpoint", "*.pth *.pt")])
        if p:
            self.path_ckpt.set(p)
            threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        path = self.path_ckpt.get()
        self.lbl_model.config(text="● đang tải...", fg="#e67e22")
        self.status.set(f"Đang tải model từ {Path(path).name} ...")
        try:
            MODEL.load(path)
            self.lbl_model.config(text=f"● CDDFuse-AG  [{MODEL.device}]", fg="#27ae60")
            self.status.set("Model đã sẵn sàng. Chọn 2 ảnh rồi bấm Fuse.")
            self._update_fuse_btn()
        except Exception as e:
            self.lbl_model.config(text="● lỗi", fg="#c0392b")
            self.status.set(f"Lỗi load model: {e}")

    def _browse_img(self, idx: int):
        p = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Ảnh", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if not p:
            return
        arr = load_gray(p)
        if idx == 0:
            self.arr_a = arr
            self.path_a.set(p)
        else:
            self.arr_b = arr
            self.path_b.set(p)
        ph = arr_to_photoimage(arr)
        self._ph[idx] = ph
        c = self.canvas[idx]
        c.delete("all")
        c.create_image(THUMB[0]//2, THUMB[1]//2, image=ph)
        self.status.set(f"Đã tải ảnh {'A' if idx==0 else 'B'}: {Path(p).name}  "
                        f"({arr.shape[1]}×{arr.shape[0]})")
        self._update_fuse_btn()

    def _update_fuse_btn(self):
        ready = MODEL.loaded and self.arr_a is not None and self.arr_b is not None
        self.btn_fuse.config(state="normal" if ready else "disabled")

    def _on_fuse(self):
        self.btn_fuse.config(state="disabled")
        self.status.set("Đang tổng hợp ảnh...")
        threading.Thread(target=self._run_fuse, daemon=True).start()

    def _run_fuse(self):
        try:
            # Resize B to match A if needed
            a = self.arr_a
            b = self.arr_b
            if a.shape != b.shape:
                b = np.array(Image.fromarray(b).resize(
                    (a.shape[1], a.shape[0]), Image.LANCZOS))

            out = MODEL.fuse(a, b)
            self.arr_out = out
            ph = arr_to_photoimage(out)
            self._ph[2] = ph
            c = self.canvas[2]
            c.delete("all")
            c.create_image(THUMB[0]//2, THUMB[1]//2, image=ph)
            self.status.set(f"Tổng hợp xong!  ({out.shape[1]}×{out.shape[0]} px)  "
                            "— Bấm 'Lưu kết quả' để xuất file.")
        except Exception as e:
            self.status.set(f"Lỗi: {e}")
        finally:
            self._update_fuse_btn()

    def _on_save(self):
        if self.arr_out is None:
            messagebox.showwarning("Chưa có kết quả", "Hãy fuse ảnh trước.")
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
            initialfile="fused_cddfuse_ag.png")
        if p:
            Image.fromarray(self.arr_out).save(p)
            self.status.set(f"Đã lưu: {p}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
