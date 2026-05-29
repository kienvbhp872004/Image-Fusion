"""Helper: tạo + push 1 Kaggle kernel cho 1 asym variant.

Cách dùng:
    # Phase II only (E/D frozen, default Module D mode):
    python kaggle_run/push_asym_variant.py AsymD-DB3-BaseVSM

    # Full training (Phase I 15 ep + Phase II 30 ep, like AG-45ep):
    python kaggle_run/push_asym_variant.py AsymD-DB5-BaseLocalEntropy --phase1 15 --epochs 30

Sẽ:
  1. Đọc kaggle_run/train_asym.ipynb (template)
  2. Sửa VARIANT, PHASE1_EPOCHS, EPOCHS trong Cell 2
  3. Sửa kernel-metadata-asym.json -> id + title theo variant_name
  4. Tạo thư mục kaggle_run/_runs_asym/<variant_name>/ chứa notebook + metadata
  5. Chạy `kaggle kernels push` từ thư mục đó

Yêu cầu: `kaggle` CLI đã cài + `~/.kaggle/kaggle.json` token.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def slugify(s: str) -> str:
    s = re.sub(r'[^a-zA-Z0-9]+', '-', s).strip('-').lower()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--phase1", type=int, default=0,
                    help="Phase 1 epochs (default 0 = frozen E/D mode)")
    ap.add_argument("--epochs", type=int, default=30, help="Phase 2 epochs")
    args = ap.parse_args()
    variant = args.variant

    root = Path(__file__).resolve().parent
    template_nb = root / "train_asym.ipynb"
    template_meta = root / "kernel-metadata-asym.json"

    out_dir = root / "_runs_asym" / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Sửa notebook ----
    with open(template_nb, encoding="utf-8") as f:
        nb = json.load(f)

    # Cell 2 chứa VARIANT/EPOCHS — find and replace; thêm PHASE1_EPOCHS
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell.get("source", [])
        if not isinstance(src, list):
            continue
        # Only process Cell 2 (config) — detect via VARIANT line
        if not any("VARIANT" in s and "=" in s for s in src):
            continue
        new_src = []
        for line in src:
            line = re.sub(
                r"^VARIANT\s*=\s*'[^']+'(.*)$",
                f"VARIANT     = '{variant}'\\1",
                line, flags=re.M)
            line = re.sub(
                r"^EPOCHS\s*=\s*\d+(.*)$",
                f"EPOCHS      = {args.epochs}\\1",
                line, flags=re.M)
            new_src.append(line)
        # Inject PHASE1_EPOCHS variable if not present
        if not any("PHASE1_EPOCHS" in s for s in new_src):
            for i, line in enumerate(new_src):
                if line.startswith("EPOCHS"):
                    new_src.insert(i + 1, f"PHASE1_EPOCHS = {args.phase1}    # 0 = frozen E/D, >0 = full training\n")
                    break
        cell["source"] = new_src
        break

    out_nb = out_dir / "train_asym.ipynb"
    with open(out_nb, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    # ---- Sửa metadata ----
    with open(template_meta, encoding="utf-8") as f:
        meta = json.load(f)
    kernel_slug = f"cddfuse-{slugify(variant)}"
    meta["id"] = f"kienvbhp1234/{kernel_slug}"
    # Title không có "(Module D)" để slug khớp id (tránh 409 conflict khi re-push)
    meta["title"] = f"CDDFuse-{variant}"
    meta["code_file"] = "train_asym.ipynb"

    out_meta = out_dir / "kernel-metadata.json"
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[ready] {out_dir}")
    print(f"[id   ] {meta['id']}")

    # ---- Push ----
    print(f"[push ] kaggle kernels push -p {out_dir}")
    r = subprocess.run(["kaggle", "kernels", "push", "-p", str(out_dir)],
                       capture_output=False)
    if r.returncode != 0:
        print(f"[err  ] kaggle push failed (exit {r.returncode})")
        sys.exit(r.returncode)
    print(f"[done ] variant={variant}")


if __name__ == "__main__":
    main()
