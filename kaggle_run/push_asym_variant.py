"""Helper: tạo + push 1 Kaggle kernel cho 1 asym variant.

Cách dùng:
    python kaggle_run/push_asym_variant.py AsymD-DB3-BaseVSM

Sẽ:
  1. Đọc kaggle_run/train_asym.ipynb (template)
  2. Sửa VARIANT trong Cell 2 thành variant_name
  3. Sửa kernel-metadata-asym.json -> id + title theo variant_name
  4. Tạo thư mục kaggle_run/_runs_asym/<variant_name>/ chứa notebook + metadata
  5. Chạy `kaggle kernels push` từ thư mục đó

Yêu cầu: `kaggle` CLI đã cài + `~/.kaggle/kaggle.json` token.
"""
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
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    variant = sys.argv[1]

    root = Path(__file__).resolve().parent
    template_nb = root / "train_asym.ipynb"
    template_meta = root / "kernel-metadata-asym.json"

    out_dir = root / "_runs_asym" / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Sửa notebook ----
    with open(template_nb, encoding="utf-8") as f:
        nb = json.load(f)

    # Cell 2 chứa VARIANT — find and replace
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell.get("source", [])
        if isinstance(src, list):
            new_src = []
            for line in src:
                new_src.append(re.sub(
                    r"^VARIANT\s*=\s*'[^']+'(.*)$",
                    f"VARIANT     = '{variant}'\\1",
                    line, flags=re.M))
            cell["source"] = new_src

    out_nb = out_dir / "train_asym.ipynb"
    with open(out_nb, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    # ---- Sửa metadata ----
    with open(template_meta, encoding="utf-8") as f:
        meta = json.load(f)
    kernel_slug = f"cddfuse-{slugify(variant)}"
    meta["id"] = f"kienvbhp1234/{kernel_slug}"
    meta["title"] = f"CDDFuse-{variant} (Module D)"
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
