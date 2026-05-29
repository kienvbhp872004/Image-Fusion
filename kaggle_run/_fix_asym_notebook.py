"""Fix train_asym.ipynb: replace broken `\\n` literal trong train/eval cell with single-line commands."""
import json
from pathlib import Path

nb_path = Path(__file__).resolve().parent / "train_asym.ipynb"
with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

# Cell 10: train command — single line (no continuation)
nb["cells"][10]["source"] = [
    "import glob, pathlib\n",
    "candidates = glob.glob(f'/kaggle/input/**/CDDFuse_MIF.pth', recursive=True)\n",
    "assert candidates, 'CDDFuse_MIF.pth không tìm thấy trong /kaggle/input/* — attach dataset cddfuse-pretrained'\n",
    "PRETRAINED = candidates[0]\n",
    "print(f'[pretrained] {PRETRAINED}')\n",
    "\n",
    "amp_flag = '--amp' if USE_AMP else ''\n",
    "# Phase 1 epochs: 0 = frozen E/D mode (default), >0 = full training\n",
    "p1_ep = globals().get('PHASE1_EPOCHS', 0)\n",
    "!python train_asymmetric.py --variant $VARIANT --pretrained $PRETRAINED --num_epochs $EPOCHS --num_phase1_epochs $p1_ep --batch $BATCH --coeff_decomp 2.0 --coeff_tv 10.0 --seed $SEED --output /kaggle/working/ $amp_flag",
]

# Cell 12: eval — single line per modality
nb["cells"][12]["source"] = [
    "import glob\n",
    "ckpts = sorted(glob.glob(f'/kaggle/working/CDDFuse-{VARIANT}_MIF_*.pth'))\n",
    "assert ckpts, 'No checkpoint found from Cell 5'\n",
    "CKPT = ckpts[-1]\n",
    "OUT_DIR = f'/kaggle/working/CDDFuse-{VARIANT}'\n",
    "print(f'[ckpt] {CKPT}')\n",
    "\n",
    "import shutil, pathlib\n",
    "MYDS = pathlib.Path('/kaggle/working/Image-Fusion/Havard-Medical-Image-Fusion-Datasets-main/Havard-Medical-Image-Fusion-Datasets-main/MyDatasets')\n",
    "TEST_OUT = pathlib.Path('/kaggle/working/Image-Fusion/data/reference')\n",
    "for modal in ['CT-MRI', 'PET-MRI', 'SPECT-MRI']:\n",
    "    sub = modal.split('-')[0]\n",
    "    src_test = MYDS / modal / 'test'\n",
    "    if not src_test.exists():\n",
    "        print(f'[skip] {modal} test missing'); continue\n",
    "    (TEST_OUT / modal).mkdir(parents=True, exist_ok=True)\n",
    "    shutil.copytree(src_test, TEST_OUT / modal, dirs_exist_ok=True)\n",
    "    n = len(list((TEST_OUT / modal / sub).glob('*.png')))\n",
    "    print(f'[test ] {modal}: {n} cặp staged')\n",
    "\n",
    "HARVARD_ROOT = '/kaggle/working/Image-Fusion/data/reference'\n",
    "for modal in ['CT', 'PET', 'SPECT']:\n",
    "    !python evaluate_cddfuse.py --variant $VARIANT --modal $modal --ckpt $CKPT --harvard_root $HARVARD_ROOT --out_dir $OUT_DIR --save_perimage",
]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Fixed {nb_path}")
