import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import logging
import json
import pandas as pd
import sys
import datetime
import csv
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import metric as M
from PSF import PSF
from saver import resume
from utils import RGB2YCrCb, YCbCr2RGB

def compute_metrics(img_a: np.ndarray,
                    img_b: np.ndarray,
                    img_f: np.ndarray) -> dict:
    results = {}
    for name, fn in [('EN', M.entropy), ('VAR', M.variance), ('AG', M.average_gradient), 
                     ('SF', M.spatial_frequency), ('EI', M.edge_intensity)]:
        try: results[name] = float(fn(img_f))
        except: results[name] = None
    
    ref_list = [('NCIE', M.ncie), ('MI_mutual', M.mutual_information), ('NABF', M.nabf), 
                ('FMI', M.fmi), ('CE', M.cross_entropy), ('SSIM', M.ssim), ('PSNR', M.psnr), 
                ('RMSE', M.rmse), ('QG', M.qg_petrovic), ('QM', M.wavelet_qm), ('QC', M.piella_qc), 
                ('QS', M.piella_qs), ('QCB', M.chen_blum), ('QCV',   M.chen_varshney), 
                ('QY', M.yang_ssim), ('QMI', M.mi_normalized), ('QSF', M.sf_relative), 
                ('QNCIE', M.ncc_entropy), ('QTE', M.tsallis_entropy)]
    
    for name, fn in ref_list:
        try: results[name] = float(fn(img_a, img_b, img_f))
        except: results[name] = None
    return results

def evaluate_psfusion(modal='PET'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Paths
    model_path = SCRIPT_DIR / "results" / "PSFusion" / "checkpoints" / "best_model.pth"
    import os as _os
    data_root = Path(_os.environ['HARVARD_ROOT']) if _os.environ.get('HARVARD_ROOT') else REPO_ROOT / "Havard-Medical-Image-Fusion-Datasets-main" / "Havard-Medical-Image-Fusion-Datasets-main"
    output_base = Path(_os.environ['OUT_DIR']) if _os.environ.get('OUT_DIR') else REPO_ROOT / "results" / "PSFusion"
    
    if modal == 'PET':
        vi_dir = data_root / "PET-MRI" / "MRI"
        ir_dir = data_root / "PET-MRI" / "PET"
    elif modal == 'SPECT':
        vi_dir = data_root / "SPECT-MRI" / "MRI"
        ir_dir = data_root / "SPECT-MRI" / "SPECT"
    elif modal == 'CT':
        vi_dir = data_root / "CT-MRI" / "MRI"
        ir_dir = data_root / "CT-MRI" / "CT"
    else:
        raise ValueError(f"Unknown modality: {modal}")

    save_dir = output_base / modal
    save_dir.mkdir(parents=True, exist_ok=True)
    fusion_dir = save_dir / "Fusion"
    fusion_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    class_nb = 9 # Default from options.py
    model = PSF(class_nb).to(device)
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
        
    model = resume(model, model_save_path=str(model_path), device=device, is_train=False)
    model.eval()

    results = []

    mri_files = sorted([f for f in vi_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.bmp')])
    src_map = {f.stem: f for f in ir_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.bmp')}
    pairs = [(mf, src_map[mf.stem]) for mf in mri_files if mf.stem in src_map]
    
    # 10 pairs as standard
    pairs = pairs[:10]

    logger.info(f"Evaluating {modal} modality: {len(pairs)} pairs found")

    for i, (vi_path, ir_path) in enumerate(pairs):
        # Original code resizes to multiple of 32
        img_vi_pil = Image.open(str(vi_path)).convert('RGB')
        img_ir_pil = Image.open(str(ir_path)).convert('RGB')
        
        orig_w, orig_h = img_vi_pil.size
        # Resize to multiple of 32
        new_w = orig_w - (orig_w % 32)
        new_h = orig_h - (orig_h % 32)
        
        vi_resized = img_vi_pil.resize((new_w, new_h))
        ir_resized = img_ir_pil.resize((new_w, new_h))
        
        vi_tensor = TF.to_tensor(vi_resized).unsqueeze(0).to(device)
        ir_tensor = TF.to_tensor(ir_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            # PSFusion: rgb, depth
            # semantic_out, binary_out, boundary_out, fused_img, vi_img, ir_img
            _, _, _, fused_img_y, _, _ = model(vi_tensor, ir_tensor)
            
            # Color restoration using functional image (IR)
            _, ir_cb, ir_cr = RGB2YCrCb(ir_tensor)
            fused_rgb = YCbCr2RGB(fused_img_y, ir_cb, ir_cr)
            
        f_img_np = (fused_rgb.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        f_pil = Image.fromarray(f_img_np).resize((orig_w, orig_h), Image.BICUBIC)
        
        # Save fused image
        save_path = fusion_dir / vi_path.name
        # For non-color modalities (like CT), use Y
        if modal == 'CT':
            f_final = f_pil.convert('L')
        else:
            f_final = f_pil

        f_final.save(save_path)

        # Standard metrics calculation on Y channel (grayscale)
        f_y_np = np.array(f_pil.convert('L'))
        vi_gray_np = np.array(img_vi_pil.convert('L'))
        ir_gray_np = np.array(img_ir_pil.convert('L'))

        logger.info(f"  [{i+1}/{len(pairs)}] Processing {vi_path.name}...")
        metrics = compute_metrics(ir_gray_np, vi_gray_np, f_y_np)
        metrics['image'] = vi_path.name
        results.append(metrics)

    # Aggregation
    if not results: return
    df = pd.DataFrame(results)
    summary = df.mean(numeric_only=True).to_dict()
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_data = {'model': 'PSFusion', 'modal': modal, 'n_images': len(results), 'timestamp': timestamp}
    summary_data.update(summary)

    # Save CSV and JSON summary
    df.to_csv(save_dir / f"PSFusion_{modal}_all_metrics.csv", index=False)
    with open(save_dir / f"PSFusion_{modal}_summary.json", 'w') as f:
        json.dump(summary_data, f, indent=2)
        
    # Append to PSFusion Master CSV
    csv_path = output_base / 'PSFusion_summary.csv'
    file_exists = csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_data.keys()))
        if not file_exists: writer.writeheader()
        writer.writerow(summary_data)
        
    logger.info(f"Finished {modal}. Results saved to {save_dir}")
    return summary_data

if __name__ == "__main__":
    modalities = ['PET', 'SPECT', 'CT']
    all_summaries = {}
    for mod in modalities:
        try:
            all_summaries[mod] = evaluate_psfusion(mod)
        except Exception as e:
            logger.error(f"Failed to evaluate {mod}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("Evaluation complete for all modalities.")
    
    # Trigger global aggregate script
    logger.info("Triggering global aggregate scripts...")
    import subprocess
    subprocess.run(["python", str(REPO_ROOT / "results" / "aggregate_results.py")])
