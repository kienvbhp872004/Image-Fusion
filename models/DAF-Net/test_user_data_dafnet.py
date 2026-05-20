import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from model.net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtractor, DetailFeatureExtractor
from utils.imageUtils import img_save, image_read_cv2

def run_dafnet():
    print("Running DAF-Net inference on user data...")
    ckpt_path = r"checkPoints/daf_net.pth"
    dir_MRI = r"../../data/MRI"
    dir_PET = r"../../data/PET"
    test_out_folder = r"../../results/DAF-Net"
    os.makedirs(test_out_folder, exist_ok=True)

    device = 'cpu' # Using CPU
    Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    BaseFuseLayer = nn.DataParallel(BaseFeatureExtractor(dim=64, num_heads=8)).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtractor(num_layers=1)).to(device)

    # Load weights
    checkpoint = torch.load(ckpt_path, map_location=device)
    Encoder.load_state_dict(checkpoint["DIDF_Encoder"])
    Decoder.load_state_dict(checkpoint["DIDF_Decoder"])
    BaseFuseLayer.load_state_dict(checkpoint["BaseFuseLayer"])
    DetailFuseLayer.load_state_dict(checkpoint["DetailFuseLayer"])
    
    Encoder.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()

    img_list = [f for f in os.listdir(dir_MRI) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    
    with torch.no_grad():
        for img_name in img_list:
            path_PET = os.path.join(dir_PET, img_name)
            path_MRI = os.path.join(dir_MRI, img_name)
            
            if not os.path.exists(path_PET):
                continue
                
            data_IR = image_read_cv2(path_PET, mode='GRAY')[np.newaxis, np.newaxis, ...]/255.0
            data_VIS = image_read_cv2(path_MRI, mode='GRAY')[np.newaxis, np.newaxis, ...]/255.0

            data_IR, data_VIS = torch.FloatTensor(data_IR).to(device), torch.FloatTensor(data_VIS).to(device)

            feature_V_B, feature_V_D, feature_V = Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I = Encoder(data_IR)
            feature_F_B = BaseFuseLayer(feature_V_B + feature_I_B)
            feature_F_D = DetailFuseLayer(feature_V_D + feature_I_D)
            
            data_Fuse, _ = Decoder(data_VIS, feature_F_B, feature_F_D)
            data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))
            fi = np.squeeze((data_Fuse * 255).cpu().numpy())
            
            cv2.imwrite(os.path.join(test_out_folder, img_name), fi.astype(np.uint8))
            print(f"Fused: {img_name}")

if __name__ == "__main__":
    run_dafnet()
