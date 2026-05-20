import numpy as np


def standard_mse(a, b):
    """Mean Squared Error chuẩn (ISO/IEC):  MSE = (1/N) * sum((a-b)^2)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(A, B, F):
    """Fusion PSNR chuẩn quốc tế.

    Tính trung bình PSNR giữa ảnh tổng hợp F và mỗi ảnh nguồn A, B:
        PSNR_X = 10 * log10(MAX^2 / MSE(X, F))
        Fusion_PSNR = (PSNR_A + PSNR_B) / 2

    Đầu vào: uint8 grayscale (0-255). MAX = 255.

    Ghi chú: phiên bản trước (theo MATLAB VIFB từ "code on the Internet")
    có công thức MSE sai khiến giá trị PSNR cao bất thường (~55 dB cho
    ảnh random). Phiên bản này dùng công thức MSE chuẩn.
    """
    MAX = 255.0

    mse_AF = standard_mse(A, F)
    mse_BF = standard_mse(B, F)

    psnr_AF = float('inf') if mse_AF == 0 else 10.0 * np.log10(MAX ** 2 / mse_AF)
    psnr_BF = float('inf') if mse_BF == 0 else 10.0 * np.log10(MAX ** 2 / mse_BF)

    return (psnr_AF + psnr_BF) / 2.0
