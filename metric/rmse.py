import numpy as np


def standard_rmse(a, b):
    """Root Mean Squared Error chuẩn:  RMSE = sqrt((1/N) * sum((a-b)^2))."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def rmse(A, B, F):
    """Fusion RMSE chuẩn quốc tế.

    Tính trung bình RMSE giữa ảnh tổng hợp F và mỗi ảnh nguồn A, B:
        Fusion_RMSE = (RMSE(A, F) + RMSE(B, F)) / 2

    Đầu vào: uint8 grayscale (0-255).

    Ghi chú: phiên bản trước (theo MATLAB VIFB) có công thức MSE sai
    khiến RMSE chỉ ~0.2 cho ảnh khác hẳn nhau (đúng phải ~100). Phiên
    bản này dùng công thức RMSE chuẩn.
    """
    rmse_AF = standard_rmse(A, F)
    rmse_BF = standard_rmse(B, F)
    return (rmse_AF + rmse_BF) / 2.0
