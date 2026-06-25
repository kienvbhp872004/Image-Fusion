# Đồ án tốt nghiệp

**Sinh viên**: Đỗ Trung Kiên  
**MSSV**: 20224869  
**Trường**: Đại học Bách Khoa Hà Nội (HUST)  
**Khoa**: Khoa học máy tính, Trường CNTT & Truyền thông  
**Đề tài**: Đề xuất mô hình CDDFuse-AG cho tổng hợp ảnh y tế đa phương thức  
**Giảng viên hướng dẫn**: TS. Phạm Đăng Hải · PGS. TS. Phạm Văn Hải  
**Năm**: 2026

---

## Cấu trúc thư mục

```
20224869-DoTrungKien-DATN/
│
├── 01_BaoCao/                  # Báo cáo đồ án
│   ├── latex/                  # Source files LaTeX (report_latex/)
│   └── pdf/                    # Bản PDF cuối cùng
│
├── 02_HuongDanCaiDat/          # Hướng dẫn cài đặt môi trường
│   └── README.md
│
├── 03_PowerPoint/              # Slide thuyết trình (PPT)
│
├── 04_Video/                   # Video demo (cập nhật sau)
│   ├── training/               # Video huấn luyện model
│   ├── case_studies/           # Video case studies
│   └── installation/           # Video hướng dẫn cài đặt
│
├── 05_Code/                    # Toàn bộ source code
│   └── Image-Fusion/           # Repository chính
│
├── 06_DuLieu/                  # Dữ liệu thực nghiệm
│   ├── train_data/             # 738 cặp ảnh Harvard MIF (train)
│   └── test_data/              # 72 cặp ảnh test (CT/PET/SPECT × 24)
│
├── 07_KetQuaThucNghiem/        # Kết quả đánh giá tất cả variants
│
└── 08_TaiLieuThamKhao/         # Bài báo tham khảo
```

---

## Tóm tắt nội dung

### Mục tiêu

Cải tiến mô hình CDDFuse (CVPR 2023) cho bài toán tổng hợp ảnh y tế đa phương
thức (CT-MRI, PET-MRI, SPECT-MRI) thông qua thiết kế **Asymmetric Fusion**:
tách riêng quy tắc tổng hợp cho nhánh Base (tần số thấp) và nhánh Detail (tần
số cao), thay vì dùng chung phép cộng đơn giản như CDDFuse gốc.

### Mô hình đề xuất: CDDFuse-AG

| Thành phần | Mô tả |
|---|---|
| **Nhánh Base** | Weighted Average Scalar (WAvg) — 1 tham số học $\theta$, trung bình hóa toàn cục có thể điều chỉnh |
| **Nhánh Detail** | Sum-Modified-Laplacian (SML) — lựa chọn cục bộ theo độ sắc nét Laplacian, không tham số học |
| **Tham số mới** | 4.161 tham số (~0,35% so với CDDFuse gốc 1.188.272) |
| **Huấn luyện** | 2 pha: Pha I (40 epoch, Encoder+Decoder); Pha II (80 epoch, end-to-end) |

### Đóng góp chính

1. **Thiết kế bất đối xứng** (Asymmetric Fusion Rule) — tận dụng bản chất khác nhau giữa thông tin tần số thấp (tương quan cao) và tần số cao (bổ trợ nhau)
2. **Ablation study hệ thống 3 giai đoạn**: khảo sát 13 quy tắc tổng hợp (5 Base, 8 Detail), so sánh bất đối xứng vs đối xứng
3. **Phát hiện Modality-Specificity**: không có quy tắc duy nhất tối ưu cho mọi modality; CT-MRI hưởng lợi nhiều nhất từ asymmetric fusion
4. **Mô hình tối giản hiệu quả**: chỉ ~4K tham số fusion, vượt CDDFuse trên nhóm chỉ số edge/texture

### Kết quả nổi bật

**6 chỉ số tốt nhất (SF · Qabf · AG · EI · QM · QMI):**

| Modality | CDDFuse-AG #1 | Cải thiện so với CDDFuse |
|---|---|---|
| MRI-CT | 6/6 chỉ số | SF +1.3, QM gần gấp đôi |
| MRI-PET | 5/6 chỉ số | QM +21.9% |
| MRI-SPECT | 3/6 chỉ số | QM +10.4% |
| **Tổng** | **14/18 chỉ số** | |

**Z-score tổng hợp (8 chỉ số, so sánh nhóm 6 phương pháp):**

| Phương pháp | z CT | z PET | z SPECT | z avg |
|---|---|---|---|---|
| CDDFuse | −0.014 | +0.432 | +0.440 | +0.286 |
| **CDDFuse-AG** | **+0.309** | +0.248 | **+0.475** | **+0.344** |

**So sánh 22 phương pháp SOTA** (Composite Z-score toàn bộ 22 chỉ số):  
CDDFuse (pretrained paper) xếp hạng **#2/22** trong bộ SOTA, xác nhận kiến trúc nền tảng mạnh.

---

## Hướng dẫn nhanh

1. Đọc báo cáo PDF tại `01_BaoCao/pdf/` (source LaTeX: `report_latex/` ở root repo)
2. Đọc paper NCKH tại `paper_nckh/main.pdf`
3. Xem slide thuyết trình tại `03_PowerPoint/`
4. Cài đặt môi trường theo hướng dẫn ở `02_HuongDanCaiDat/README.md`
5. Chạy thực nghiệm với code tại `05_Code/Image-Fusion/`

---

## Liên hệ

- Email: kien.dt224869@sis.hust.edu.vn
- GitHub: <https://github.com/kienvbhp872004/MMIF-CDDFuse-AG>
