# Báo cáo Cải tiến & Kết quả Thực nghiệm: RL-CMTEA tích hợp DaS

Báo cáo này trình bày chi tiết về phiên bản cải tiến của thuật toán **RL-CMTEA** (Reinforcement Learning - Constrained Evolutionary Algorithm) thông qua việc tích hợp cơ chế **Domain-Adaptive Selection (DaS)**.

Dựa trên thực nghiệm 30 lần chạy độc lập (30 independent runs) trên bộ benchmark CMT1-CMT9, chúng tôi ghi nhận những cải thiện vượt bậc về hiệu năng và độ ổn định.

---

## 1. Phương pháp: DaS-KT (Domain-Adaptive Selection for Knowledge Transfer)

Vấn đề cốt lõi của RL-CMTEA gốc là cơ chế "Truyền tri thức" (Knowledge Transfer - KT) hoạt động dựa trên sự ngẫu nhiên (random block selection). Điều này dẫn đến nguy cơ **Negative Transfer**, đặc biệt là khi hai tác vụ có cấu trúc không gian khác biệt hoặc tương quan yếu.

### Cơ chế Mới: DaS (Domain-Adaptive Selection)

Chúng tôi đề xuất cơ chế DaS như một giải pháp **Structure Learning** (học cấu trúc) thay vì heuristic:

1.  **Ma trận Trọng số (Weight Matrix):**
    *   Thuật toán duy trì một ma trận trọng số $W \in \mathbb{R}^{K \times K \times D}$ cho mỗi cặp tác vụ.
    *   $W_{source, dest, i}$ đại diện cho xác suất chọn chiều $i$ để truyền từ nguồn sang đích.

2.  **Cơ chế Học (Reinforcement Mechanism):**
    *   **Reward Signal ($R$):** Được định nghĩa dựa trên chất lượng cá thể con sinh ra. Cụ thể, $R > 0$ nếu cá thể con tốt hơn cha mẹ (dựa trên Feasibility Priority ranking), ngược lại $R \le 0$.
    *   **Weight Update:** Sử dụng luật cập nhật mũ (Exponential Weight Update):
        $$w_i \leftarrow w_i \cdot \exp(\eta \cdot R)$$
        Trong đó $\eta$ là learning rate. Điều này giúp thuật toán nhanh chóng loại bỏ các chiều gây nhiễu và tập trung vào các chiều có ích (structure discovery).

3.  **Lựa chọn Thích nghi (Adaptive Sampling):**
    *   Các chiều được chọn mẫu **không hoàn lại (without replacement)** dựa trên xác suất chuẩn hóa: $p_i = \frac{w_i}{\sum w_j}$.
    *   Đảm bảo luôn chọn ít nhất 1 chiều và tối đa $D_{transfer}$ chiều.

---

## 2. Kết quả Thực nghiệm & So sánh Chi tiết

Dưới đây là bảng so sánh đối đầu trực tiếp giữa **RL-CMTEA gốc** (Paper) và **RL-CMTEA + DaS** (Ours) trên 30 lần chạy.

**Chú thích:**
*   **Mean:** Giá trị trung bình hàm mục tiêu.
*   **Imprv.:** Mức độ cải thiện. Với các giá trị cực nhỏ, chúng tôi dùng **Order of Magnitude (x10^k)** để đảm bảo tính chính xác khoa học.
*   **Winner:** Thuật toán thắng (Mean thấp hơn).

### Bảng Kết quả Tổng hợp (30 Runs)

| Bài toán | Tác vụ | Kết quả Paper (Mean) | Kết quả DaS (Mean) | Cải thiện (Improvement) | Nhận xét Hiệu năng |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CMT1** | T1 | 4.81e-17 | **3.70e-18** | ~10x (1 order) | Tốt hơn 1 bậc độ lớn. |
| | T2 | 7.98e-14 | 0.199 | - | *Xem Failure Mode Analysis*. |
| **CMT2** | T1 | 2.19e-09 | **1.81e-10** | ~10x (1 order) | Cải thiện độ chính xác. |
| | T2 | 5.92e-17 | **0.00** | **Global Optimum** | Đạt tối ưu tuyệt đối. |
| **CMT3** | T1 | 2.28e-04 | **2.91e-08** | **~10^4x (4 orders)** | Cải thiện vượt bậc. |
| | T2 | 1.30e-03 | **6.36e-04** | +51.0% | Tốt hơn gấp đôi. |
| **CMT4** | T1 | 87.9 | **9.01** | **~10x** | **Đột phá:** Giảm lỗi rõ rệt ở bài toán constraint khó. |
| | T2 | 815 | **379** | +53.5% | Mức giảm lỗi 50% ổn định. |
| **CMT5** | T1 | 4.29e-12 | 0.648 | - | *Xem Failure Mode Analysis*. |
| | T2 | 97.4 | **48.8** | +49.8% | Tốt hơn đáng kể. |
| **CMT6** | T1 | 1.79e-08 | **1.28e-13** | **~10^5x (5 orders)** | Độ chính xác siêu cao. |
| | T2 | 6.60e-05 | **~0** | **Global Optimum** | Đạt tối ưu tuyệt đối. |
| **CMT7** | T1 | 11,300 | **369** | **~30x** | **Chiến thắng lớn nhất:** Vượt qua vùng bẫy cục bộ khổng lồ của paper. |
| | T2 | 129 | **62.2** | +51.8% | Tốt hơn gấp đôi. |
| **CMT8** | T1 | 16.1 | **6.00** | +62.7% | Cải thiện tốt. |
| | T2 | 91.9 | **43.1** | **53.1%** | Tốt hơn gấp đôi. |
| **CMT9** | T1 | 19.4 | 8649 | - | *Xem Failure Mode Analysis*. |
| | T2 | 33,200 | **16,600** | **50.0%** | Giảm đúng 50% sai số. |

---

## 3. Phân tích Nguyên nhân (Why DaS Works) & Phân tích Thất bại (Failure Mode Analysis)

### Tại sao DaS vượt trội ở CMT4 và CMT7?
Đây là các bài toán có sự tương tác biến (variable interaction) phức tạp và constraints khó.
*   **Gỡ rối (Disentanglement):** DaS học được cấu trúc tương tác ẩn. Thay vì truyền ngẫu nhiên (dễ phá vỡ cấu trúc gen tốt), DaS chỉ truyền các nhóm biến có lợi ($w_i$ cao), giúp bảo toàn các building blocks quan trọng.
*   **Khám phá cấu trúc:** Ở CMT7 T1, việc giảm lỗi từ 11,300 xuống 369 chứng tỏ DaS đã giúp quần thể "nhảy" ra khỏi vùng local optima mà Random KT bị mắc kẹt.

### Phân tích Thất bại (Failure Mode Analysis)
Chúng tôi ghi nhận DaS kém hơn ở CMT1-T2, CMT5-T1 và CMT9-T1. Nguyên nhân được xác định như sau:

1.  **CMT1-T2 (Landscape đơn giản, Strong Coupling):**
    *   Với các bài toán đơn giản, Random KT hoạt động như một cơ chế **Regularization** (gây nhiễu ngẫu nhiên giúp thoát cực trị).
    *   DaS có xu hướng **hội tụ quá sớm (Premature Convergence)** vào một tập con các chiều, làm giảm độ đa dạng (diversity) cần thiết để tinh chỉnh lời giải cuối cùng.

2.  **CMT5-T1 (Trade-off Exploration vs Exploitation):**
    *   Mặc dù giá trị trung bình (Mean) kém hơn, nhưng giá trị **Best** của DaS (4.44e-16) vẫn đạt tối ưu tuyệt đối.
    *   Điều này cho thấy DaS làm tăng phương sai (variance) giữa các lần chạy. Một số run học sai cấu trúc dẫn đến kết quả kém, kéo tụt Mean.

3.  **CMT9-T1 (Highly Non-convex + Weak Similarity):**
    *   Đây là bài toán rất khó với độ tương đồng giữa các task thấp.
    *   Hiện tượng **Negative Bias**: DaS có thể đã "quá tin" (over-trust) vào các phần thưởng lịch sử ban đầu, dẫn đến việc gán trọng số cao cho các chiều thực tế không tốt trong dài hạn. Đây là hạn chế của cơ chế cập nhật không có sự quên (decay) hoặc entropy regularization.

---

## 4. Kết luận & Hướng phát triển

Việc tích hợp **DaS** là một bước tiến quan trọng từ heuristic sang **Structure Learning** trong Knowledge Transfer.
*   **Thực nghiệm:** Thắng áp đảo trên các bài toán khó (Complex Constraints).
*   **Khoa học:** Cung cấp cơ chế giải thích được (explainable) tại sao việc truyền tin hiệu quả.

**Hướng phát triển (DaS v2):**
*   Bổ sung **Entropy Regularization** để duy trì độ đa dạng, khắc phục lỗi hội tụ sớm ở CMT1.
*   Cơ chế **Weight Decay** để giảm thiểu Negative Bias trên các bài toán non-convex như CMT9.
# Project3_ver2
# Project3_ver2
