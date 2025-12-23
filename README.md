# Domain-Adaptive Selection for Constrained Multitask Evolutionary Optimization

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **Cải tiến thuật toán RL-CMTEA thông qua cơ chế Domain-Adaptive Selection (DaS) cho Knowledge Transfer**

---

## 📋 Tóm tắt (Abstract)

Repository này trình bày một cải tiến quan trọng cho thuật toán **RL-CMTEA** (Reinforcement Learning - Constrained Multitask Evolutionary Algorithm) thông qua việc tích hợp cơ chế **Domain-Adaptive Selection (DaS)** vào quá trình Knowledge Transfer (KT).

**Vấn đề nghiên cứu:** Thuật toán RL-CMTEA gốc sử dụng random block selection để chuyển tri thức giữa các tác vụ, dẫn đến nguy cơ **Negative Transfer** - việc truyền thông tin không liên quan hoặc có hại giữa các tác vụ.

**Giải pháp đề xuất:** DaS-KT thay thế cơ chế ngẫu nhiên bằng một hệ thống học trọng số thích nghi, cho phép thuật toán tự động phát hiện và ưu tiên các chiều không gian (dimensions) có lợi cho việc truyền tri thức.

**Kết quả thực nghiệm:** Trên bộ benchmark CMT1-CMT9 (30 independent runs, 200K FES), DaS-KT đạt được:
- **83% win rate** (15/18 tasks) so với thuật toán gốc
- Cải thiện đột phá trên các bài toán khó: CMT7 T1 (~30×), CMT4 T1 (~10×)
- Đạt global optimum trên nhiều bài toán (CMT2-T2, CMT6-T2)

---

## 🎯 Động lực nghiên cứu (Motivation)

### Vấn đề của Random Knowledge Transfer

Trong thuật toán RL-CMTEA gốc, Knowledge Transfer được thực hiện qua hai bước:
1. **K-means clustering** (`divK`): Nhóm các cá thể tương đồng
2. **Random block selection** (`divD`): Chọn ngẫu nhiên các chiều để truyền

Cơ chế này có hai hạn chế chính:

**1. Negative Transfer:**
```
Task 1: f(x₁, x₂, ..., x₁₀₀) - Chỉ có x₁, x₅, x₇ liên quan đến optimum
Task 2: g(x₁, x₂, ..., x₁₀₀) - Chỉ có x₂, x₅, x₉ liên quan đến optimum

Random KT có thể truyền x₃₄, x₇₈ (nhiễu) → Phá vỡ cấu trúc tốt đang hình thành
```

**2. Không tận dụng được cấu trúc tương đồng:**
- Các tác vụ thường có một số chiều chung quan trọng (ví dụ: x₅ ở trên)
- Random selection không học được pattern này qua các thế hệ

### Tại sao DaS giải quyết được?

DaS hoạt động như một **Structure Learning Mechanism**:
- Học ma trận trọng số $W_{src→dst}$ cho mỗi cặp tác vụ
- Chiều nào giúp sinh ra cá thể con tốt → Tăng trọng số
- Chiều nào gây nhiễu → Giảm trọng số
- Kết quả: Chỉ truyền "tri thức tinh túy", loại bỏ nhiễu

---

## 🔬 Phương pháp (Methodology)

### 1. Kiến trúc tổng quan

```
RL-CMTEA Core (Preserved)
├── Dual Population (Main + Auxiliary)
├── Q-Learning for Operator Selection
├── Feasibility Priority + ε-constraint
└── Knowledge Transfer ← [DaS INTEGRATION HERE]
```

**Nguyên tắc thiết kế:** Chỉ thay đổi dimension selection trong KT, giữ nguyên toàn bộ các thành phần khác của RL-CMTEA.

### 2. DaS-KT Algorithm

#### Bước 1: Khởi tạo ma trận trọng số
```python
W[src, dst, i] = 1.0  # Uniform initialization
# W ∈ ℝ^(K×K×D) where K = số task, D = số chiều
```

#### Bước 2: Adaptive Dimension Selection
```python
# Chuẩn hóa trọng số thành phân phối xác suất
p[i] = W[src, dst, i] / Σ W[src, dst, j]

# Sampling without replacement
selected_dims = sample(p, size=divD, replace=False)
```

#### Bước 3: Knowledge Transfer & Evaluation
```python
offspring = KT(parent, selected_dims)  # Crossover trên các chiều đã chọn
fitness_offspring = evaluate(offspring)
```

#### Bước 4: Reward Computation
```python
# Reward dựa trên Feasibility Priority ranking
if offspring better than worst_parent:
    R = improvement_rate  # Positive reward
else:
    R = -penalty  # Negative reward
```

#### Bước 5: Weight Update (Exponential Multiplicative Weights)
```python
for dim in selected_dims:
    W[src, dst, dim] *= exp(η * R)
    
# Normalize to prevent overflow
W[src, dst] = clip(W[src, dst], min=1e-10, max=1e10)
W[src, dst] /= sum(W[src, dst])
```

**Tham số:**
- Learning rate: `η = 0.05`
- Warmup period: `10 generations` (để thu thập dữ liệu ban đầu)

### 3. Phân tích lý thuyết: Tại sao DaS hoạt động?

#### Định lý 1: Convergence to Optimal Dimensions (Informal)
Với giả thiết rằng tồn tại một tập con chiều $D^* \subset \{1, ..., D\}$ mà việc truyền chúng luôn cho kết quả tốt hơn, thì:

$$\lim_{t \to \infty} P(\text{select } i | i \in D^*) \to 1$$

**Chứng minh trực quan:**
- Các chiều trong $D^*$ nhận được reward dương liên tục
- Theo công thức $w_i \gets w_i \cdot \exp(\eta R)$, trọng số của chúng tăng mũ
- Các chiều ngoài $D^*$ có reward âm hoặc 0 → trọng số giảm dần
- Sau chuẩn hóa, xác suất chọn $D^*$ tiến về 1

#### Định lý 2: Robustness to Noise
DaS có khả năng chống nhiễu tốt hơn random selection vì:
- Random: $P(\text{select bad dim}) = \frac{|D \setminus D^*|}{D}$ (constant)
- DaS: $P(\text{select bad dim}) \propto \exp(-\eta \cdot t \cdot |R|)$ (exponential decay)

---

## 📊 Kết quả thực nghiệm (Experimental Results)

### Setup
- **Benchmark:** CMT1-CMT9 (Constrained Multitask Test Suite)
- **Runs:** 30 independent runs per problem
- **Budget:** 200,000 FES (Function Evaluations)
- **Comparison:** RL-CMTEA (Paper) vs RL-CMTEA + DaS (Ours)

### Tổng quan kết quả

![Performance Comparison](docs/comparison_cmt1_9_line.png)
*Hình 1: So sánh hiệu năng trên CMT1-CMT9. DaS (đường xanh) thắng áp đảo trên hầu hết các bài toán.*

### Bảng kết quả chi tiết (30-Run Mean)

| Problem | Task | Paper Mean | **DaS Mean** | Improvement | Status |
|---------|------|------------|--------------|-------------|--------|
| **CMT1** | T1 | 4.81e-17 | **3.70e-18** | ~10× | ✅ Win |
| | T2 | **7.98e-14** | 0.199 | - | ❌ Loss* |
| **CMT2** | T1 | 2.19e-09 | **1.81e-10** | ~10× | ✅ Win |
| | T2 | 5.92e-17 | **0.00** | Global Opt. | ✅ Win |
| **CMT3** | T1 | 2.28e-04 | **2.91e-08** | **~10⁴×** | ✅ Win |
| | T2 | 1.30e-03 | **6.36e-04** | +51% | ✅ Win |
| **CMT4** | T1 | 87.9 | **9.01** | **~10×** | ✅ **Huge Win** |
| | T2 | 815 | **379** | +53.5% | ✅ Win |
| **CMT5** | T1 | **4.29e-12** | 0.648 | - | ❌ Loss* |
| | T2 | 97.4 | **48.8** | +49.8% | ✅ Win |
| **CMT6** | T1 | 1.79e-08 | **1.28e-13** | **~10⁵×** | ✅ Win |
| | T2 | 6.60e-05 | **~0** | Global Opt. | ✅ Win |
| **CMT7** | T1 | 11,300 | **369** | **~30×** | ✅ **Huge Win** |
| | T2 | 129 | **62.2** | +51.8% | ✅ Win |
| **CMT8** | T1 | 16.1 | **6.00** | +62.7% | ✅ Win |
| | T2 | 91.9 | **43.1** | +53.1% | ✅ Win |
| **CMT9** | T1 | **19.4** | 8649 | - | ❌ Loss* |
| | T2 | 33,200 | **16,600** | +50.0% | ✅ Win |

**Tổng kết:** 15/18 tasks thắng (83% win rate)

*Xem phần "Failure Mode Analysis" để hiểu nguyên nhân

### Phân tích sâu: Tại sao DaS thắng?

#### Case Study 1: CMT7 - Vượt qua Local Optima Trap

CMT7 là bài toán có fitness landscape cực kỳ phức tạp với nhiều local optima sâu.

**Paper's Problem:**
- Random KT liên tục "phá vỡ" các building blocks tốt
- Quần thể bị kẹt ở local optimum với lỗi ~11,300

**DaS's Solution:**
- Học được rằng chỉ nên truyền dimensions 1, 5, 7 (giả sử)
- Bảo toàn cấu trúc gen tốt → Escape local optima
- Kết quả: Lỗi giảm xuống ~369 (**~30× improvement**)

![CMT7 Convergence](docs/convergence_CMT7.png)
*Hình 2: Đường hội tụ của CMT7. DaS (xanh) thoát khỏi plateau mà Paper bị kẹt.*

#### Case Study 2: CMT4 - Structure Discovery

CMT4 có constraint phức tạp với strong variable interaction.

**Insight từ DaS:**
- Ma trận trọng số học được cho thấy chỉ có ~10/100 dimensions thực sự quan trọng
- DaS tập trung vào các dimensions này → Giảm lỗi từ 87.9 xuống 9.01

![CMT4 Convergence](docs/convergence_CMT4.png)
*Hình 3: CMT4 convergence. DaS hội tụ nhanh hơn và sâu hơn.*

---

## ⚠️ Failure Mode Analysis

DaS không phải là "silver bullet". Chúng tôi phân tích 3 trường hợp thất bại:

### 1. CMT1-T2: Premature Convergence
**Nguyên nhân:**
- Landscape quá đơn giản, không cần structure learning
- Random KT hoạt động như regularization (diversity maintenance)
- DaS hội tụ quá sớm vào một tập dimensions → Mất diversity

**Bài học:** DaS cần thêm entropy regularization cho bài toán đơn giản.

### 2. CMT5-T1: High Variance
**Quan sát:**
- Mean: DaS kém (0.648 vs 4.29e-12)
- Best: DaS vẫn đạt optimum (4.44e-16)

**Nguyên nhân:**
- Một số runs học sai structure ban đầu → Kết quả kém
- Kéo tụt Mean nhưng Best vẫn tốt

**Bài học:** Cần cơ chế "reset" hoặc "exploration boost" khi phát hiện stagnation.

### 3. CMT9-T1: Negative Bias
**Nguyên nhân:**
- Weak inter-task similarity
- DaS "over-trust" historical rewards → Gán trọng số cao cho dimensions thực tế không tốt

**Bài học:** Cần weight decay hoặc forgetting mechanism.

---

## 🚀 Hướng phát triển (Future Work)

### DaS v2: Entropy-Regularized Adaptive Selection
```python
# Thêm entropy term vào objective
H(W) = -Σ W[i] * log(W[i])
W[i] ← W[i] * exp(η * R + λ * ∂H/∂W[i])
```
**Mục tiêu:** Duy trì diversity, khắc phục premature convergence.

### DaS v3: Forgetting Mechanism
```python
# Weight decay theo thời gian
W[i] ← α * W[i] + (1-α) * 1.0  # α = 0.95
```
**Mục tiêu:** Giảm negative bias trên bài toán non-convex.

---

## 📁 Cấu trúc Repository

```
.
├── RL_CMTEA_DaS_v2.py      # Main algorithm (DaS integrated)
├── DaS_KT.py               # DaS module
├── test_all_cmt_das.py     # Experiment script
├── docs/                   # Figures and results
│   ├── comparison_*.png
│   └── convergence_*.png
└── README.md               # This file
```

---

## 📚 Trích dẫn (Citation)

Nếu bạn sử dụng code này trong nghiên cứu, vui lòng trích dẫn:

```bibtex
@misc{das_rlcmtea2024,
  title={Domain-Adaptive Selection for Constrained Multitask Evolutionary Optimization},
  author={Your Name},
  year={2024},
  note={Research Implementation}
}
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Original RL-CMTEA algorithm from [Paper Reference]
- CMT benchmark suite
- Inspiration from Domain-Adaptive Selection literature
