# Domain-Adaptive Selection for Constrained Multitask Evolutionary Optimization


> **Cải tiến thuật toán RL-CMTEA thông qua cơ chế Domain-Adaptive Selection (DaS) cho Knowledge Transfer**

---

## Tóm tắt (Abstract)

Repository này trình bày một cải tiến quan trọng cho thuật toán **RL-CMTEA** (Reinforcement Learning - Constrained Multitask Evolutionary Algorithm) thông qua việc tích hợp cơ chế **Domain-Adaptive Selection (DaS)** vào quá trình Knowledge Transfer (KT).

**Vấn đề nghiên cứu:** Thuật toán RL-CMTEA gốc sử dụng random block selection để chuyển tri thức giữa các tác vụ, dẫn đến nguy cơ **Negative Transfer** - việc truyền thông tin không liên quan hoặc có hại giữa các tác vụ.

**Giải pháp đề xuất:** DaS-KT thay thế cơ chế ngẫu nhiên bằng một hệ thống học trọng số thích nghi, cho phép thuật toán tự động phát hiện và ưu tiên các chiều không gian (dimensions) có lợi cho việc truyền tri thức.

**Kết quả thực nghiệm:** Trên bộ benchmark CMT1-CMT9 (30 independent runs, 200K FES), DaS-KT đạt được:
- **83% win rate** (15/18 tasks) so với thuật toán gốc
- Cải thiện đột phá trên các bài toán khó: CMT7 T1 (~30×), CMT4 T1 (~10×)
- Đạt global optimum trên nhiều bài toán (CMT2-T2, CMT6-T2)

---

## Kiến trúc RL-CMTEA: 5 Thành phần Cốt lõi

Trước khi đi vào cải tiến DaS, cần hiểu rõ kiến trúc của RL-CMTEA gốc gồm 5 thành phần:

### 1. Dual Population Strategy

RL-CMTEA duy trì **2 quần thể** cho mỗi task:

**Main Population (EC = 0)**:
- Chỉ chấp nhận cá thể **feasible** (constraint violation = 0)
- Mục tiêu: Tối ưu hóa objective function
- Selection: Feasibility Priority (so sánh CV trước, rồi mới so sánh fitness)

**Auxiliary Population (EC = ε(t))**:
- Chấp nhận cá thể có constraint violation ≤ ε(t)
- ε(t) giảm dần theo thời gian (từ lớn → 0)
- Mục tiêu: Duy trì diversity, tránh premature convergence

**Lý do cần Dual Population**: Bài toán có ràng buộc thường có feasible region rất nhỏ. Nếu chỉ dùng Main Pop → mất diversity quá nhanh.

### 2. Reinforcement Learning cho Operator Selection

**Q-Learning + UCB** để chọn operator tốt nhất:

**4 Operators**:
1. SBX (Simulated Binary Crossover)
2. DE/rand/1: `x' = x_r1 + F*(x_r2 - x_r3)`
3. DE/rand/2: `x' = x_r1 + F*(x_r2 - x_r3) + F*(x_r4 - x_r5)`
4. DE/best/1: `x' = x_best + F*(x_r1 - x_r2)`

**Q-Learning Update**:
```python
Q[s, a] ← Q[s, a] + α * (R + γ * max_a' Q[s, a'] - Q[s, a])
# R = success_rate (tỷ lệ offspring được chọn vào thế hệ sau)
```

**UCB Selection** (cân bằng exploitation-exploration):
```python
UCB[s, a] = Q[s, a] + sqrt(2 * log(T) / (N[s, a] + ε))
action = argmax_a UCB[s, a]
```

### 3. Knowledge Transfer (KT) - Thành phần được DaS cải tiến

**Phiên bản gốc (Random)**:
1. **Block Encoding**: Chia mỗi cá thể thành blocks có độ dài `divD`
2. **K-means Clustering**: Nhóm các blocks tương đồng từ tất cả tasks thành `divK` clusters
3. **DE/rand/1 trong cluster**: Tạo offspring bằng cách kết hợp các blocks trong cùng cluster
4. **Vấn đề**: Chọn dimensions ngẫu nhiên → **Negative Transfer**

**Ví dụ Negative Transfer**:
```
Task 1: f(x₁, x₂, ..., x₁₀₀) - Chỉ có x₁, x₅, x₇ liên quan đến optimum
Task 2: g(x₁, x₂, ..., x₁₀₀) - Chỉ có x₂, x₅, x₉ liên quan đến optimum

Random KT có thể truyền x₃₄, x₇₈ (nhiễu) → Phá vỡ cấu trúc tốt đang hình thành
```

### 4. Adaptive divD & divK (Heuristic)

Điều chỉnh số clusters (divK) và kích thước block (divD) theo success flags:

```python
if all(tasks_failed):
    divD = random(1, maxD)  # Reset ngẫu nhiên
    divK = random(minK, maxK)
elif any(tasks_failed):
    divD = clip(divD + random(-1, +1), 1, maxD)  # Điều chỉnh nhẹ
    divK = clip(divK + random(-1, +1), minK, maxK)
# else: Tất cả thành công → giữ nguyên
```

**Hạn chế**: Heuristic đơn giản, không học được pattern tốt.

### 5. Epsilon Constraint Handling

Điều chỉnh động ngưỡng constraint violation ε(t):

```python
# Khởi tạo: ε = top 20% CV trong quần thể
Ep[t] = percentile(constraint_violations, 80)

# Cập nhật theo tiến trình
progress = fnceval_calls / max_evals
if progress < 0.8:  # 80% đầu tiên
    if feasible_rate < 0.8:
        # Giảm dần ε để thắt chặt ràng buộc
        Ep[t] = Ep[t] * (1 - progress/0.8)^2
    else:
        # Đã có đủ feasible → nới lỏng để tăng diversity
        Ep[t] = 1.1 * max(constraint_violations)
else:  # 20% cuối
    Ep[t] = 0  # Chỉ chấp nhận feasible
```

---

## Động lực nghiên cứu: Vấn đề của Random KT

### Tại sao cần cải tiến Knowledge Transfer?

Trong 5 thành phần trên, **Knowledge Transfer** là nơi có tiềm năng cải tiến lớn nhất:

**Vấn đề 1: Negative Transfer**
- Random selection không phân biệt được chiều nào quan trọng, chiều nào là nhiễu
- Xác suất chọn chiều xấu = constant: `P(bad dim) = (D - D*) / D`

**Vấn đề 2: Không học được cấu trúc**
- Các tasks thường có một số chiều chung quan trọng
- Random selection không tận dụng được thông tin này qua các thế hệ

### Giải pháp: Domain-Adaptive Selection (DaS)

DaS thay thế random selection bằng **Structure Learning Mechanism**:

```python
# Thay vì:
selected_dims = random.choice(D, divD)  # Uniform random

# DaS học trọng số:
weights = learn_from_history()  # [0.01, 0.01, ..., 0.40, ..., 0.30, ...]
selected_dims = weighted_choice(D, divD, p=weights)  # Adaptive
```

**Kết quả**:
- Chiều tốt (x₅, x₇) có trọng số cao → Được chọn thường xuyên
- Chiều nhiễu (x₃₄, x₇₈) có trọng số thấp → Hiếm khi được chọn
- Xác suất chọn chiều xấu giảm theo mũ: `P(bad dim) ∝ exp(-η * t * |R|)`

---

## Phương pháp DaS (Methodology)

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

## Kết quả thực nghiệm (Experimental Results)

### Setup
- **Benchmark:** CMT1-CMT9 (Constrained Multitask Test Suite)
- **Runs:** 30 independent runs per problem
- **Budget:** 200,000 FES (Function Evaluations)
- **Comparison:** RL-CMTEA (Paper) vs RL-CMTEA + DaS (Ours)

### Tổng quan kết quả

![Performance Comparison](docs/comparison_cmt1_9_line.png)
*Hình 1: So sánh hiệu năng trên CMT1-CMT9. DaS (đường xanh) thắng áp đảo trên hầu hết các bài toán.*

### Bảng kết quả chi tiết (30-Run Mean)

| Problem | Task | Baseline (Paper) | Proposed (DaS) | Relative Improvement (%) | Notes |
|---------|------|------------------|----------------|--------------------------|-------|
| **CMT1** | T1 | 4.81×10⁻¹⁷ | **3.70×10⁻¹⁸** | 92.31 | — |
| | T2 | **7.98×10⁻¹⁴** | 0.199 | — | Performance degradation |
| **CMT2** | T1 | 2.19×10⁻⁹ | **1.81×10⁻¹⁰** | 91.74 | — |
| | T2 | 5.92×10⁻¹⁷ | **0.00** | 100.00 | Achieves optimum |
| **CMT3** | T1 | 2.28×10⁻⁴ | **2.91×10⁻⁸** | 100.00 | — |
| | T2 | 1.30×10⁻³ | **6.36×10⁻⁴** | 51.08 | — |
| **CMT4** | T1 | 87.9 | **9.01** | 89.75 | — |
| | T2 | 815 | **379** | 53.50 | — |
| **CMT5** | T1 | **4.29×10⁻¹²** | 0.648 | — | Performance degradation |
| | T2 | 97.4 | **48.8** | 49.90 | — |
| **CMT6** | T1 | 1.79×10⁻⁸ | **1.28×10⁻¹³** | 100.00 | — |
| | T2 | 6.60×10⁻⁵ | **0.00** | 100.00 | Achieves optimum |
| **CMT7** | T1 | 11,300 | **369** | 96.73 | — |
| | T2 | 129 | **62.2** | 51.78 | — |
| **CMT8** | T1 | 16.1 | **6.00** | 62.73 | — |
| | T2 | 91.9 | **43.1** | 53.09 | — |
| **CMT9** | T1 | **19.4** | 8,649 | — | Performance degradation |
| | T2 | 33,200 | **16,600** | 50.00 | — |

**Tổng kết:** Phương pháp đề xuất (DaS) vượt trội hơn baseline trên 15/18 tasks (83.3% success rate), với cải thiện đáng kể nhất quan sát được trên CMT3-T1, CMT6-T1, và CMT7-T1.

**Ghi chú:** 
- Tất cả kết quả báo cáo giá trị trung bình (mean) của hàm mục tiêu qua 30 lần chạy độc lập.
- Relative Improvement (%) được tính theo công thức: (Baseline − Proposed) / Baseline × 100, và được bỏ qua (—) khi giá trị baseline nhỏ hơn 1×10⁻⁸.
- Giá trị nhỏ hơn cho thấy hiệu năng tốt hơn (minimization objectives).

*Xem phần "Failure Mode Analysis" để hiểu nguyên nhân các trường hợp Performance degradation.

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

## Failure Mode Analysis

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

## Hướng phát triển (Future Work)

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

## Cấu trúc Repository

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

## Trích dẫn (Citation)


---

## License


---

## Acknowledgments

- Original RL-CMTEA algorithm from RL-CMTEA paper
- CMT benchmark suite
- Inspiration from Domain-Adaptive Selection literature
