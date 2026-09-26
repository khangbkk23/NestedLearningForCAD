**Meta-NATH (NeoViT-CAD)**
**Hệ Thống Continual Anomaly Detection (CAD) Dựa Trên Nested Learning Paradigm**
**Ngày:** 04/04/2026  

---

## Tóm tắt Nghiên cứu

Meta-NATH (NeoViT-CAD) là hệ thống **Continual Anomaly Detection (CAD)** được thiết kế theo triết lý **Nested Learning paradigm** (Google Research, NeurIPS 2025 – "\[Nested Learning] NL.pdf"). Hệ thống đạt được ba mục tiêu cốt lõi:

- **Test-Time Adaptation realtime** trên edge device ví dụ: 4GB VRAM (RTX 3050 Ti + 16GB RAM).
- **Forgetting rate bounded** mà không cần generative replay hay EWC - thay vào đó dùng **Incremental Coreset (CADIC)** để giữ tối đa ~1000 embeddings đại diện cho toàn bộ lịch sử task.
- **Backbone tiến hóa định kỳ** (N2B-NC) với unified memory bank mà vẫn giữ kiến thức nền tảng ổn định.

Kiến trúc là một **Nested System of Associative Memories (NSAM)** với ba tầng rõ ràng:

- **Nhà Thông Thái** = Frozen DINOv3 backbone (self-supervised, register tokens).
- **Sổ Nháp** = TITANS Memory (Fast Memory - self-modifying qua Delta Rule trong forward pass).
- **Tủ Hồ Sơ** = Slow Memory + NSP2 (Null-Space Projection) + CBP (Continual Backprop) + Subspace Recycling + **Incremental Coreset (CADIC)** + **Pixel-level Anomaly Decoder** (lấy cảm hứng từ ReplayCAD).

Toàn bộ pipeline tuân thủ **3 nhịp sinh học** (Phản xạ → Tiêu hóa → Tiến hóa) và được tối ưu hóa để chạy thực tế trên hardware thực tế (Phase 1 trên laptop edge, Phase 2-3 trên cloud). Tất cả quyết định thiết kế đều xuất phát trực tiếp từ **ConversationLog.txt** (evolution từ Dual-LoRA → TITANS, EWC/Replay → NSP2+CBP+N2B-NC, ViT-5 → DINOv3, v.v.)

> **Lưu ý quan trọng về triết lý thiết kế:**  
> - **CADIC** (Incremental Coreset + Unified Memory Bank) là xương sống của Slow Memory, thay thế hoàn toàn mọi hình thức generative replay.  
> - **ReplayCAD** được giữ lại CHỈ như baseline so sánh và mượn tư duy Pixel-level Anomaly Decoder. Phần Generative Diffusion bị bác bỏ hoàn toàn.  
> - ***CAD_Fundamental (SCALE/Super-Resolution Compressed Replay) bị loại bỏ hoàn toàn khỏi kiến trúc vì với Incremental Coreset chỉ giữ ~1000 embeddings, không cần thêm layer nén/giải nén ảnh gây over-engineering và lãng phí GPU.***

**BẢNG  COMPONENT**

| Thành phần       | Công nghệ cốt lõi                          | Đóng góp toán học / Kỹ thuật quan trọng                          |
|------------------|--------------------------------------------|------------------------------------------------------------------|
| Backbone         | Frozen DINOv3                              | Features general, self-supervised, register tokens               |
| Fast Memory      | TITANS (Associative)                       | Delta Rule trong forward pass (O(d²), no_grad)                   |
| Plasticity       | CBP + Smooth-Leaky                         | Tái sinh neuron dựa trên utility score                           |
| Stability        | NSP2 + Subspace Recycling                  | Bounded forgetting, tránh null space collapse                    |
| **Slow Memory**  | **CADIC Incremental Coreset**              | **Unified Memory Bank, max ~1000 entries, no task fragmentation** |
| Evolution        | N2B-NC                                     | Distillation trực giao chỉ trên vài layer cuối                   |
| **Anomaly Map**  | **Pixel-level Anomaly Decoder**            | **Multi-scale patch features → pixel-level anomaly map**         |
| Output Scoring   | Nearest-Neighbor trong Coreset             | Anomaly score = min dist(x, C), image-level AUROC + pixel-level AP |

---

## 1. Bối cảnh Đặt vấn đề và Sự Dịch chuyển Hệ hình

Từ ConversationLog.txt, hành trình của dự án bắt đầu với việc phân loại 72 papers (Khu 1–5) và loại bỏ hàng loạt phương pháp cũ (iCaRL, L2P, EWC, Replay Buffer, Dual-LoRA tĩnh, v.v.) vì chúng không giải quyết được **Loss of Plasticity** (Dohare et al., Nature 2024) và **Catastrophic Forgetting** đồng thời.

Hành trình của dự án tập trung giải quyết bài toán Continual Anomaly Detection (CAD). Các phương pháp CAD hiện tại đang mắc kẹt giữa hai thái cực:

- **Generative Diffusion Replay (ReplayCAD):** Cực kỳ ngốn VRAM, chạy chậm, dễ sinh ảo giác (Hallucination), không phù hợp edge 4GB VRAM.
- **Class-specific sub-memory banks (DNE, UCAD, DFM):** Gặp vấn đề phân mảnh bộ nhớ (Fragmented memory) theo từng task, không đạt được task-agnostic continual learning thực sự.
- ***SCALE / Super-Resolution Compressed Replay (CAD_Fundamental): Nén ảnh để tiết kiệm replay memory — bị bác bỏ vì với Incremental Coreset chỉ cần embeddings, không cần lưu raw images hay compress/decompress gì cả.***

Để giải quyết, hệ thống Meta-NATH CAD thiết lập các quyết định then chốt sau:

**Quyết định then chốt đã được log ghi nhận rõ ràng:**
- **Backbone:** Frozen DINOv3 (giữ lại các chi tiết patch-level, register tokens làm sạch nhiễu).
- **Fast Memory:** TITANS (Delta Rule) giúp Test-time Adaptation với biến động môi trường.
- **Slow Memory & Forgetting:** Incremental Coreset theo triết lý CADIC (Unified Memory Bank, bounded size ~1000 entries, incremental update) + NSP2 (Bounded forgetting bằng toán học, không cần sinh ảnh giả, không cần nén ảnh).
- **Anomaly Output:** Thay vì phân loại nhãn, dùng **Pixel-level Anomaly Decoder** hợp nhất multi-scale patch features → xuất Pixel-level Anomaly Map + Image-level Score. Anomaly score tính bằng nearest-neighbor trong Coreset (theo CADIC inference, Eq. 8-9).
- **Plasticity:** CBP + Smooth-Leaky Activation.
- **Evolution:** N2B-NC thuần Nested Learning.

Kết quả là một hệ thống không thỏa hiệp giữa stability và plasticity, khả thi trên edge + cloud.

---

## 2. Nền tảng Toán học của Nested Learning (NSAM)

**Delta Rule (TITANS – Fast Memory):**  
$$M_t = (1 - \alpha)M_{t-1} + \eta_t \cdot (v_t - M_{t-1}k_t)k_t^T$$  
với **α = 0.9** (decay rate), **η₀ = 0.01** (base learning rate), η_t = η₀ / (1 + surprise_t).  
Bắt buộc thực hiện trong `with torch.no_grad()`.

**LEJEPA Surrogate Loss:**  
$$L = \| \text{pred}(z_{\text{context}}) - \text{stop\_grad}(z_{\text{target}}) \|^2$$

**ACC Gating:**  
$$ACC = \cos_{\text{sim}}(z_{\text{updated}}, z_{\text{original}}) - H_{\text{term}}$$  
với $H_{\text{term}} = \|z_{\text{updated}} - z_{\text{original}}\|^2 / d$.

**NSP2:**  
$$P_{\text{null}} = I - V_{\text{task}} V_{\text{task}}^T$$  
$$\Delta W_{\text{safe}} = P_{\text{null}} \cdot \Delta W$$

**CBP Utility & Reinit:**  
$$\text{utility}_i = |a_i| / (\text{mean}_a + \epsilon)$$  
$$w_{\text{new}} = \text{projector.project}(\text{randn\_like}(w_{\text{dead}}) \times 0.02)$$  
**τ_cbp = 0.01** (utility_i < 0.01 → reset). Monitoring: log dead_neuron_ratio. Nếu ratio > 30% → tăng τ_cbp lên 0.05.

**N2B-NC Loss:**  
$$\mathcal{L}_{\text{N2B-NC}} = \mathcal{L}_{\text{distill}} + \lambda_1 \mathcal{L}_{\text{TITANS-TTT}}$$  
với **λ₁ = 0.1**.  
$$\mathcal{L}_{\text{distill}} = \text{F.mse\_loss}(z_{\text{backbone}}, z_{\text{target}})$$

**Smooth-Leaky Activation:**  
$$f(x) = x \quad (x > 0)$$  
$$f(x) = \alpha_{\text{sl}} x + \beta_{\text{sl}} \sin(x) \quad (x \leq 0) \quad (\alpha_{\text{sl}} \approx 0.1, \beta_{\text{sl}} \approx 0.05)$$

---

## 2.1 Toán học Inference CAD — Anomaly Scoring (CADIC-based)

> **[MỚI — Bổ sung từ CADIC paper]**  
> Phần này thay thế hoàn toàn cách tính anomaly score kiểu classification cũ. Toàn bộ inference dựa trên nearest-neighbor matching trong Incremental Coreset C.

**Pixel-level anomaly score:**  
$$s_{\text{pix}} = \min_{c \in C} \|x - c\|_2$$

**Image-level anomaly score (theo PatchCore/CADIC Eq. 8-9):**  
$$s_{\text{img}} = \left(1 - \frac{\exp \|x - c^*\|_2}{\sum_{c \in N_b(c^*)} \exp \|x - c\|_2}\right) \cdot s^*_{\text{pix}}$$

với $s^*_{\text{pix}}$ = largest anomaly score among all patch embeddings of the test image, $c^*$ = corresponding coreset embedding, $N_b(c^*)$ = b nearest neighbors of $c^*$ in C.

**Pixel-level Anomaly Map:**  
Anomaly scores tính cho từng patch embedding, sau đó upsample về spatial dimensions của input image. Anomalous regions được xác định qua binarization. *(Cấu trúc Anomaly Decoder lấy cảm hứng từ ReplayCAD — giữ tư duy multi-scale spatial feature, bỏ phần diffusion generation.)*

---

## 3. Phẫu thuật Kiến trúc Hệ thống Meta-NATH Unleashed

**3.1. Backbone:** Frozen DINOv3 (load từ facebookresearch/dinov3).  
Output format: backbone(x) trả về dict. Bắt buộc dùng: `z_backbone = backbone(x)["x_norm_clstoken"]` (shape [batch, d=768] cho ViT-B/14). Dùng `["x_norm_patchtokens"]` cho patch-level analysis (cần thiết để xây dựng Pixel-level Anomaly Map).  
Lý do chọn DINOv3: self-supervised, register tokens sẵn, features general hơn rất nhiều.

```python
# Load chính xác backbone
backbone = torch.hub.load("facebookresearch/dinov3", "dinov3_vitb14", pretrained=True)
backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False
```

**3.2. Fast Memory:** TITANS MAC (self-modifying trong forward pass).

**3.3. Slow Memory + Stability: Incremental Coreset (CADIC) + NSP2 + Subspace Recycling + CBP**

> **[CỐT LÕI — Đây là trái tim của hệ thống CAD]**

Tuyệt đối **không lưu phân mảnh theo từng task**. Sử dụng một **Unified Memory Bank duy nhất** cho toàn bộ pipeline.

**Lưu list of tuples:** `(image_tensor [C, H, W], patch_embeddings [N_patch, d], cls_embedding [d], utility_score float)`  
*(Lưu cả patch_embeddings để tính Pixel-level Anomaly Map trong inference)*

**Interface tối thiểu cần có:**

- `update_coreset(image, patch_embs, cls_emb)`: Thay vì add bừa bãi, tính khoảng cách `d_max(X, C)`. Nếu vector mới xa nhất, thay thế mẫu có khoảng cách gần nhất trong Coreset bằng vector mới này (Eq. 1-6 CADIC).
- `get_top_k_by_utility(k)`: Trả về list k images và embeddings có utility cao nhất để làm "Anchor points" cho N2B-NC và tính LEJEPA loss.
- `update_utility(idx, new_score)`: Cập nhật utility sau quá trình quét CBP.
- `compute_anomaly_score(x_patch_embs)`: Tính pixel-level scores và image-level score theo Eq. 8-9 CADIC.

**Max capacity:** ~1000 entries (tune theo VRAM 4GB). Đảm bảo Bounded Memory.  
**Không cần nén ảnh (SCALE/SR):** Coreset lưu `patch_embeddings` [N_patch × d] per entry để phục vụ Pixel-level Anomaly Decoder. Breakdown dung lượng chính xác:

| Thành phần lưu trữ | Kích thước per entry | Tổng 1000 entries |
|--------------------|----------------------|-------------------|
| `cls_embedding` [d=768] | 768 × 4 bytes = 3 KB | 3 MB |
| `patch_embeddings` [256 × 768] | 256 × 768 × 4 bytes ≈ 786 KB | **≈ 786 MB** |
| `utility_score` (float) | 4 bytes | ~0.004 MB |
| **Tổng** | **≈ 789 KB/entry** | **≈ 789 MB** |

> **Tổng dung lượng Coreset ≈ 786 MB** — vẫn hoàn toàn an toàn và nằm gọn trong RTX 3050 Ti 4GB VRAM, nhờ không lưu raw images (~3MB/ảnh × 1000 = 3GB) và không tải Diffusion weights (~4–8GB).  
> *(Phép tính: $1000 \times 256 \times 768 \times 4 \text{ bytes} = 786{,}432{,}000 \text{ bytes} \approx 786 \text{ MB}$)*

**Subspace Recycling (van xả áp lực NSP2):** Nếu dim(Null_T) < 64 sau SVD → kích hoạt Subspace Recycling: giải phóng các chiều ít quan trọng nhất (σ_i nhỏ nhất) để tái sử dụng làm không gian học mới. Fallback cascade: dim < 64 → 32 → 16 (log cảnh báo rõ tại mỗi bước). Nếu vẫn < 16 → dừng N2B-NC cycle đó và skip.

**CADIC Incremental Update Algorithm (Eq. 1-6):**

```python
def update_coreset(self, x_batch_embs, C):
    """
    x_batch_embs: [N, d] embeddings của batch hiện tại
    C: current coreset embeddings [M, d]
    """
    # Tính distance matrix hiệu quả bằng matrix ops (Eq. 7 CADIC)
    # D = sqrt(diag(X@X.T).T * 1_{1xm} + 1_{nx1} * diag(C@C.T) - 2*X@C.T)
    while True:
        # Eq. 1: d_max(X, C) = max_{x in X} min_{c in C} ||x - c||_2
        dists_to_C = torch.cdist(x_batch_embs, C)  # [N, M]
        min_dists = dists_to_C.min(dim=1).values    # [N]
        x_star_idx = min_dists.argmax()             # Eq. 2
        d_max = min_dists[x_star_idx]

        # Eq. 4-5: |C|_min = min_{c1,c2 in C} d(c1,c2)
        C_dists = torch.cdist(C, C)
        C_dists.fill_diagonal_(float('inf'))
        c_min_idx = C_dists.min().argmin()  # index of c1
        C_min = C_dists.min()

        # Eq. 6: condition
        if d_max > C_min:
            # Replace c1 with x_star
            C[c_min_idx // C.shape[0]] = x_batch_embs[x_star_idx]
        else:
            break
    return C
```

**3.4. Output: Pixel-level Anomaly Decoder**

> **[MỚI — Thay thế CSR]**  
> Lấy cảm hứng từ tư duy spatial feature của ReplayCAD (bỏ phần diffusion). CSR sparse autoencoder được loại bỏ khỏi pipeline chính do không phù hợp với bài toán anomaly localization.

```python
class AnomalyDecoder(nn.Module):
    """
    Multi-scale patch feature aggregator → pixel-level anomaly map.
    So sánh patch-vs-patch (KHÔNG phải patch-vs-cls) để đúng không gian ngữ nghĩa.
    Gọi: coreset.get_all_patch_embs() để lấy đúng tensor [M*256, 768].
    """
    def forward(self, patch_embs, coreset_patch_embs):
        """
        patch_embs:          [N_patch, 768]     — patch embeddings của ảnh test (VD: [256, 768])
        coreset_patch_embs:  [M*N_patch, 768]   — flatten toàn bộ patch của Coreset (VD: [256000, 768])
                             → Lấy từ coreset.get_all_patch_embs(), KHÔNG truyền self.embeddings (cls)
        """
        with torch.no_grad():  # Bắt buộc: tránh cache đạo hàm gây memory leak
            dists = torch.cdist(patch_embs, coreset_patch_embs)  # [N_patch, M*N_patch] ~262MB tạm thời
            s_pix = dists.min(dim=1).values                      # [N_patch]
            del dists  # GIẢI PHÓNG VRAM NGAY — không giữ 262MB qua inference tiếp theo
            # torch.cuda.empty_cache()  # Chỉ bật nếu thật sự OOM — làm chậm inference

        H_patch = W_patch = int(s_pix.shape[0] ** 0.5)
        anomaly_map = s_pix.reshape(H_patch, W_patch)
        anomaly_map = F.interpolate(
            anomaly_map.unsqueeze(0).unsqueeze(0),
            size=(224, 224), mode='bilinear'
        ).squeeze()
        return anomaly_map, s_pix
```

**3.5. Evolution:** N2B-NC (chỉ unfreeze 2–4 layer cuối của DINOv3).

**Predictor head (dùng trong LEJEPA loss của N2B-NC):**  
`predictor = nn.Linear(d, d, bias=False)` (d = 768 cho DINOv3-B/14).

**CBP trên backbone (`_apply_cbp_to_backbone`):** Quét activation rates của 2–4 layers đã unfreeze, reset neurons có utility_i < τ_cbp = 0.01, và áp dụng NSP2 projection ngay sau khi re-init.

**Memory dimension:**  
- Edge (RTX 3050 Ti 4GB): DINOv3-B/14 → d = 768.  
- Cloud: DINOv3-L/14 → d = 1024.

---

## 4. Tương tác Hệ thống & Pipeline 3 Nhịp

**Nhịp 1 (Phản xạ – Giai đoạn 1):** Input → Frozen DINOv3 → TITANS (Delta Rule + LEJEPA) → TTT realtime.  
*Thêm: Extract cả `x_norm_patchtokens` để chuẩn bị cho Pixel-level Anomaly Map ở Nhịp 2.*

**Nhịp 2 (Tiêu hóa – Giai đoạn 2):** ACC Gating → NSP2 projection → CBP neuron reset → Subspace Recycling → **CADIC update_coreset** (cập nhật Unified Memory Bank với patch embeddings mới).

**Nhịp 3 (Tiến hóa – Giai đoạn 3):** N2B-NC: Trích xuất z_target từ Coreset top-k (theo utility) → Distill vào backbone (chỉ last layers) + NSP2 + CBP. *Không cần generative replay. Không cần SCALE compression.*

---

### Pseudocode Thực Tế

**TITANSMemory class (bắt buộc):**

```python
class TITANSMemory:
    def __init__(self, d=768):
        self.M = torch.zeros(d, d, device="cuda" if torch.cuda.is_available() else "cpu")
```

**Phase 1: TTTEngine (Test-Time Adaptation):**

```python
class TTTEngine:
    def __init__(self, core_model, tau_acc=0.25, eta0=0.01, alpha=0.9):
        self.core = core_model
        self.tau_acc = tau_acc
        self.eta0 = eta0
        self.alpha = alpha
        self.gating = ACCGating(tau=tau_acc)

    def process_stream(self, x_batch):
        with torch.no_grad():
            # Extract cls token + patch tokens
            backbone_out = self.core.backbone(x_batch)
            z_backbone = backbone_out["x_norm_clstoken"]       # [batch, d]
            z_patches = backbone_out["x_norm_patchtokens"]     # [batch, N_patch, d]

            k_t = z_backbone
            v_t = z_backbone
            pred = (self.core.memory.M @ k_t.T).T             # [batch, d]
            surprise_vec = v_t - pred
            surprise_scalar = surprise_vec.norm(dim=-1).mean()

            eta_t = self.eta0 / (1 + surprise_scalar.item())
            update = torch.einsum('bi,bj->ij', surprise_vec, k_t) / k_t.shape[0]

            self.core.memory.M.data = (
                (1 - self.alpha) * self.core.memory.M.data + eta_t * update
            )
            self.core.memory.M.data = torch.clamp(self.core.memory.M.data, -5.0, 5.0)

        z_updated = (self.core.memory.M @ z_backbone.T).T     # [batch, d]
        approved = self.gating.should_consolidate(z_updated, z_backbone)
        return z_updated, z_patches, approved
```

**Phase 2: ACCGating class:**

```python
class ACCGating:
    def __init__(self, tau=0.25):
        self.tau = tau

    def should_consolidate(self, z_updated, z_original):
        cos_sim = F.cosine_similarity(z_updated, z_original, dim=-1).mean()
        h_term = (z_updated - z_original).norm(dim=-1).mean() / z_updated.shape[-1]
        acc = (cos_sim - h_term).item()
        return acc > self.tau
```

**Phase 2 Extension: CADIC Coreset Update (sau ACCGating approve):**

```python
class CADICCoreset:
    def __init__(self, max_size=1000, d=768):
        self.max_size = max_size
        self.d = d
        self.embeddings = []    # list of cls embeddings [d]
        self.patch_embs = []    # list of patch embeddings [N_patch, d]
        self.images = []        # raw images — chỉ giữ nếu cần N2B-NC distillation
        self.utilities = []

    def update_coreset(self, image, patch_embs, cls_emb):
        """Incremental coreset update theo CADIC Eq. 1-6"""
        if len(self.embeddings) < self.max_size:
            self.embeddings.append(cls_emb)
            self.patch_embs.append(patch_embs)
            self.images.append(image)
            self.utilities.append(1.0)
            return

        C = torch.stack(self.embeddings)   # [M, d]
        X = cls_emb.unsqueeze(0)           # [1, d]

        # d_max: distance from new embedding to nearest coreset member
        dists_to_C = torch.cdist(X, C)
        d_max = dists_to_C.min().item()

        # |C|_min: minimum pairwise distance in coreset
        C_dists = torch.cdist(C, C)
        C_dists.fill_diagonal_(float('inf'))
        c_min_val = C_dists.min().item()
        c_min_idx = C_dists.argmin().item() // C.shape[0]

        if d_max > c_min_val:
            # Replace closest pair member with new embedding
            self.embeddings[c_min_idx] = cls_emb
            self.patch_embs[c_min_idx] = patch_embs
            self.images[c_min_idx] = image
            self.utilities[c_min_idx] = 1.0

    def compute_anomaly_score(self, patch_embs_test, b=2):
        """CADIC inference: Eq. 8-9
        b: số lân cận gần nhất cho Neighborhood Softmax (Eq. 9) — KHÔNG dùng .sum() toàn M
        """
        C = torch.stack(self.embeddings)  # [M, d]
        # Pixel-level scores
        dists = torch.cdist(patch_embs_test, C)   # [N_patch, M]
        s_pix_all = dists.min(dim=1).values        # [N_patch]
        # Image-level score
        s_star_idx = s_pix_all.argmax()
        x_star = patch_embs_test[s_star_idx]
        c_star_idx = dists[s_star_idx].argmin()
        # Neighborhood softmax weighting — chỉ b lân cận gần nhất (Eq. 9)
        nb_dists = torch.cdist(x_star.unsqueeze(0), C).squeeze()  # [M]
        top_k_dists, _ = torch.topk(nb_dists, k=b, largest=False) # [b] — gần nhất
        weight = 1 - (torch.exp(nb_dists[c_star_idx]) / torch.exp(top_k_dists).sum())
        s_img = weight * s_pix_all[s_star_idx]
        return s_img.item(), s_pix_all

    def get_top_k_by_utility(self, k):
        indices = sorted(range(len(self.utilities)), key=lambda i: self.utilities[i], reverse=True)[:k]
        imgs = torch.stack([self.images[i] for i in indices])
        embs = torch.stack([self.embeddings[i] for i in indices])
        return imgs, embs

    def get_all_patch_embs(self):
        """Trả về [M * N_patch, d] — flatten toàn bộ patch embeddings cho AnomalyDecoder.
        BẮT BUỘC dùng hàm này thay vì truy cập self.patch_embs trực tiếp (vốn là Python List).
        """
        if not self.patch_embs:
            return None
        return torch.cat(self.patch_embs, dim=0)  # [M * 256, 768]
```

**Phase 3 – Evolution: NestedBackboneConsolidator (N2B-NC):**

```python
class NestedBackboneConsolidator:
    def __init__(self, core_model, projector, coreset: CADICCoreset, predictor, lr_n2bnc=1e-5, cbp_threshold=0.01):
        self.core = core_model
        self.projector = projector
        self.coreset = coreset          # CADICCoreset — thay SlowMemory
        self.predictor = predictor
        self.lr_n2bnc = lr_n2bnc
        self.cbp_threshold = cbp_threshold
        self.backbone_state_before = None

    def execute_global_consolidation(self):
        # x_synthetic = top-k raw images từ Coreset theo utility cao nhất
        x_synthetic, z_target = self.coreset.get_top_k_by_utility(k=32)

        self.backbone_state_before = copy.deepcopy(self.core.backbone.state_dict())

        # Explicit unfreeze last layers only
        for name, param in self.core.backbone.named_parameters():
            param.requires_grad = ("blocks.10" in name or "blocks.11" in name or "norm" in name)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.core.backbone.parameters()),
            lr=self.lr_n2bnc, weight_decay=0.01
        )

        backbone_out = self.core.backbone(x_synthetic)
        z_backbone = backbone_out["x_norm_clstoken"]

        # LEJEPA loss
        ctx_mask = torch.rand(x_synthetic.shape[0]) > 0.5
        z_context = z_backbone[ctx_mask]
        z_tgt = z_target[~ctx_mask] if (~ctx_mask).any() else z_target
        le_jepa_loss = F.mse_loss(
            self.predictor(z_context.mean(0, keepdim=True)),
            z_tgt.mean(0, keepdim=True).detach()
        )

        loss = F.mse_loss(z_backbone, z_target) + 0.1 * le_jepa_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.core.backbone.parameters(), max_norm=1.0)
        for param in self.core.backbone.parameters():
            if param.grad is not None:
                param.grad = self.projector.project(param.grad)

        optimizer.step()
        self._apply_cbp_to_backbone()

        # Re-freeze
        for param in self.core.backbone.parameters():
            param.requires_grad = False

        # Drift check
        z_after = self.core.backbone(x_synthetic)["x_norm_clstoken"].detach()
        drift = 1 - F.cosine_similarity(z_backbone.detach(), z_after, dim=-1).mean()
        if drift > 0.05:
            self.core.backbone.load_state_dict(self.backbone_state_before)
            self.lr_n2bnc *= 0.1
            logging.warning(f"Drift {drift:.4f} > 0.05 → rollback & LR *= 0.1")
            return False
        return True
```

---

## 5. Implementation Constraints

1. SVD/NSP2: Tính **một lần cuối task**, cache.
2. TITANS update: `with torch.no_grad()`.
3. CBP reinit: Phải project qua $P_{\text{null}}$. τ_cbp = 0.01.
4. Hardware split: RTX 3050 Ti 4GB + 16GB RAM → **Chỉ Phase 1**; Cloud (Kaggle T4 x2 / P100 / TPU v5e-8) → Phase 2 + 3.
	**Kiến Trúc MLOps Hybrid Edge-Cloud (Phân tách tài nguyên theo Vòng đời)**
	- **Edge (Inference & Fast Memory - Giới hạn cứng 4GB VRAM):** Đảm nhiệm Phase 1. Chạy TTT (Test-Time Training) bằng Delta Rule và phân loại dị thường bằng CADIC Coreset scoring. Yêu cầu tính toán siêu nhẹ, không lưu Computational Graph, đảm bảo tốc độ suy luận >10 FPS trên băng chuyền sản xuất.
	- **Cloud (Deep Consolidation - GPU T4/A100):** Đảm nhiệm Phase 2 & 3. Thực hiện theo chu kỳ Offline (ví dụ: cuối ca làm việc). Cloud sẽ nhận các embeddings mới từ Edge, thực hiện Backpropagation (LEJEPA Loss) và tính toán ma trận trực giao NSP2 để hợp nhất tri thức vĩnh viễn vào DINOv3 Backbone, sau đó push mô hình đã cập nhật ngược lại xuống Edge.
5. Training phases: Phase 1 (freeze DINOv3) → Phase 2 (attach TITANS + CADIC coreset update) → Phase 3 (NSP2+CBP+streaming + N2B-NC).
6. Benchmark: MVTec AD (15 categories), VisA Dataset (12 categories).
7. **Metric đánh giá:** Bắt buộc báo cáo:
   - **Image-level AUROC** (phát hiện ảnh lỗi) — metric chính
   - **Pixel-level AUPR (AP)** (phân đoạn chính xác vết xước) — metric phụ quan trọng
   - **Forgetting Measure (FM)** — đánh giá catastrophic forgetting trên old tasks
   - *Target tham khảo từ CADIC: MVTec I-AUROC ≥ 0.972, P-AUPR ≥ 0.584; VisA I-AUROC ≥ 0.891*
8. ACC threshold: τ = 0.25 khởi đầu.
9. ***CSR (Contrastive Sparse Representation) bị loại khỏi pipeline chính. Anomaly scoring dùng nearest-neighbor trong CADIC Coreset (Eq. 8-9) thay thế. Không cần retrain CSR sau mỗi N2B-NC cycle.***
10. TITANS η clamp: `torch.clamp(..., -5.0, 5.0)`, η_t = η₀ / (1 + surprise_t).
11. NSP2 rank threshold: Σ(σ_i²) / Σ(σ_all²) ≥ 0.99. Nếu dim(Null_T) < 64 → Subspace Recycling. Fallback: nếu vẫn < 32 → ε_min: 64 → 32 → 16 (log rõ).
12. Gradient clipping trong N2B-NC: Sau `.backward()`, trước NSP2: `torch.nn.utils.clip_grad_norm_(..., max_norm=1.0)`.
	**Trigger N2B-NC:** Sau mỗi **N = 5 tasks** (default).  
	Nếu dim(Null_T) giảm nhanh → giảm N xuống 3.  
	Nếu dim(Null_T) ổn định → tăng N lên 10.
13. **CADIC Coreset size:** Default = 1000 entries. Tune nếu cần: 2500 (faster, AUROC ~0.955), 5000 (balanced). Với 4GB VRAM: lưu `patch_embeddings` [256 × 768 per entry] → **1000 entries × 256 × 768 × 4 bytes ≈ 786 MB** — vẫn an toàn, không lưu raw images hay Diffusion weights.
14. **Patch embeddings cho Pixel-level Map:** Dùng `x_norm_patchtokens` từ DINOv3. Với ViT-B/14 và input 224×224: output shape [batch, 256, 768] (16×16 patches). Upsample anomaly map từ 16×16 → 224×224.

---

## 6. Định lượng Tác động & Khả thi Code

Hệ thống được thiết kế modular:
- `titans_memory.py` — TITANS Fast Memory
- `null_space_proj.py` — NSP2 + Subspace Recycling
- `cbp.py` — Continual Backprop
- `cadic_coreset.py` — **[MỚI]** Incremental Coreset, Unified Memory Bank, Anomaly Scoring (thay thế `slow_memory.py` và `csr.py`)
- `anomaly_decoder.py` — **[MỚI]** Pixel-level Anomaly Map từ patch embeddings
- `ttt_engine.py` — TTT Phase 1
- `consolidation_engine.py` — N2B-NC Phase 3
- `main.py` — Pipeline orchestration

Chi phí tính toán: O(d²) cho TITANS, O(M×d) cho CADIC distance computation (M=1000, d=768), O(d³) SVD chỉ 1 lần/task → hoàn toàn khả thi trên hardware đã nêu.

---

## 7. Paper Selection Table

**26 papers được chọn làm nền tảng cốt lõi** (Khu 1–5) + **3 papers CAD bổ sung**:

| Khu       | Paper chính                                                                           | Source gốc                                                                                          | Lý do chọn                                                                                                                                                                                                                                                  |
| --------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1         | **[Nested Learning] NL.pdf**                                                          | Nested Learning: The Illusion of Deep Learning Architectures                                        | Khung sườn toàn bộ hệ thống (Inner Loop Fast ↔ Outer Loop Slow)                                                                                                                                                                                             |
| 1         | **[Titan] 2501.00663v1.pdf**                                                          | Titans: Learning to Memorize at Test Time (arXiv:2501.00663)                                        | TITANS Memory + Delta Rule                                                                                                                                                                                                                                  |
| 1         | **[LoRA] 2512.03402v4.pdf**                                                           | Advanced LoRA Variants (arXiv:2512.03402v4)                                                         | Blueprint cho Dual-LoRA (tham khảo)                                                                                                                                                                                                                         |
| 1         | **[LoRA] 2305.14314v1.pdf (QLoRA)**                                                   | QLoRA: Efficient Finetuning of Quantized LLMs                                                       | NF4 Quantization + VRAM optimization                                                                                                                                                                                                                        |
| 1         | **2505.11998v4.pdf (PEARL - Future)**                                                 | PEARL: Prompting with Adaptive Routing                                                              | Dynamic Routing cho adapter                                                                                                                                                                                                                                 |
| 1         | **2506.03951v2.pdf (Dual-Arch)**                                                      | Dual-Architecture Networks for Stability & Plasticity                                               | Lý thuyết mạng sâu dẻo + ổn định                                                                                                                                                                                                                            |
| 1         | **[Attention] 1706.03762v7.pdf**                                                      | Attention Is All You Need                                                                           | Nền tảng toán học Transformer                                                                                                                                                                                                                               |
| 2         | **Loss of plasticity... (Dohare et al., Nature 2024)**                                | Loss of Plasticity in Deep Continual Learning                                                       | CBP + Smooth-Leaky Activation                                                                                                                                                                                                                               |
| 2         | **2410.20098v3.pdf (SNR)**                                                            | SNR: Signal-to-Noise Ratio for Dead Neuron Detection                                                | Reset dead neurons                                                                                                                                                                                                                                          |
| 2         | **2509.22335v1.pdf (Spectral Collapse)**                                              | Preventing Spectral Collapse in Continual Learning                                                  | Giữ effective rank cao                                                                                                                                                                                                                                      |
| 2         | **2603.07787v1.pdf (ARROW)**                                                          | ARROW: Activation-based Regularization for Plasticity                                               | Tiêm vào MLP blocks của ViT                                                                                                                                                                                                                                 |
| 2         | **2509.22562v1.pdf (Activations)**                                                    | Activation Functions for Continual Learning                                                         | Thay ReLU bằng Smooth-Leaky                                                                                                                                                                                                                                 |
| 2         | **2410.20634v1.pdf (Fourier - Future)**                                               | Fourier Features for Plasticity Preservation                                                        | Fourier Features (mở rộng)                                                                                                                                                                                                                                  |
| 3         | **2512.07453v2.pdf (Social Welfare)**                                                 | Social Welfare Optimization in Multi-Agent Continual Learning                                       | Game Theory Gating (ACC)                                                                                                                                                                                                                                    |
| 3         | **2512.07462v2.pdf (Agent Behavior)**                                                 | Agent Behavior in Continual Learning                                                                | Tránh "defect" của Fast Memory                                                                                                                                                                                                                              |
| 3         | **2504.13173v1.pdf (MIRAS/TTT)**                                                      | MIRAS: Test-Time Training Framework                                                                 | Inner Loop + Test-Time Training                                                                                                                                                                                                                             |
| 4         | **NeurIPS-2024-visual-prompt-tuning-in-null-space...pdf**                             | Visual Prompt Tuning in Null Space for Continual Learning                                           | NSP2 + Subspace Recycling                                                                                                                                                                                                                                   |
| 4         | **[Team] 2508.10104v1 (DINOv3)**                                                      | DINOv3 (facebookresearch/dinov3)                                                                    | Backbone self-supervised + register tokens                                                                                                                                                                                                                  |
| 4         | **[Team] 2002.05709v3 (SimCLR)**                                                      | SimCLR: A Simple Framework for Contrastive Learning                                                 | Contrastive learning reference                                                                                                                                                                                                                              |
| 4         | **[Team] 1911.05371v3 (SeLa)**                                                        | Self-Labelling via Simultaneous Clustering                                                          | Self-labelling cho unsupervised TTT                                                                                                                                                                                                                         |
| 4         | **[Extra] Diffusion-maps.pdf**                                                        | Diffusion Maps and Spectral Geometry (Coifman & Lafon 2005)                                         | Visualization feature drift                                                                                                                                                                                                                                 |
| 5         | **2211.13218v2.pdf (CODA-P)**                                                         | CODA-Prompt: Continual Prompt Tuning                                                                | So sánh với Prompt-based (loại bỏ)                                                                                                                                                                                                                          |
| 5         | **2505.17799v1.pdf + 2106.01085v4.pdf (Coreset)**                                     | Coreset Selection Survey + Online Coreset                                                           | Nền tảng lý thuyết cho CADIC                                                                                                                                                                                                                                |
| 5         | **2504.17192v5.pdf (PaperCoder)**                                                     | PaperCoder: LLM-assisted Paper-to-Code                                                              | Reverse-engineer code từ paper                                                                                                                                                                                                                              |
| 5         | **AlphaEvolve.pdf**                                                                   | AlphaEvolve: Automated Algorithm Discovery                                                          | Tự động hóa hyper-parameter                                                                                                                                                                                                                                 |
| **CAD**   | **CADIC_Continual_Anomaly_<br>Detection_Based_on_<br>Incremental_Coreset.pdf**        | CADIC (arXiv:2511.08634) — Gen Yang et al.                                                          | **[XƯƠNG SỐNG]** Incremental Coreset, Unified Memory Bank, Eq. 1-9, inference scoring, benchmark MVTec/VisA                                                                                                                                                 |
| **CAD**   | **ReplayCAD_Generative_Diffusion_<br>Replay_for_Continual_<br>Anomaly_Detection.pdf** | ReplayCAD (arXiv:2505.06603) — Lei Hu et al.                                                        | **[BASELINE + DECODER TƯ DUY]** Dùng làm đối trọng so sánh; mượn tư duy Pixel-level spatial feature cho Anomaly Decoder. Bác bỏ hoàn toàn phần Generative Diffusion.                                                                                        |
| ***CAD*** | ***CAD_Fundamental.pdf***                                                             | ***"Continual Learning Approaches for Anomaly Detection" — Dalle Pezze et al. (arXiv:2212.11192)*** | ***[BÁC BỎ — Chỉ tham khảo lý thuyết] SCALE/Super-Resolution Compressed Replay không phù hợp: với CADIC Incremental Coreset chỉ cần embeddings (~~786MB), không cần nén/giải nén ảnh. Over-engineering. Giữ lại chỉ để trích dẫn background lịch sử CAD.*** |

---

## 8. Baseline So Sánh & Target Metrics

> **[Evaluation]**

Khi báo cáo kết quả, so sánh với các baseline sau (theo thứ tự quan trọng):

| Method | MVTec I-AUROC | MVTec P-AUPR | VisA I-AUROC | VisA P-AUPR | FM (↓) |
|--------|---------------|--------------|--------------|-------------|--------|
| UCAD (AAAI'24) | 0.930 | 0.456 | 0.874 | 0.300 | 0.010 |
| DFM (CVPR'25) | 0.969 | 0.511 | — | — | 0.015 |
| ReplayCAD (IJCAI'25) | 0.948 | 0.537 | 0.903 | 0.415 | 0.045/0.055 |
| **CADIC (baseline chính)** | **0.972** | **0.584** | **0.891** | **0.438** | **0.011/0.043** |
| **Meta-NATH CAD (target)** | **≥ 0.972** | **≥ 0.584** | **≥ 0.891** | **≥ 0.438** | **≤ 0.015** |

*Meta-NATH CAD kế thừa CADIC làm Slow Memory, kỳ vọng cải thiện nhờ TITANS TTT adaptation và N2B-NC backbone evolution.*

---

**Repository tham khảo:**
- `kmccleary3301/nested_learning` (TITANS + HOPE implementation)
- `galilai-group/lejepa` (LEJEPA loss)
- `khangbkk23/NestedLearningForCAD` (base repo implementation của project)
- `facebookresearch/dinov3` (DINOv3 chính thức)
- `HULEI7/ReplayCAD` (tham khảo Pixel-level Anomaly Decoder architecture)
- *CADIC implementation — sẽ open-source on GitHub theo paper*
