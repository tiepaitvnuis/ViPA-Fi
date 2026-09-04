# Báo cáo đề xuất cải tiến ViPA-Fi

## 1. Tên hướng nghiên cứu

**ViPA-Fi: Visibility-Aware Probabilistic Skeleton Estimation for WiFi-based 3D Human Pose Estimation**

Hướng này tập trung vào bài toán ước lượng tư thế người 3D từ WiFi CSI trong điều kiện tín hiệu không quan sát đầy đủ mọi khớp. Khác với các hướng chỉ thêm attention, Mamba hoặc consistency loss, ViPA-Fi xem WiFi-based pose estimation là một bài toán **partial observability**: có khớp được tín hiệu WiFi quan sát tốt, có khớp rất mơ hồ, đặc biệt là tay và khuỷu tay.

## 2. Motivation

Ước lượng tư thế 3D bằng WiFi có nhiều ưu điểm so với camera: không cần ánh sáng, hoạt động tốt hơn khi bị che khuất, ít xâm phạm riêng tư và có thể tận dụng hạ tầng WiFi sẵn có. Tuy nhiên, WiFi CSI không phải dữ liệu hình ảnh. CSI phản ánh sự thay đổi của kênh truyền do môi trường, phản xạ đa đường, vị trí antenna và chuyển động cơ thể gây ra.

Điểm khó của bài toán là CSI không cung cấp quan sát trực tiếp từng khớp. Các khớp gần thân như torso, hip, shoulder thường tạo ảnh hưởng mạnh và ổn định hơn lên CSI. Ngược lại, các khớp nhỏ và xa thân như hand, elbow có tín hiệu yếu, dễ bị nhiễu và thường có sai số lớn. Vì vậy, nếu model chỉ dự đoán một tọa độ duy nhất cho mỗi joint mà không biết joint đó đang đáng tin hay không, model dễ dự đoán over-confident và tạo lỗi lớn ở các khớp khó.

Động lực chính của ViPA-Fi là: thay vì chỉ hồi quy pose deterministic, model cần dự đoán cả **pose**, **uncertainty** và **visibility/observability** cho từng joint. Sau đó, refinement module dùng các thông tin này để tập trung sửa các khớp mơ hồ, nhất là hand và elbow.

## 3. Điểm yếu của hướng cũ và lý do pivot

Ban đầu hướng cải tiến dựa trên PASTA-Fi gồm adaptive joint-conditioned CSI alignment, hierarchical refinement và CSI consistency. Sau khi rà soát các công trình mới, đặc biệt là C-MambaPose và RePos, hướng này cần điều chỉnh.

### 3.1. Adaptive joint-query alignment dễ bị trùng

C-MambaPose đã sử dụng cơ chế cross-attention joint-query mapping: các learnable joint queries attend vào CSI tokens rồi đưa qua GraFormer/graph decoder. Cơ chế này rất gần với module adaptive joint-conditioned alignment. Vì vậy, nếu lấy joint-query alignment làm đóng góp chính thì novelty yếu, đặc biệt khi nộp hội nghị A*.

Trong code hiện tại, phần `SkeletonEvidenceAggregator` vẫn được giữ như một component hỗ trợ để tạo joint-level evidence, nhưng không nên claim nó là novelty chính.

### 3.2. Mamba không phải contribution chính

Các paper gần đây đã khai thác Mamba/SSM cho WiFi pose. Vì vậy, paper mới không nên viết theo hướng "GraphPose-Fi + Mamba". Mamba chỉ nên xuất hiện trong related work hoặc novelty audit.

### 3.3. Consistency/domain adaptation không đủ mới

Các hướng như WiFi-JEPA, DT-Pose và AdaPose đã khai thác self-supervised learning, consistency hoặc domain-robust representation. Vì vậy, consistency trong ViPA-Fi chỉ được giữ như một tùy chọn robustness experiment, không phải contribution chính.

### 3.4. Bottleneck còn lại: WiFi quan sát không đều các joint

Điểm còn mạnh để khai thác là bản chất không quan sát đầy đủ của WiFi. Thay vì giả định mọi joint đều có độ tin cậy giống nhau, ViPA-Fi dự đoán uncertainty và visibility riêng cho từng joint. Đây là điểm khác biệt chính so với các mô hình deterministic pose regression.

## 4. Method tổng quan

Pipeline hiện tại của ViPA-Fi:

```text
WiFi CSI
-> Shared CNN encoder
-> Skeleton evidence aggregation
-> GraFormer coarse pose decoder
-> Joint uncertainty and visibility head
-> Visibility-aware limb refinement
-> Visibility-aware extremity refinement
-> Final 3D pose
```

Trong đó:

- `Shared CNN encoder`: trích xuất feature từ CSI theo từng antenna.
- `Skeleton evidence aggregation`: chuyển CSI feature thành joint-level representation.
- `GraFormer coarse pose decoder`: dự đoán pose thô.
- `Joint uncertainty and visibility head`: dự đoán độ bất định và độ quan sát được của từng joint.
- `Visibility-aware refinement`: refine limb/extremity bằng residual gate dựa trên uncertainty/visibility.

## 5. Cải tiến 1: Visibility-aware probabilistic prediction

Thay vì chỉ dự đoán tọa độ 3D:

```text
CSI -> p_j = (x_j, y_j, z_j)
```

ViPA-Fi dự đoán thêm:

```text
mu_j: tọa độ 3D dự đoán
log_var_j: log variance / uncertainty của joint
visibility_j: điểm quan sát được của joint
```

Ý nghĩa:

- `mu_j` là tọa độ joint cuối cùng.
- `log_var_j` biểu diễn uncertainty. Joint khó hoặc mơ hồ có thể có uncertainty cao hơn.
- `visibility_j` biểu diễn mức độ model tin rằng joint đang được CSI quan sát tốt.

Lợi ích:

1. Model không còn bị ép phải over-confident cho mọi joint.
2. Có thể phân tích được joint nào đáng tin, joint nào mơ hồ.
3. Tạo thêm chỉ số đánh giá cho paper: uncertainty-error correlation, mean visibility, per-joint uncertainty.

Trong code, phần này nằm ở class `JointUncertaintyHead`.

## 6. Cải tiến 2: Visibility-aware extremity refinement

ViPA-Fi dùng coarse-to-fine refinement:

```text
P_coarse -> P_limb -> P_final
```

Stage đầu tiên, GraFormer dự đoán pose thô `P_coarse`. Sau đó:

1. Limb refinement sửa các khớp thuộc limb.
2. Extremity refinement sửa các khớp xa thân như hand và foot.

Điểm khác biệt là residual refinement không hoạt động mù. Nó nhận thêm:

- joint feature
- parent joint feature
- coarse pose
- parent pose
- uncertainty
- parent uncertainty
- visibility
- parent visibility

Sau đó module học residual gate:

```text
P_limb = P_coarse + gate_limb * delta_limb
P_final = P_limb + gate_ext * delta_ext
```

Gate giúp model quyết định nên sửa joint mạnh hay nhẹ. Khi joint có uncertainty cao hoặc visibility thấp, refinement có thể dựa nhiều hơn vào skeleton context và parent-joint context.

Mục tiêu chính của module này là cải thiện các khớp khó như elbow và hand, không chỉ cải thiện MPJPE trung bình.

Trong code, phần này nằm ở class `VisibilityAwareExtremityRefinement`.

## 7. Cải tiến 3: Resource-efficient training objective

Loss tổng trong training:

```text
L_total = L_pose
        + lambda_ext * L_ext
        + lambda_coarse * L_coarse
        + lambda_nll * L_nll
        + lambda_vis * L_visibility
        + lambda_cons * L_cons
```

Trong đó:

- `L_pose`: lỗi pose cuối cùng, dùng joint distance tương tự MPJPE.
- `L_ext`: loss riêng cho các joint khó, gồm elbow và hand.
- `L_coarse`: giám sát pose thô để coarse decoder học ổn định.
- `L_nll`: uncertainty-aware Gaussian negative log-likelihood.
- `L_visibility`: calibration loss cho visibility.
- `L_cons`: optional consistency loss, mặc định tắt để tiết kiệm tài nguyên.

Cấu hình mặc định hiện tại:

```text
lambda_ext = 0.35
lambda_coarse = 0.25
lambda_nll = 0.10
lambda_vis = 0.02
lambda_cons = 0.0
```

Consistency được để `0.0` mặc định vì nếu bật sẽ cần forward thêm một lần với CSI perturbation, làm tăng thời gian train và GPU memory. Khi cần thí nghiệm robustness cross-environment, có thể bật lại với giá trị nhỏ như `0.03`.

## 8. Các chỉ số cần báo cáo khi viết paper

Để chứng minh hướng ViPA-Fi, không nên chỉ báo cáo MPJPE trung bình. Cần báo cáo thêm:

1. MPJPE, PA-MPJPE, PCK.
2. Per-joint MPJPE.
3. Extremity MPJPE cho elbow/hand.
4. Mean visibility.
5. Mean uncertainty.
6. Tương quan giữa uncertainty và prediction error.
7. Cross-environment performance.

Điểm quan trọng nhất là chứng minh hand/elbow giảm lỗi và uncertainty/visibility phản ánh đúng độ khó của từng joint.

## 9. Các file code đã cải tiến

### 9.1. `model/model.py`

Đây là file quan trọng nhất vì chứa kiến trúc ViPA-Fi.

Các thành phần chính:

- `SkeletonEvidenceAggregator`: tạo joint-level evidence từ CSI feature. Đây là component hỗ trợ, không phải novelty chính.
- `JointUncertaintyHead`: dự đoán `log_var` và `visibility` cho từng joint.
- `VisibilityAwareExtremityRefinement`: refine pose bằng residual gate dựa trên uncertainty/visibility.
- `ViPAFiNet`: model chính thay cho tên model cũ.

Các cải tiến kỹ thuật:

- `Resize` được đưa ra khỏi forward loop để giảm overhead.
- ResNet34 dùng API `weights` mới của torchvision.
- Forward hỗ trợ `return_aux=True` để trả về `coarse_pose`, `log_var`, `visibility`, `limb_gate`, `extremity_gate`.

### 9.2. `train.py`

Đây là file chứa training pipeline và loss mới.

Các thay đổi chính:

- Import `ViPAFiNet`.
- Default experiment là `vipafi`.
- Thêm `probabilistic_pose_loss`.
- Thêm `visibility_calibration_loss`.
- Thêm extremity loss cho elbow/hand.
- Thêm coarse pose auxiliary loss.
- Thêm optional consistency loss nhưng mặc định tắt.
- Log thêm `mean_visibility`, `mean_uncertainty`, `extremity_mpjpe`, `per_joint_mpjpe`.
- Chỉ apply `_weights_init` khi không dùng pretrained weights, tránh ghi đè pretrained ResNet.

### 9.3. `feeder/mmfi.py`

File này xử lý dataset và dataloader.

Các thay đổi chính:

- `make_dataloader` nhận `num_workers` từ CLI.
- Thêm `pin_memory` khi có CUDA.
- Thêm `persistent_workers` khi `num_workers > 0`.

Mục tiêu là tiết kiệm tài nguyên và cho phép điều chỉnh tốc độ đọc data tùy máy.

### 9.4. `model/GraFormer.py`

File này chứa graph pose decoder.

Các thay đổi chính:

- `adj` và `src_mask` được đăng ký bằng `register_buffer`.
- Mục đích là để adjacency matrix và mask tự đi theo device của model khi chuyển CPU/GPU.

Đây là chỉnh sửa kỹ thuật để ổn định training, không phải contribution chính.

### 9.5. `config/mmfi/pose_config_p1s1.yaml`, `pose_config_p1s2.yaml`, `pose_config_p1s3.yaml`

Các config này đã đổi:

```text
experiment_name: vipafi
```

### 9.6. `README.md`

README đã được viết lại theo hướng ViPA-Fi, gồm mô tả model, lệnh train và các output metric.

### 9.7. `reports/ViPA-Fi_direction_vi.md`

File ghi chú hướng nghiên cứu đã chốt, dùng để giải thích nhanh motivation và claim chính.

## 10. Lệnh train đề xuất

Lệnh train chính, tiết kiệm tài nguyên:

```bash
python train.py --config_file config/mmfi/pose_config_p1s1.yaml
```

Lệnh explicit:

```bash
python train.py --config_file config/mmfi/pose_config_p1s1.yaml --experiment_name vipafi --agg_mode joint_attn --ext_loss_weight 0.35 --coarse_loss_weight 0.25 --nll_loss_weight 0.10 --visibility_loss_weight 0.02 --consistency_weight 0
```

Nếu muốn giảm RAM/CPU khi train:

```bash
python train.py --config_file config/mmfi/pose_config_p1s1.yaml --num_workers 4 --max_device_batch_size 128 --batch_size 256
```

Nếu muốn thử robustness cross-environment:

```bash
python train.py --config_file config/mmfi/pose_config_p1s3.yaml --experiment_name vipafi_consistency --consistency_weight 0.03
```

## 11. Kết luận

ViPA-Fi hiện tại đã chuyển trọng tâm từ "joint-query alignment" sang **visibility-aware probabilistic WiFi pose estimation**. Đây là hướng hợp lý hơn cho mục tiêu hội nghị A* vì nó đặt lại formulation của bài toán: WiFi không quan sát đồng đều mọi joint, nên model cần biết mức độ không chắc chắn và khả năng quan sát được của từng joint.

Các đóng góp chính nên claim khi viết paper là:

1. Visibility-aware probabilistic formulation cho WiFi-based 3D HPE.
2. Uncertainty/visibility-guided residual refinement cho các joint khó.
3. Phân tích chi tiết per-joint, extremity, uncertainty và cross-environment robustness.

Các thành phần như joint-query aggregation và consistency chỉ nên xem là component hỗ trợ hoặc ablation, không phải novelty chính.
