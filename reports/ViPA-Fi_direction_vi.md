# ViPA-Fi: hướng cải tiến đã chốt

## Tên hướng

**ViPA-Fi: Visibility-Aware Probabilistic Skeleton Estimation for WiFi-based 3D Human Pose Estimation**

## Motivation

Các mô hình WiFi-based 3D HPE hiện tại thường dự đoán trực tiếp một tọa độ 3D cho mỗi joint. Cách này giả định mọi khớp đều được WiFi quan sát đủ tốt, nhưng thực tế không phải vậy. Tín hiệu CSI chịu ảnh hưởng mạnh bởi multipath, môi trường, vị trí antenna và chuyển động cục bộ. Các khớp gần thân như torso/head thường ổn định hơn, còn hand/elbow dễ mơ hồ và có sai số lớn.

Vì vậy, hướng ViPA-Fi không đặt trọng tâm vào việc chỉ thêm joint-query attention hay Mamba. Hướng chính là mô hình hóa **partial observability**: joint nào WiFi quan sát tốt, joint nào mơ hồ, và khi mơ hồ thì model cần biết mức độ không chắc chắn thay vì dự đoán over-confident.

## Điểm yếu của GraphPose-Fi và các hướng gần đây

GraphPose-Fi đã đưa skeleton graph vào pose decoder, nhưng vẫn hồi quy deterministic pose. Các công trình mới như C-MambaPose đã khai thác joint-query mapping và hiệu quả tham số, nên adaptive joint query không còn đủ mạnh để làm contribution chính. Các hướng domain consistency, JEPA/link masking, amplitude/phase preprocessing và bone loss cũng đã có nhiều hàng xóm gần.

Điểm còn đáng khai thác là: WiFi pose estimation là bài toán quan sát không đầy đủ. Sai số của các joint không đồng đều, nhưng đa số mô hình vẫn huấn luyện như thể mọi joint có độ tin cậy như nhau.

## Method

### 1. Visibility-aware probabilistic prediction

Model dự đoán thêm hai đại lượng cho từng joint:

```text
mu_j: tọa độ joint dự đoán
sigma_j: uncertainty của joint
v_j: visibility / observability score của joint
```

`sigma_j` được học bằng uncertainty-aware Gaussian NLL. `v_j` được học bằng calibration loss nhẹ dựa trên lỗi pose hiện tại, giúp visibility thấp hơn khi joint khó dự đoán.

### 2. Visibility-aware extremity refinement

Sau khi GraFormer tạo pose thô, model dùng residual refinement cho limb và extremity. Refinement nhận thêm uncertainty và visibility, rồi dùng gate để quyết định mức độ sửa residual.

```text
coarse pose -> limb refinement -> extremity refinement -> final pose
```

Mục tiêu chính là cải thiện hand/elbow, không chỉ giảm MPJPE trung bình.

### 3. Resource-efficient training

Consistency augmentation vẫn giữ trong code để chạy robustness experiment, nhưng mặc định `consistency_weight = 0` để không cần forward thêm một lần. Điều này tiết kiệm GPU memory và thời gian train. Objective mặc định tập trung vào MPJPE, extremity loss, coarse auxiliary loss, uncertainty NLL và visibility calibration.

## Contribution claim nên dùng

1. Đề xuất visibility-aware probabilistic formulation cho WiFi-based 3D HPE.
2. Đề xuất uncertainty/visibility-guided residual refinement cho các joint khó như hand/elbow.
3. Đánh giá bằng per-joint MPJPE, extremity MPJPE, uncertainty/visibility statistics và cross-environment robustness.

## Những gì không claim

- Không claim joint-query alignment là đóng góp chính.
- Không claim Mamba/SSM là đóng góp chính.
- Không claim amplitude/phase preprocessing là mới.
- Không claim bone-length loss là mới.
- Không claim generic consistency/domain adaptation là mới.

## Code chính đã đổi

- `model/model.py`: `ViPAFiNet`, `SkeletonEvidenceAggregator`, `JointUncertaintyHead`, `VisibilityAwareExtremityRefinement`.
- `train.py`: uncertainty NLL, visibility calibration loss, extremity loss, coarse loss, optional CSI consistency.
- `config/mmfi/*.yaml`: experiment name chuyển sang `vipafi`.
- `README.md`: mô tả và lệnh chạy theo ViPA-Fi.

## Lệnh train chính

```bash
python train.py --config_file config/mmfi/pose_config_p1s1.yaml
```

Lệnh explicit:

```bash
python train.py --config_file config/mmfi/pose_config_p1s1.yaml --experiment_name vipafi --agg_mode joint_attn --ext_loss_weight 0.35 --coarse_loss_weight 0.25 --nll_loss_weight 0.10 --visibility_loss_weight 0.02 --consistency_weight 0
```
