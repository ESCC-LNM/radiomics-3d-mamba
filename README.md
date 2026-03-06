# ESCC-LNM

本仓库用于 **3D CT ROI + Radiomics 融合建模**。当前版本的流程已经重构为一个严格闭环：

1. **先做 radiomics 提取**；
2. **再做病人级 5 折划分 + fold 内特征筛选**；
3. **5 折训练只负责调参，不输出最终测试指标**；
4. **最终训练脚本读取 5 折输出的推荐参数，在全量 trainval 上训练一次，并且只在 internal/external test 上评一次最终指标**。

这个版本的重点不是“把 5 折结果当最终结果”，而是把 **CV 调参** 和 **最终性能评估** 严格拆开，避免方法学混乱和数据泄露。

---

## 1. 核心原则

### 1.1 严格按 `patient_id` 划分

- 5 折划分在 `select_radiomics_features.py` 中一次性生成；
- 划分结果保存为 `cv_split_manifest.csv`；
- 后续特征筛选和 5 折训练都读取同一份 manifest；
- 同一个病人的样本不会同时落入 fold-train 和 fold-val。

### 1.2 CV 阶段不碰最终测试集指标

`train_fusion_mamba.py` 的职责只有：

- 在 5 折内部做训练 / 验证；
- 输出每折表现、OOF 预测、推荐 epoch、推荐 threshold；
- 不把 internal test / external test 当作调参依据；
- 不把 internal test / external test 当作 5 折结果的一部分。

### 1.3 预处理和筛选严格 fold 内进行

在 5 折阶段：

- radiomics 特征筛选只看 fold-train；
- imputer / scaler 只在 fold-train 上 fit；
- fold-val 只做 transform；
- internal/external test 不参与 5 折调参。

### 1.4 最终指标只来自 final 脚本

`train_final_mamba.py` 会：

- 读取 `cv_tuning_summary.json`；
- 自动继承推荐 epoch / threshold / radiomics 输入维度；
- 在全量 internal trainval 上重新训练；
- 仅在训练完成后，对 internal test 和 external test 各评一次。

---

## 2. 仓库结构

```text
ESCC-LNM/
├── LICENSE
├── README.md
├── README_DATA.md
├── requirements.txt
├── .gitignore
│
├── run_standard_radiomics.py     # 第一步：提取原始 radiomics 特征
├── select_radiomics_features.py  # 第二步：病人级 5 折划分 + fold 内筛选 + final 筛选
├── data_pipeline_fusion.py       # 第三步（CV）：读取 manifest，构建 5 折 dataloader
├── train_fusion_mamba.py         # 第四步（CV）：只做 5 折调参
├── data_pipeline_final.py        # 第五步（Final）：全量 trainval / internal test / external test 数据流
├── train_final_mamba.py          # 第六步（Final）：最终训练与最终指标输出
├── mamba_fusion_model.py         # 模型定义
│
└── outputs/                      # 运行后生成（不要上传）
    ├── radiomics_features/
    ├── selected_features/
    ├── tuning_runs/
    └── final_model/
```

---

## 3. 环境安装

建议使用独立环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果你使用 GPU，请自行安装与你机器匹配的 PyTorch + CUDA 版本。

---

## 4. 数据要求

数据格式见：[`README_DATA.md`](./README_DATA.md)

当前流程默认使用：

- `./data3D/internal_group.csv`
- `./data3D/external_label.csv`
- `./data3D/trainval_images/`
- `./data3D/internal_test_images/`
- `./data3D/external_test_images/`

注意：

- `internal_group.csv` 必须包含 `patient_id`；
- `group` 列用于区分 internal trainval 与 internal test；
- 5 折只从 `group == train` 的样本池中切分；
- `group == test` 只作为最终 internal test，不参与 5 折调参。

---

## 5. 标准流程

整个闭环必须按这个顺序跑，不能乱。

### Step 1：提取 radiomics 特征

```bash
python run_standard_radiomics.py \
  --data_dir ./data3D \
  --output_dir ./outputs/radiomics_features \
  --skip_existing
```

预期输出：

```text
outputs/radiomics_features/
├── radiomics_internal_trainval.csv
├── radiomics_internal_test.csv
└── radiomics_external_test.csv
```

这一步只做特征提取，不做特征筛选。

---

### Step 2：生成病人级 5 折划分，并完成特征筛选

```bash
python select_radiomics_features.py \
  --internal_csv ./data3D/internal_group.csv \
  --radiomics_dir ./outputs/radiomics_features \
  --out_dir ./outputs/selected_features \
  --patient_id_col patient_id \
  --group_col group \
  --train_group_value train \
  --n_splits 5 \
  --mode all
```

这个脚本会做两件事。

#### 2.1 先固定 5 折病人级划分

输出：

```text
outputs/selected_features/cv_split_manifest.csv
```

这个文件是整个 5 折流程唯一合法的折划分来源。后面的训练不允许再自己重新切折。

#### 2.2 再导出两套 radiomics 特征表

**A. 给 5 折调参用：**

```text
outputs/selected_features/
├── fold01/
├── fold02/
├── fold03/
├── fold04/
└── fold05/
```

每个 `foldXX/` 下会有：

```text
radiomics_internal_trainval_sel.csv
radiomics_internal_test_sel.csv
radiomics_external_test_sel.csv
selected_features.txt
```

这里的特征集合是 **只用该 fold 的 train 子集选出来的**。

**B. 给最终训练用：**

```text
outputs/selected_features/final/
├── radiomics_internal_trainval_sel.csv
├── radiomics_internal_test_sel.csv
├── radiomics_external_test_sel.csv
└── selected_features.txt
```

这里的特征集合是 **用整个 internal trainval 池** 选出来的，只供最终训练脚本使用。

另外还会输出：

```text
outputs/selected_features/selection_summary.json
```

---

### Step 3：5 折训练，只负责调参

```bash
python train_fusion_mamba.py \
  --split_manifest_csv ./outputs/selected_features/cv_split_manifest.csv \
  --rad_root_dir ./outputs/selected_features \
  --internal_csv ./data3D/internal_group.csv \
  --external_csv ./data3D/external_label.csv \
  --trainval_img_dir ./data3D/trainval_images \
  --internal_test_img_dir ./data3D/internal_test_images \
  --external_test_img_dir ./data3D/external_test_images \
  --output_dir ./outputs/tuning_runs
```

这一步的职责只有：

- 跑 5 折 train / val；
- 依据验证集保存每折最佳 checkpoint；
- 汇总 fold 指标；
- 生成 OOF 预测；
- 输出推荐 epoch / threshold / radiomics 输入维度。

这一步 **不负责输出最终 internal/external test 性能**。

预期输出：

```text
outputs/tuning_runs/
├── fold01_best.pth
├── fold02_best.pth
├── fold03_best.pth
├── fold04_best.pth
├── fold05_best.pth
├── cv_fold_metrics.csv
├── cv_oof_predictions.csv
└── cv_tuning_summary.json
```

其中：

- `cv_fold_metrics.csv`：每折验证结果；
- `cv_oof_predictions.csv`：OOF 预测明细；
- `cv_tuning_summary.json`：最终训练阶段要读取的推荐配置。

---

### Step 4：最终训练 + 最终指标输出

```bash
python train_final_mamba.py \
  --tuning_summary_json ./outputs/tuning_runs/cv_tuning_summary.json \
  --rad_trainval_csv ./outputs/selected_features/final/radiomics_internal_trainval_sel.csv \
  --rad_internal_test_csv ./outputs/selected_features/final/radiomics_internal_test_sel.csv \
  --rad_external_test_csv ./outputs/selected_features/final/radiomics_external_test_sel.csv \
  --internal_csv ./data3D/internal_group.csv \
  --external_csv ./data3D/external_label.csv \
  --trainval_img_dir ./data3D/trainval_images \
  --internal_test_img_dir ./data3D/internal_test_images \
  --external_test_img_dir ./data3D/external_test_images \
  --output_dir ./outputs/final_model
```

这个脚本会：

- 自动加载 `cv_tuning_summary.json` 中的推荐参数；
- 在全量 internal trainval 上训练；
- 训练结束后，在 internal test 和 external test 上各评估一次；
- 输出最终模型、最终指标和预测文件。

预期输出：

```text
outputs/final_model/
├── final_mamba_fusion.pth
├── final_evaluation_metrics.csv
├── final_predictions.csv
└── final_run_protocol.json
```

---

## 6. 两个脚本各自负责什么

### `train_fusion_mamba.py`

只负责：

- 5 折内部训练；
- 5 折内部验证；
- OOF 结果；
- 推荐 threshold；
- 推荐 epochs。

它 **不是** 最终性能报告脚本。

### `train_final_mamba.py`

只负责：

- 使用 CV 输出的推荐参数进行最终训练；
- 产出最终 internal / external test 指标。

它才是最终性能报告脚本。

---

## 7. 为什么必须这样改

因为原先那种混合式流程有两个硬伤：

1. **特征筛选折** 和 **训练折** 可能不是同一套；
2. **CV 调参阶段** 和 **最终测试阶段** 没有严格拆开。

这两个问题会直接影响结果可信度。现在这个版本就是为了解决这两个问题：

- 同一份 `cv_split_manifest.csv` 贯穿整个 CV；
- 5 折只调参；
- 最终性能只由 final 脚本输出；
- 病人级隔离从 split 到筛选到训练全链路一致。

---

## 8. 常见错误

### 8.1 `internal_group.csv` 没有 `patient_id`

5 折分组的核心就是 `patient_id`。如果没有这一列，就谈不上严格的病人级隔离。

### 8.2 同一病人有冲突标签

如果同一个 `patient_id` 下出现不同 `label`，分组 CV 会直接失去定义。这个版本会直接报错。

### 8.3 5 折脚本和最终脚本混用不同 radiomics 表

- CV 阶段必须用 `fold01...fold05/`；
- final 阶段必须用 `final/`；
- 不能混着来。

### 8.4 在 CV 阶段用 internal/external test 调阈值

这会造成方法学污染。阈值应该由验证集 / OOF 结果给出，而不是测试集。

---

## 9. 最小可运行命令

```bash
python run_standard_radiomics.py \
  --data_dir ./data3D \
  --output_dir ./outputs/radiomics_features \
  --skip_existing

python select_radiomics_features.py \
  --internal_csv ./data3D/internal_group.csv \
  --radiomics_dir ./outputs/radiomics_features \
  --out_dir ./outputs/selected_features \
  --patient_id_col patient_id \
  --group_col group \
  --train_group_value train \
  --n_splits 5 \
  --mode all

python train_fusion_mamba.py \
  --split_manifest_csv ./outputs/selected_features/cv_split_manifest.csv \
  --rad_root_dir ./outputs/selected_features \
  --internal_csv ./data3D/internal_group.csv \
  --external_csv ./data3D/external_label.csv \
  --trainval_img_dir ./data3D/trainval_images \
  --internal_test_img_dir ./data3D/internal_test_images \
  --external_test_img_dir ./data3D/external_test_images \
  --output_dir ./outputs/tuning_runs

python train_final_mamba.py \
  --tuning_summary_json ./outputs/tuning_runs/cv_tuning_summary.json \
  --rad_trainval_csv ./outputs/selected_features/final/radiomics_internal_trainval_sel.csv \
  --rad_internal_test_csv ./outputs/selected_features/final/radiomics_internal_test_sel.csv \
  --rad_external_test_csv ./outputs/selected_features/final/radiomics_external_test_sel.csv \
  --internal_csv ./data3D/internal_group.csv \
  --external_csv ./data3D/external_label.csv \
  --trainval_img_dir ./data3D/trainval_images \
  --internal_test_img_dir ./data3D/internal_test_images \
  --external_test_img_dir ./data3D/external_test_images \
  --output_dir ./outputs/final_model
```

---

## 10. 说明

- `data_pipeline_fusion.py` 用于 **CV 调参阶段**；
- `data_pipeline_final.py` 用于 **最终训练阶段**；
- `mamba_fusion_model.py` 只是模型定义，本身不决定是否泄露，真正决定是否泄露的是 split、筛选和 fit/transform 的边界。

这套流程的目的只有一个：

**让 5 折调参和最终性能评估真正闭环，而且方法学上说得过去。**
