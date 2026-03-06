# README_DATA.md

这个文件只讲数据准备，不讲训练策略。

当前版本的代码默认数据根目录为 `./data3D`。如果你改目录名，命令行参数也要一起改。

---

## 1. 你必须准备的内容

这套流程至少需要三部分数据：

1. **internal trainval 图像**
2. **internal test 图像**
3. **external test 图像**

再配两张元数据表：

- `internal_group.csv`
- `external_label.csv`

另外，radiomics 原始特征表不是手工准备，而是由 `run_standard_radiomics.py` 生成。

---

## 2. 推荐目录结构

```text
data3D/
├── internal_group.csv
├── external_label.csv
│
├── trainval_images/
├── trainval_masks/
│
├── internal_test_images/
├── internal_test_masks/
│
├── external_test_images/
└── external_test_masks/
```

说明：

- `trainval_images/`：internal train/val 池对应的 ROI 图像；
- `trainval_masks/`：对应 ROI mask；
- `internal_test_images/`：internal test 图像；
- `internal_test_masks/`：对应 ROI mask；
- `external_test_images/`：external test 图像；
- `external_test_masks/`：对应 ROI mask。

---

## 3. 文件命名规则

### 3.1 图像和 mask 必须同名同 stem

例如：

```text
trainval_images/0001.nii.gz
trainval_masks/0001.nii.gz
```

或者：

```text
external_test_images/ESCC_021.nii.gz
external_test_masks/ESCC_021.nii.gz
```

代码默认支持这些后缀：

- `.nii.gz`
- `.nii`
- `.nii(1).gz`

### 3.2 CSV 里的 `ID` 要能和文件名对上

例如文件名是：

```text
0001.nii.gz
```

那 CSV 中的 `ID` 应该写：

```text
0001
```

不要写成别的版本，比如：

- `1`
- `0001.nii.gz`
- `case_0001`  

除非你磁盘上的文件也真是那个名字。

---

## 4. `internal_group.csv` 的要求

这是最重要的一张表。

### 4.1 必需字段

`internal_group.csv` 至少要有这几列：

- `ID`
- `label`
- `group`
- `patient_id`

推荐格式：

| ID   | label | group | patient_id |
|------|------:|-------|------------|
| 0001 | 0     | train | P001       |
| 0002 | 1     | train | P002       |
| 0101 | 0     | test  | P101       |

### 4.2 各字段含义

#### `ID`

样本 ID，对应图像和 mask 的文件名 stem。

#### `label`

分类标签，当前代码按二分类写的，建议只用：

- `0`
- `1`

#### `group`

用于区分 internal trainval 和 internal test：

- `train`：进入 5 折调参池；
- `test`：作为 internal test，只在最终评估时使用。

注意：

- `group == train` 的样本会被拿来做病人级 5 折；
- `group == test` 不参与 5 折切分。

#### `patient_id`

这是整个流程里最关键的一列。

5 折划分是按 `patient_id` 做的，不是按 `ID` 做的。也就是说：

- 同一个病人的多个样本必须共享同一个 `patient_id`；
- 同一个病人的样本不能同时出现在 fold-train 和 fold-val；
- 如果没有 `patient_id`，你就做不到严格的病人级隔离。

### 4.3 强约束

#### 同一病人不能有冲突标签

例如下面这种表是不合法的：

| ID   | label | group | patient_id |
|------|------:|-------|------------|
| 0001 | 0     | train | P001       |
| 0002 | 1     | train | P001       |

同一个 `patient_id` 出现不同 `label`，当前流程会直接报错。

#### `ID` 不能重复

同一张表里同一个 `ID` 不应该出现两次。

---

## 5. `external_label.csv` 的要求

### 5.1 必需字段

最少需要：

- `ID`
- `label`

可选字段：

- `patient_id`

推荐格式：

| ID    | label | patient_id |
|-------|------:|------------|
| E001  | 0     | EP001      |
| E002  | 1     | EP002      |

如果你不给 `patient_id`，当前数据管线会默认把 `ID` 当成 `patient_id` 使用。这不会影响最终评估运行，但如果 external 数据里同一个病人有多条样本，你最好还是把 `patient_id` 明确写出来。

---

## 6. ROI 图像和 mask 的要求

### 6.1 这是 ROI 级数据，不是原始整幅 CT

当前流程默认你输入的是已经裁剪好的 ROI 体数据，而不是整套原始 DICOM/CT。模型侧会再做统一 resize/windowing，但不会替你自动找 lesion。

### 6.2 mask 中目标区域标签默认应为 1

`run_standard_radiomics.py` 默认按 PyRadiomics 的常见用法提取 ROI。如果你的 mask 不是用 `1` 表示病灶，而是别的整数编码，你需要同步检查 radiomics 提取配置。

### 6.3 图像和 mask 必须一一对应

任何一个样本都必须同时存在：

- 图像文件
- 对应 mask 文件
- CSV 中的记录

三者缺一个，流程就会断。

---

## 7. radiomics 特征表是怎么来的

不要手工写 radiomics 选后表。正确顺序是：

### 7.1 先提取原始 radiomics

```bash
python run_standard_radiomics.py \
  --data_dir ./data3D \
  --output_dir ./outputs/radiomics_features \
  --skip_existing
```

输出通常是：

```text
outputs/radiomics_features/
├── radiomics_internal_trainval.csv
├── radiomics_internal_test.csv
└── radiomics_external_test.csv
```

### 7.2 再做筛选

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

这个脚本会自动输出：

- `cv_split_manifest.csv`
- `fold01 ... fold05/` 的 fold-specific 选后特征表
- `final/` 的最终训练特征表

---

## 8. 你在数据准备阶段最容易犯的错

### 8.1 `patient_id` 填成每行唯一

如果一个病人的多条样本其实属于同一人，但你把 `patient_id` 也写成每行唯一，那分组 CV 就等于白做。

### 8.2 internal test 混进了 train

只要样本属于最终 internal test，就应该在 `group` 列标成 `test`，不要放进 `train`。

### 8.3 radiomics 表和图像目录不是同一批样本

你必须保证：

- `trainval_images/` 中的样本，能在 `radiomics_internal_trainval.csv` 里找到；
- `internal_test_images/` 中的样本，能在 `radiomics_internal_test.csv` 里找到；
- `external_test_images/` 中的样本，能在 `radiomics_external_test.csv` 里找到。

### 8.4 一个 `ID` 对不上文件名

这是最常见的低级错误。只要 ID 对不上，后面的图像和 radiomics 就拼不起来。

### 8.5 同一个病人的标签矛盾

这类数据问题不要指望训练脚本给你“自动修复”。该清洗就先清洗。

---

## 9. 开跑前检查清单

正式训练前，先自己核对一遍：

- [ ] `internal_group.csv` 包含 `ID / label / group / patient_id`
- [ ] `external_label.csv` 至少包含 `ID / label`
- [ ] `group` 中同时存在 `train` 和 `test`
- [ ] 同一病人的 `patient_id` 一致
- [ ] 同一病人没有冲突标签
- [ ] 图像、mask、CSV 的 `ID` 一一对应
- [ ] radiomics 提取输出和图像目录属于同一批样本
- [ ] internal test 没有进入 5 折训练池

---

## 10. 不要上传什么

不要上传到 GitHub：

- 原始医学影像
- mask
- 含病人标识的 CSV
- `outputs/` 下所有中间结果
- checkpoint
- logs
- radiomics 提取表

这是数据安全底线，不是建议项。
