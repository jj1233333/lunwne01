# COSTE 数据预处理

在项目根目录运行，使用项目现有虚拟环境。

```powershell
# 按 7:3 划分正常训练/校准组，检查全量数据并拟合训练组参数
.venv/Scripts/python.exe -m costep_diagnosis.data_convert.data --output data/COSTE_cca_split_audit

# 导出标准化模型输入，正常训练/校准组分别存放
.venv/Scripts/python.exe -m costep_diagnosis.data_convert.data --prepare --output data/COSTE_cca_preprocessed

# 如果守恒分析需要直接读取 NPZ，可同时保存原始物理量
.venv/Scripts/python.exe -m costep_diagnosis.data_convert.data --prepare --keep-raw --output data/COSTE_with_raw
```

输出目录必须不存在。失败时留下 `INCOMPLETE.txt`；成功时生成 `report.json` 并移除标记。
每块默认 20,000 行，适合大 CSV；用 `--chunksize` 调整。

## 第一层 CCA 的正常样本分组

原始 `train_faultfree` 中的每条轨迹按时间先后分为两组，默认比例 7:3。
对于目前这条 90,001 行的轨迹，无隔离段时前 63,000 行是 `normal_train`，后 27,001 行是 `normal_calibration`。

| 数据组 | 用途 |
|---|---|
| `normal_train` | 拟合 161 维标准化参数、CCA 及其训练得到的模型参数 |
| `normal_calibration` | 复用固定标准化参数和固定 CCA，计算正常预测残差，联合选择阈值与持续报警规则 |
| `test_faultfree` | 在报警规则冻结后评估正常数据的虚警表现 |
| `test_fault` | 在报警规则冻结后评估故障检测、延迟和分类表现 |
| `train_fault` | 后续故障分类模型的训练数据 |

当前程序落实的是样本分组、标准化和读取接口。CCA、预测残差、阈值与持续报警规则尚待第一层模型实现。
`normal_calibration` 的数值不参与均值、标准差、常量和重复变量的学习，也不用于重新拟合 CCA。
由于校准组参与报警规则选择，其虚警结果属于校准结果；最终表现使用独立测试集报告。

用 `--normal-train-ratio` 调整训练比例，用 `--normal-gap-rows` 指定两组间排除的样本数。
先扣除隔离段，再对保留样本按比例划分；每条轨迹的分组范围、时间和隔离段记入 `report.json`。
行范围采用从 0 开始的 `[row_start, row_stop)`。`scaler.json` 的 `fit_partitions` 记录实际拟合范围。
默认隔离长度为 0；同一连续轨迹的两段仍可能相关，应结合后续采样、滞后和相关性选择隔离长度。
如果以后有多条独立正常仿真轨迹，可进一步改为按整条轨迹分组。

滑窗必须在组内构造，不跨训练、隔离、校准边界；持续报警计数在新轨迹或新组开始时重置，同组相邻存储块之间则延续。

## 处理规则

- 原始表含 41 个 `XMEAS`、12 个 `XMV`、138 个 `SM`，共 191 个过程变量。
- 按变量名排除与 COSTEP `SM`（smeas）重复的 `XMEAS_01`～`06`、`10`、`14`、`17`、`21`～`41`，共 30 列。它们在原始表中对应 Excel 的 F～K、**O**、S、V、Z～AT 列；Q 列的 `XMEAS_12` 保留。
- 模型输入保留 11 个 `XMEAS`（07、08、09、11、12、13、15、16、18、19、20）、12 个 `XMV` 和 138 个 `SM`，共 **161 列**。标准化参数、`X`、可选 `X_raw` 和 `feature_names` 均使用此顺序；原始 CSV 按完整表头校验。
- `time_h`、`scenario_index`、`fault_id`、`fault_label`、`random_seed` 只保存为元数据。
- `y_detect = (fault_label != 0)`；`y_class = fault_label`。故障文件中故障发生前的样本仍是正常类 0。
- 类别编号不连续。`report.json` 的 `class_to_index` 将训练类别映射到从 0 开始的连续编号，原始标签仍保留。
- 均值和总体标准差只由 `normal_train` 计算，正常校准和其他分区共用这些固定参数。
- 除上述明确排除的 XMEAS 外，其余常量列保留并设 `scale=1`，故障时该列的变化仍可见。报告记录保留变量在正常训练集中的重复关系；重复关系不代表它们在所有故障下都相同。
- 每个 CSV 保持原始时间顺序，校验固定场景、种子、故障编号和严格递增时间。缺失、无穷值和非法标签报错，交由使用者核查原因。
- 原始 CSV 保持原样。模型使用标准化输入，守恒计算使用原始单位。

`report.json` 和 `scaler.json` 记录 `excluded_feature_names` 及排除原因。已有的 191 维结果，以及用全部正常样本拟合的 161 维结果，均需要用新规则重新生成；读取时检查特征集合和拟合分组。输出目录已存在时，使用 `--output` 指定新目录。

## 读取处理结果

```python
from costep_diagnosis.data_convert.data import iter_prepared_chunks

for batch in iter_prepared_chunks("data/COSTE_cca_preprocessed", "normal_train"):
    X = batch["X"]                 # (当前块样本数, 161)，float32
    y_detect = batch["y_detect"]   # 0/1
    y_class = batch["y_class"]     # 0 或原始故障编号
    time_h = batch["time_h"]
    source = str(batch["source"])  # 原始 CSV 的相对路径

# 同样读取 normal_calibration，复用训练组的模型与参数进行残差校准。
# train_faultfree 是原始目录名；读取导出数据时须明确选择两组之一。
```

下一步按模型确定滑窗长度和步长。滑窗不能跨 CSV；相邻存储块属于同一 CSV 时，保留窗口上下文。
混合正常与故障时段的窗口须明确采用末时刻标签或其他规则，检测延迟评估保留故障切换时间。

保留原始测试集。CCA 结构和模型超参数如需选择，应在正常训练组内部安排验证；正常校准组用于报警规则校准。
故障轨迹长度不同，先报告各类样本数量，再在训练阶段选择类权重或采样策略。测试集保留原分布。

CCA 建模前，按变量物理分组处理其余常量与重复列，并检查矩阵秩。保留变量沿用原始名称与编号，便于后续与图谱及守恒关系对齐。

```powershell
.venv/Scripts/python.exe -m unittest costep_diagnosis.data_convert.test_data -v
```
