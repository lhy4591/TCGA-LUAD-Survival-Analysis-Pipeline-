# TCGA-LUAD-Survival-Analysis-Pipeline-

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Pandas](https://img.shields.io/badge/pandas-Data%20Engineering-orange.svg)
![Lifelines](https://img.shields.io/badge/lifelines-Survival%20Analysis-success.svg)

## 📌 项目概述

本项目是一个针对高维多组学数据（以 TCGA-LUAD 肺腺癌队列为例）的自动化生存分析与数据清洗引擎。
区别于传统的调用现有 R 包，本项目基于纯 Python 栈构建底层清洗逻辑，专注于解决真实医疗开源数据中常见的**特征维度错乱、右删失数据缺失、非标准文本特征**等工程痛点，实现了从原始 `.tsv/.gz` 文件到 Kaplan-Meier 生存曲线的端到端（End-to-End）自动化分析。

## 🚀 核心工程亮点 (Engineering Highlights)

* **高维矩阵自动对齐 (Automated Matrix Transposition):** 针对 UCSC Xena 等数据库下载的 RNA-seq 表达谱常出现的维度反转问题（如 20,000+ 基因在行，500+ 样本在列），内置防御性检测机制，自动执行矩阵转置与双表 `Inner Join` 主键对齐。
* **高鲁棒性的临床特征工程 (Robust Feature Engineering):**
    * **右删失数据填补 (Right Censoring Imputation):** 自动识别生存分析中的右删失特性，联合 `days_to_death` 与 `days_to_last_followup` 字段，实现总生存期 (`OS.time`) 的动态逻辑补齐。
    * **防弹级文本清洗 (Bulletproof Text Parsing):** 利用向量化字符串处理（`.str.upper()`），彻底消除 `vital_status` 中存在的混合大小写（如 Alive/LIVING/Dead/DECEASED）带来的编码错误，并强制类型转换以隔离未知脏数据。
* **全链路自动化建模:** 自动化完成单基因靶点（如 EGFR）的表达量分层，计算 Log-rank 检验 p 值，并输出带有 95% 置信区间的工业级学术图表。

## 🛠️ 技术栈与依赖

* 核心数据清洗：`pandas`, `numpy`
* 生存分析与统计检验：`lifelines`, `scipy`
* 数据可视化：`matplotlib`, `seaborn`

```bash
# 快速安装依赖
pip install pandas numpy matplotlib seaborn lifelines scipy
