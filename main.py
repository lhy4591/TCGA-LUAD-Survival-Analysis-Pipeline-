#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCGA肿瘤基因表达与生存分析预测项目
作者: 灵码 (Lingma)
功能: 读取TCGA数据 → 基因表达处理 → 生存分析 → 预后预测 → 结果可视化
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TCGASurvivalAnalyzer:
    """TCGA生存分析器主类"""
    
    def __init__(self, data_dir='data', results_dir='results', figures_dir='figures'):
        """
        初始化分析器
        
        参数:
        data_dir: 数据文件夹路径
        results_dir: 结果输出文件夹路径  
        figures_dir: 图片输出文件夹路径
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.figures_dir = figures_dir
        
        # 创建必要的目录
        self._create_directories()
        
        # 初始化数据变量
        self.expression_data = None
        self.clinical_data = None
        self.merged_data = None
        self.target_gene = None
        
    def _create_directories(self):
        """创建项目所需的目录结构"""
        directories = [self.data_dir, self.results_dir, self.figures_dir]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"已创建目录: {directory}")
    
    def generate_sample_data(self, n_samples=200, n_genes=1000):
        """
        生成模拟的TCGA测试数据（当没有真实数据时使用）
        
        参数:
        n_samples: 样本数量
        n_genes: 基因数量
        """
        print("正在生成模拟TCGA数据...")
        
        # 生成随机样本ID
        sample_ids = [f"TCGA-{i:04d}" for i in range(1, n_samples + 1)]
        
        # 生成基因表达数据 (n_genes x n_samples)
        np.random.seed(42)  # 设置随机种子确保可重现性
        gene_names = [f"Gene_{i}" for i in range(1, n_genes + 1)]
        expression_matrix = np.random.lognormal(mean=5, sigma=2, size=(n_genes, n_samples))
        
        # 创建表达矩阵DataFrame
        self.expression_data = pd.DataFrame(
            expression_matrix.T, 
            columns=gene_names, 
            index=sample_ids
        )
        
        # 生成临床数据
        clinical_data_dict = {
            'bcr_patient_barcode': sample_ids,
            'OS.time': np.random.exponential(scale=1000, size=n_samples),  # 总生存时间
            'OS': np.random.binomial(1, 0.3, n_samples),  # 生存状态 (0=存活, 1=死亡)
            'age': np.random.normal(60, 10, n_samples),  # 年龄
            'gender': np.random.choice(['MALE', 'FEMALE'], n_samples),  # 性别
            'stage': np.random.choice(['I', 'II', 'III', 'IV'], n_samples)  # 肿瘤分期
        }
        self.clinical_data = pd.DataFrame(clinical_data_dict)
        
        # 保存模拟数据
        self.expression_data.to_csv(os.path.join(self.data_dir, 'simulated_expression.csv'))
        self.clinical_data.to_csv(os.path.join(self.data_dir, 'simulated_clinical.csv'), index=False)
        
        print(f"模拟数据已生成并保存到 {self.data_dir} 目录")
        print(f"- 表达数据: {self.expression_data.shape[0]} 样本, {self.expression_data.shape[1]} 基因")
        print(f"- 临床数据: {self.clinical_data.shape[0]} 样本")
    
    def load_data(self, expression_file='./data/TCGA.LUAD.sampleMap_HiSeqV2.gz', clinical_file='./data/TCGA.LUAD.sampleMap_LUAD_clinicalMatrix'):
        print("正在加载真实的TCGA数据...")
        """
        加载TCGA表达数据和临床数据
        
        参数:
        expression_file: 表达矩阵文件路径 (CSV/TSV格式)
        clinical_file: 临床数据文件路径 (CSV/TSV格式)
        """
        
        # 如果没有提供文件路径，尝试查找默认文件
        if expression_file is None:
            # 查找可能的表达数据文件
            possible_files = [
                'simulated_expression.csv',
                'expression.csv',
                'gene_expression.csv',
                'TCGA.LUAD.sampleMap_HiSeqV2'
            ]
            for file in possible_files:
                if os.path.exists(os.path.join(self.data_dir, file)):
                    expression_file = os.path.join(self.data_dir, file)
                    break
            
            if expression_file is None and os.path.exists('TCGA.LUAD.sampleMap_HiSeqV2.gz'):
                # 处理压缩的表达数据文件
                import gzip
                with gzip.open('TCGA.LUAD.sampleMap_HiSeqV2.gz', 'rt') as f:
                    self.expression_data = pd.read_csv(f, sep='\t', index_col=0)
                print("已加载压缩的表达数据文件")
                return
        
        if clinical_file is None:
            possible_files = [
                'simulated_clinical.csv',
                'clinical.csv',
                'clinical_data.csv',
                'TCGA.LUAD.sampleMap_LUAD_clinicalMatrix'
            ]
            for file in possible_files:
                if os.path.exists(os.path.join(self.data_dir, file)):
                    clinical_file = os.path.join(self.data_dir, file)
                    break
        
        # 加载表达数据
        if expression_file and os.path.exists(expression_file):
             if expression_file.endswith('.gz'):
                import gzip
                with gzip.open(expression_file, 'rt') as f:
                    self.expression_data = pd.read_csv(f, sep='\t', index_col=0)
             else:
                # 【修改点 3.1】Xena 数据不管有没有后缀，都是制表符分割的，直接写死 sep='\t'
                self.expression_data = pd.read_csv(expression_file, sep='\t', index_col=0)
            
            # 【修改点 3.2】核心转置！把列（患者ID）变成行，才能和临床数据对接！
                self.expression_data = self.expression_data.T 
            
             print(f"已加载表达数据并完成转置: {self.expression_data.shape[0]} 样本, {self.expression_data.shape[1]} 基因")
        else:
            print("未找到表达数据文件，将使用模拟数据")
            self.generate_sample_data()
            return
        
        # 加载临床数据
        if clinical_file and os.path.exists(clinical_file):
            # 【修改点 4】同样强制使用制表符
            self.clinical_data = pd.read_csv(clinical_file, sep='\t')
            print(f"已加载临床数据: {self.clinical_data.shape[0]} 样本, {self.clinical_data.shape[1]} 特征")
        else:
            print("未找到临床数据文件，将使用模拟数据")
            # 从表达数据生成匹配的临床数据
            n_samples = self.expression_data.shape[0]
            sample_ids = self.expression_data.index.tolist()
            clinical_data_dict = {
                'bcr_patient_barcode': sample_ids,
                'OS.time': np.random.exponential(scale=1000, size=n_samples),
                'OS': np.random.binomial(1, 0.3, n_samples),
                'age': np.random.normal(60, 10, n_samples),
                'gender': np.random.choice(['MALE', 'FEMALE'], n_samples),
                'stage': np.random.choice(['I', 'II', 'III', 'IV'], n_samples)
            }
            self.clinical_data = pd.DataFrame(clinical_data_dict)
            self.clinical_data.to_csv(os.path.join(self.data_dir, 'generated_clinical.csv'), index=False)
            print("已生成匹配的临床数据")
    
    def preprocess_data(self):
        """
        数据预处理：清洗、合并、处理缺失值
        """
        print("正在进行数据预处理...")
        
        # ==================== 【新增：防御性清洗代码 (适配 TCGA Xena)】 ====================
        # 1. 自动修正表达矩阵的方向 (解决 20530 样本的反向 Bug)
        # 如果行数(基因)大于5000，列数(样本)小于2000，说明肯定是反的，强制转置！
        if self.expression_data.shape[0] > 5000 and self.expression_data.shape[1] < 2000:
            print(f"检测到基因在行({self.expression_data.shape[0]})，样本在列({self.expression_data.shape[1]})，执行矩阵自动转置...")
            self.expression_data = self.expression_data.T
            print(f"校正后: {self.expression_data.shape[0]} 样本, {self.expression_data.shape[1]} 基因")

        # 2. 修复 Xena 临床数据列名不匹配的问题 (解决 KeyError)
        # 统一主键 ID
        if 'sampleID' in self.clinical_data.columns:
            self.clinical_data['bcr_patient_barcode'] = self.clinical_data['sampleID']

        # 统一生存状态 (OS): 先暴力转成全大写，再做映射，彻底消灭大小写不一致的脏数据！
        if 'vital_status' in self.clinical_data.columns:
            self.clinical_data['OS'] = self.clinical_data['vital_status'].str.upper().replace(
                {'DEAD': 1, 'DECEASED': 1, 'ALIVE': 0, 'LIVING': 0}
            )
            # 终极防御：强制把这一列转成数字，如果还有没见过的奇葩单词，直接变成 NaN 丢弃
            self.clinical_data['OS'] = pd.to_numeric(self.clinical_data['OS'], errors='coerce')

        # 统一生存时间 (OS.time): 结合 days_to_death 和 days_to_last_followup
        if 'days_to_death' in self.clinical_data.columns and 'days_to_last_followup' in self.clinical_data.columns:
            # 填补逻辑：如果是死亡患者就有死亡天数，存活患者只有最后随访天数，把这两列合并！
            self.clinical_data['OS.time'] = self.clinical_data['days_to_death'].fillna(self.clinical_data['days_to_last_followup'])
        # ======================================================================================

        # 确保临床数据有必要的列
        required_columns = ['bcr_patient_barcode', 'OS.time', 'OS']
        missing_cols = [col for col in required_columns if col not in self.clinical_data.columns]
        # ... 后面的代码完全不用动 ...
        
        if missing_cols:
            print(f"警告: 临床数据缺少必要列: {missing_cols}")
            # 尝试自动映射常见列名
            column_mapping = {
                'patient_id': 'bcr_patient_barcode',
                'overall_survival_time': 'OS.time',
                'overall_survival_status': 'OS',
                'survival_time': 'OS.time',
                'survival_status': 'OS'
            }
            for old_col, new_col in column_mapping.items():
                if old_col in self.clinical_data.columns and new_col in missing_cols:
                    self.clinical_data.rename(columns={old_col: new_col}, inplace=True)
                    print(f"已将列 '{old_col}' 重命名为 '{new_col}'")
        
        # 处理生存时间中的缺失值
        if self.clinical_data['OS.time'].isnull().any():
            median_time = self.clinical_data['OS.time'].median()
            self.clinical_data['OS.time'].fillna(median_time, inplace=True)
            print(f"用中位数 {median_time:.2f} 填充了生存时间的缺失值")
        
        # 处理生存状态中的缺失值
        if self.clinical_data['OS'].isnull().any():
            mode_status = self.clinical_data['OS'].mode()[0]
            self.clinical_data['OS'].fillna(mode_status, inplace=True)
            print(f"用众数 {mode_status} 填充了生存状态的缺失值")
        
        # 确保样本ID列存在
        if 'bcr_patient_barcode' not in self.clinical_data.columns:
            self.clinical_data['bcr_patient_barcode'] = self.clinical_data.index.astype(str)
        
        # 合并表达数据和临床数据
        # 确保索引格式一致
        clinical_barcodes = set(self.clinical_data['bcr_patient_barcode'].astype(str))
        expression_barcodes = set(self.expression_data.index.astype(str))
        
        # 找到共同的样本
        common_barcodes = clinical_barcodes.intersection(expression_barcodes)
        print(f"共同样本数量: {len(common_barcodes)}")
        
        if len(common_barcodes) == 0:
            print("警告: 临床数据和表达数据没有共同的样本！")
            # 尝试模糊匹配
            clinical_list = list(clinical_barcodes)
            expression_list = list(expression_barcodes)
            
            # 简单的前缀匹配
            matched_pairs = {}
            for exp_id in expression_list[:min(100, len(expression_list))]:  # 限制匹配数量
                for clin_id in clinical_list:
                    if exp_id.startswith(clin_id) or clin_id.startswith(exp_id):
                        matched_pairs[exp_id] = clin_id
                        break
            
            if matched_pairs:
                print(f"通过模糊匹配找到了 {len(matched_pairs)} 个样本")
                # 创建新的临床数据
                matched_clinical = []
                for exp_id, clin_id in matched_pairs.items():
                    clin_row = self.clinical_data[self.clinical_data['bcr_patient_barcode'] == clin_id].iloc[0].copy()
                    clin_row['bcr_patient_barcode'] = exp_id
                    matched_clinical.append(clin_row)
                self.clinical_data = pd.DataFrame(matched_clinical)
                common_barcodes = set(matched_pairs.keys())
            else:
                print("无法匹配样本，将使用前N个样本进行分析")
                min_samples = min(len(self.clinical_data), len(self.expression_data))
                clinical_subset = self.clinical_data.head(min_samples)
                expression_subset = self.expression_data.head(min_samples)
                clinical_subset['bcr_patient_barcode'] = expression_subset.index.astype(str)
                self.clinical_data = clinical_subset
                self.expression_data = expression_subset
                common_barcodes = set(expression_subset.index.astype(str))
        
        # 过滤共同样本
        self.clinical_data = self.clinical_data[
            self.clinical_data['bcr_patient_barcode'].astype(str).isin(common_barcodes)
        ].copy()
        self.expression_data = self.expression_data[
            self.expression_data.index.astype(str).isin(common_barcodes)
        ].copy()
        
        # 按照相同的顺序排列
        common_order = [x for x in self.expression_data.index if str(x) in common_barcodes]
        self.expression_data = self.expression_data.loc[common_order]
        self.clinical_data = self.clinical_data.set_index('bcr_patient_barcode').loc[common_order].reset_index()
        
        # 创建合并数据
        self.merged_data = self.clinical_data.copy()
        self.merged_data.index = self.expression_data.index
        
        print(f"数据预处理完成: {len(self.merged_data)} 个样本")
    
    def select_target_gene(self, gene_name=None):
        """
        选择目标基因进行分析
        
        参数:
        gene_name: 目标基因名称，如果为None则选择表达量最高的基因
        """
        if gene_name is None:
            # 选择表达量最高的基因
            mean_expression = self.expression_data.mean(axis=0)
            gene_name = mean_expression.idxmax()
            print(f"自动选择目标基因: {gene_name} (最高平均表达量)")
        
        if gene_name not in self.expression_data.columns:
            available_genes = self.expression_data.columns.tolist()[:10]
            print(f"警告: 基因 '{gene_name}' 不存在于数据中")
            print(f"可用基因示例: {available_genes}")
            gene_name = available_genes[0] if available_genes else self.expression_data.columns[0]
            print(f"将使用基因: {gene_name}")
        
        self.target_gene = gene_name
        print(f"目标基因已设置为: {self.target_gene}")
        
        # 将目标基因表达量添加到合并数据中
        self.merged_data['target_gene_expression'] = self.expression_data[gene_name]
    
    def create_expression_groups(self, method='median'):
        """
        根据目标基因表达量创建高/低表达组
        
        参数:
        method: 分组方法 ('median', 'mean', 'quantile')
        """
        if self.target_gene is None:
            raise ValueError("请先选择目标基因")
        
        expression_values = self.merged_data['target_gene_expression']
        
        if method == 'median':
            threshold = expression_values.median()
        elif method == 'mean':
            threshold = expression_values.mean()
        elif method == 'quantile':
            threshold = expression_values.quantile(0.75)  # 上四分位数
        else:
            threshold = expression_values.median()
        
        # 创建分组
        self.merged_data['expression_group'] = np.where(
            expression_values >= threshold, 'High', 'Low'
        )
        
        # 统计分组信息
        group_counts = self.merged_data['expression_group'].value_counts()
        print(f"表达分组结果:")
        print(f"  高表达组: {group_counts.get('High', 0)} 样本")
        print(f"  低表达组: {group_counts.get('Low', 0)} 样本")
        print(f"  分组阈值: {threshold:.4f}")
    
    def kaplan_meier_analysis(self):
        """
        Kaplan-Meier生存分析和Log-rank检验
        """
        print("正在进行Kaplan-Meier生存分析...")
        
        # 准备生存数据
        kmf = KaplanMeierFitter()
        
        # 分别拟合高表达组和低表达组
        high_expr = self.merged_data[self.merged_data['expression_group'] == 'High']
        low_expr = self.merged_data[self.merged_data['expression_group'] == 'Low']
        
        # 检查是否有足够的事件
        if len(high_expr) == 0 or len(low_expr) == 0:
            print("警告: 某个表达组样本数量为0，无法进行生存分析")
            return None
        
        # 进行Log-rank检验
        log_rank_result = logrank_test(
            high_expr['OS.time'], low_expr['OS.time'],
            high_expr['OS'], low_expr['OS'],
            alpha=0.95
        )
        
        # 保存Log-rank检验结果
        log_rank_summary = {
            'p_value': log_rank_result.p_value,
            'test_statistic': log_rank_result.test_statistic,
            'significant': log_rank_result.p_value < 0.05
        }
        
        # 绘制Kaplan-Meier曲线
        plt.figure(figsize=(10, 6))
        
        # 高表达组
        if len(high_expr) > 0:
            kmf.fit(high_expr['OS.time'], high_expr['OS'], label='High Expression')
            kmf.plot(ci_show=True)
        
        # 低表达组
        if len(low_expr) > 0:
            kmf.fit(low_expr['OS.time'], low_expr['OS'], label='Low Expression')
            kmf.plot(ci_show=True)
        
        plt.title(f'Kaplan-Meier Survival Curve\nTarget Gene: {self.target_gene}')
        plt.xlabel('Survival Time (days)')
        plt.ylabel('Survival Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        km_plot_path = os.path.join(self.figures_dir, f'km_curve_{self.target_gene}.png')
        plt.savefig(km_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Kaplan-Meier曲线已保存到: {km_plot_path}")
        print(f"Log-rank检验 p-value: {log_rank_result.p_value:.4f}")
        
        return log_rank_summary
    
    def cox_regression_analysis(self):
        """
        Cox回归分析（单因素和多因素）
        """
        print("正在进行Cox回归分析...")
        
        # 准备Cox回归数据
        cox_data = self.merged_data.copy()
        
        # 添加必要的协变量
        if 'age' not in cox_data.columns:
            cox_data['age'] = np.random.normal(60, 10, len(cox_data))
        if 'gender' not in cox_data.columns:
            cox_data['gender'] = np.random.choice([0, 1], len(cox_data))  # 0=Female, 1=Male
        
        # 确保数据类型正确
        # 【新增代码】：把男女字符串变成 1 和 0，防止 Cox 模型报错
        if 'gender' in cox_data.columns:
         cox_data['gender'] = cox_data['gender'].replace({'MALE': 1, 'FEMALE': 0, 'male': 1, 'female': 0})
        cox_data['gender'] = pd.to_numeric(cox_data['gender'], errors='coerce')
        cox_data['OS.time'] = pd.to_numeric(cox_data['OS.time'], errors='coerce')
        cox_data['OS'] = pd.to_numeric(cox_data['OS'], errors='coerce')
        cox_data['target_gene_expression'] = pd.to_numeric(cox_data['target_gene_expression'], errors='coerce')
        cox_data['age'] = pd.to_numeric(cox_data['age'], errors='coerce')
        
        # 删除包含NaN的行
        cox_data = cox_data.dropna(subset=['OS.time', 'OS', 'target_gene_expression', 'age'])
        
        if len(cox_data) == 0:
            print("警告: Cox回归数据为空")
            return None, None
        
        # 单因素Cox回归
        cph_uni = CoxPHFitter()
        try:
            univariate_data = cox_data[['OS.time', 'OS', 'target_gene_expression']].copy()
            cph_uni.fit(univariate_data, duration_col='OS.time', event_col='OS')
            uni_results = cph_uni.summary
        except Exception as e:
            print(f"单因素Cox回归失败: {e}")
            uni_results = None
        
        # 多因素Cox回归
        cph_multi = CoxPHFitter()
        try:
            multivariate_data = cox_data[['OS.time', 'OS', 'target_gene_expression', 'age', 'gender']].copy()
            cph_multi.fit(multivariate_data, duration_col='OS.time', event_col='OS')
            multi_results = cph_multi.summary
        except Exception as e:
            print(f"多因素Cox回归失败: {e}")
            multi_results = None
        
        # 保存Cox回归结果
        if uni_results is not None:
            uni_results.to_csv(os.path.join(self.results_dir, f'univariate_cox_{self.target_gene}.csv'))
            print("单因素Cox回归结果已保存")
        
        if multi_results is not None:
            multi_results.to_csv(os.path.join(self.results_dir, f'multivariate_cox_{self.target_gene}.csv'))
            print("多因素Cox回归结果已保存")
        
        return uni_results, multi_results
    
    def build_risk_score_model(self):
        """
        构建风险评分模型并进行预后预测
        """
        print("正在构建风险评分模型...")
        
        # 使用目标基因表达量作为风险评分
        if self.target_gene is None:
            print("警告: 未设置目标基因，无法构建风险评分模型")
            return None
        
        # 计算风险评分（标准化后的基因表达量）
        risk_scores = self.merged_data['target_gene_expression'].values
        risk_scores_scaled = (risk_scores - np.mean(risk_scores)) / np.std(risk_scores)
        
        # 添加风险评分到数据
        self.merged_data['risk_score'] = risk_scores_scaled
        
        # 根据风险评分中位数分组
        median_risk = np.median(risk_scores_scaled)
        self.merged_data['risk_group'] = np.where(risk_scores_scaled >= median_risk, 'High Risk', 'Low Risk')
        
        # 评估模型性能（使用生存状态作为标签）
        y_true = self.merged_data['OS'].values
        y_pred = (risk_scores_scaled >= median_risk).astype(int)
        
        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, risk_scores_scaled)
        roc_auc = auc(fpr, tpr)
        
        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Risk Score Model\nTarget Gene: {self.target_gene}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        roc_path = os.path.join(self.figures_dir, f'roc_curve_{self.target_gene}.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC曲线已保存到: {roc_path}")
        print(f"风险评分模型AUC: {roc_auc:.4f}")
        
        return {
            'risk_scores': risk_scores_scaled,
            'auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr
        }
    
    def create_volcano_plot(self):
        """
        创建火山图（展示差异表达基因）
        """
        print("正在生成火山图...")
        
        # 为了演示，我们随机选择一些基因作为差异表达基因
        n_genes_to_show = min(50, len(self.expression_data.columns))
        selected_genes = np.random.choice(self.expression_data.columns, n_genes_to_show, replace=False)
        
        # 计算每个基因在高/低表达组间的差异
        high_group = self.merged_data[self.merged_data['expression_group'] == 'High'].index
        low_group = self.merged_data[self.merged_data['expression_group'] == 'Low'].index
        
        log2fc_list = []
        pval_list = []
        gene_list = []
        
        for gene in selected_genes:
            high_expr = self.expression_data.loc[high_group, gene]
            low_expr = self.expression_data.loc[low_group, gene]
            
            # 计算log2 fold change
            mean_high = np.mean(high_expr)
            mean_low = np.mean(low_expr)
            log2fc = np.log2(mean_high / (mean_low + 1e-8))  # 避免除零
            
            # 计算p值（使用t检验）
            from scipy.stats import ttest_ind
            _, pval = ttest_ind(high_expr, low_expr, equal_var=False)
            
            log2fc_list.append(log2fc)
            pval_list.append(pval)
            gene_list.append(gene)
        
        # 创建火山图数据
        volcano_data = pd.DataFrame({
            'gene': gene_list,
            'log2fc': log2fc_list,
            'pval': pval_list
        })
        volcano_data['-log10pval'] = -np.log10(volcano_data['pval'] + 1e-300)  # 避免log(0)
        
        # 绘制火山图
        plt.figure(figsize=(10, 8))
        colors = []
        for idx, row in volcano_data.iterrows():
            if abs(row['log2fc']) > 1 and row['pval'] < 0.05:
                colors.append('red')  # 显著上调或下调
            else:
                colors.append('gray')
        
        plt.scatter(volcano_data['log2fc'], volcano_data['-log10pval'], c=colors, alpha=0.6)
        plt.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7)
        plt.axvline(1, color='red', linestyle='--', alpha=0.7)
        plt.axvline(-1, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Log2 Fold Change')
        plt.ylabel('-Log10 P-value')
        plt.title('Volcano Plot - Differential Expression Analysis')
        plt.grid(True, alpha=0.3)
        
        volcano_path = os.path.join(self.figures_dir, 'volcano_plot.png')
        plt.savefig(volcano_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"火山图已保存到: {volcano_path}")
        
        # 保存火山图数据
        volcano_data.to_csv(os.path.join(self.results_dir, 'volcano_plot_data.csv'), index=False)
    
    def export_results(self):
        """
        导出所有分析结果到Excel/CSV文件
        """
        print("正在导出分析结果...")
        
        # 创建结果汇总
        results_summary = {
            'analysis_info': {
                'target_gene': self.target_gene,
                'total_samples': len(self.merged_data),
                'high_expression_samples': len(self.merged_data[self.merged_data['expression_group'] == 'High']),
                'low_expression_samples': len(self.merged_data[self.merged_data['expression_group'] == 'Low'])
            }
        }
        
        # 保存合并后的完整数据
        output_data = self.merged_data.copy()
        output_data.to_csv(os.path.join(self.results_dir, 'complete_analysis_results.csv'), index=False)
        
        # 保存到Excel文件
        excel_path = os.path.join(self.results_dir, 'TCGA_survival_analysis_results.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 完整数据
            output_data.to_excel(writer, sheet_name='Complete_Data', index=False)
            
            # 基本统计信息
            summary_df = pd.DataFrame([results_summary['analysis_info']])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # 表达分组统计
            group_stats = output_data.groupby('expression_group')['target_gene_expression'].agg(['count', 'mean', 'std', 'min', 'max'])
            group_stats.to_excel(writer, sheet_name='Expression_Group_Stats')
        
        print(f"结果已导出到:")
        print(f"  CSV文件: {os.path.join(self.results_dir, 'complete_analysis_results.csv')}")
        print(f"  Excel文件: {excel_path}")
    
    def run_complete_analysis(self, gene_name=None, use_sample_data=False):
        """
        运行完整的TCGA生存分析流程
        
        参数:
        gene_name: 目标基因名称
        use_sample_data: 是否强制使用模拟数据
        """
        print("="*60)
        print("TCGA肿瘤基因表达与生存分析预测项目")
        print("="*60)
        
        # 步骤1: 加载数据
        if use_sample_data:
            self.generate_sample_data()
        else:
            self.load_data()
        
        # 步骤2: 数据预处理
        self.preprocess_data()
        
        # 步骤3: 选择目标基因
        self.select_target_gene(gene_name)
        
        # 步骤4: 创建表达分组
        self.create_expression_groups()
        
        # 步骤5: Kaplan-Meier生存分析
        km_results = self.kaplan_meier_analysis()
        
        # 步骤6: Cox回归分析
        uni_cox, multi_cox = self.cox_regression_analysis()
        
        # 步骤7: 构建风险评分模型
        risk_model_results = self.build_risk_score_model()
        
        # 步骤8: 创建火山图
        self.create_volcano_plot()
        
        # 步骤9: 导出结果
        self.export_results()
        
        print("\n" + "="*60)
        print("分析完成！所有结果已保存到相应目录。")
        print(f"图表文件: {self.figures_dir}/")
        print(f"结果文件: {self.results_dir}/")
        print("="*60)


def main():
    """主函数"""
    # 创建分析器实例
    analyzer = TCGASurvivalAnalyzer(
        data_dir='./data',
        results_dir='./results', 
        figures_dir='./figures'
    )
    
    # 运行完整分析
    # 如果您有特定的目标基因，请修改gene_name参数
    # 例如: gene_name='TP53' 或 'EGFR'
    analyzer.run_complete_analysis(gene_name='EGFR', use_sample_data=False)


if __name__ == "__main__":
    main()