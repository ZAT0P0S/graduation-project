# 0. 导入所需库
import pandas as pd
import numpy as np
from collections import Counter
import re

# 机器学习相关库
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, \
    StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
                             precision_recall_curve, auc, ConfusionMatrixDisplay, RocCurveDisplay,
                             PrecisionRecallDisplay,
                             silhouette_score)
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import clone
import xgboost as xgb
from sklearn.metrics.pairwise import cosine_similarity

# SHAP (如果安装了)
try:
    import shap

    SHAP_INSTALLED = True
except ImportError:
    SHAP_INSTALLED = False
    print("警告: SHAP库未安装。模型解释部分将无法执行。请运行 'pip install shap' 进行安装。")

# 可视化库
import matplotlib.pyplot as plt
import seaborn as sns

# 设置Matplotlib中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("已设置Matplotlib中文字体为SimHei (全局)。")
except Exception as e_font:
    print(f"设置Matplotlib中文字体失败 (全局): {e_font}。部分图表中文可能无法正确显示。")

# 定义行为类型和数据文件路径
DEFINED_ACTION_TYPES = ['view', 'detail_view', 'add_to_cart', 'favorite', 'share']
DATA_FILE_PATH = '数据集v4.csv'


# --- 2. 数据加载与预处理 ---
def load_and_preprocess_data(file_path):
    print(f"开始从以下路径加载数据: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print("成功作为 CSV 文件读取。")
    except Exception as e_csv:
        print(f"作为 CSV 读取失败: {e_csv}。请检查文件路径和格式。")
        return None
    print("\n--- 原始数据信息 ---")
    df.info(verbose=True, show_counts=True)
    expected_columns = ['user_id', 'age', 'gender', 'behavior_sequence',
                        'search_sequence', 'purchase_history', 'purchase_label']
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        print(f"严重错误: 数据集中缺少以下必需列: {missing_cols}")
        return None
    df['user_id'] = df['user_id'].astype(str)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['purchase_label'] = pd.to_numeric(df['purchase_label'], errors='coerce').fillna(0).astype(int)
    df['gender'] = df['gender'].astype(str).str.lower().str.strip()
    df['gender'] = df['gender'].replace(['nan', '', 'unknown', '保密'], 'other')
    sequence_cols = ['behavior_sequence', 'search_sequence', 'purchase_history']
    for col in sequence_cols:
        df[col] = df[col].fillna('').astype(str)
    df.dropna(subset=['age'], inplace=True)
    print("\n--- 数据预处理后信息 ---")
    df.info(verbose=True, show_counts=True)
    print("\n--- 处理后数据前5行预览 ---")
    pd.set_option('display.max_colwidth', 50)
    print(df.head())
    print(f"\n数据形状: {df.shape}")
    return df


# --- 辅助函数：解析序列数据 ---
def parse_rich_action_sequence(sequence_str):
    if not isinstance(sequence_str, str) or not sequence_str.strip(): return []
    parsed_sequence = []
    sequence_str = sequence_str.strip("[]'")
    actions_items_pairs = sequence_str.split(',')
    for pair_str in actions_items_pairs:
        pair_str = pair_str.strip()
        if not pair_str: continue
        parts = pair_str.split(':', 1)
        if len(parts) == 2:
            action, item = parts[0].strip(), parts[1].strip()
            if action in DEFINED_ACTION_TYPES and item:
                parsed_sequence.append((action, item))
            elif item:
                parsed_sequence.append(('unknown_action', item))
        elif len(parts) == 1 and parts[0]:
            parsed_sequence.append(('implicit_interaction', parts[0]))
    return parsed_sequence


def parse_simple_sequence(sequence_str):
    if not isinstance(sequence_str, str) or not sequence_str.strip(): return []
    sequence_str = sequence_str.strip("[]'")
    return [item.strip() for item in sequence_str.split(',') if item.strip()]


# --- 辅助函数：绘制K-means评估图表 ---
def plot_kmeans_evaluation_graphs(data_for_clustering, max_k=10):
    if data_for_clustering is None or data_for_clustering.empty:
        print("用于K-Means评估的数据为空，跳过K值选择图表绘制。")
        return
    print("\n--- 生成K-Means K值选择辅助图表 ---")
    wcss = []
    silhouette_scores = []
    k_values_wcss = range(1, max_k + 1)
    k_values_silhouette = range(2, max_k + 1)

    for k_w in k_values_wcss:
        kmeans_eval_wcss = KMeans(n_clusters=k_w, random_state=42, n_init='auto')
        kmeans_eval_wcss.fit(data_for_clustering)
        wcss.append(kmeans_eval_wcss.inertia_)

    for k_s in k_values_silhouette:
        kmeans_eval_sil = KMeans(n_clusters=k_s, random_state=42, n_init='auto')
        cluster_labels_eval = kmeans_eval_sil.fit_predict(data_for_clustering)
        if len(np.unique(cluster_labels_eval)) > 1:
            silhouette_avg = silhouette_score(data_for_clustering, cluster_labels_eval)
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(-1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_values_wcss, wcss, marker='o', linestyle='--')
    plt.title('K-Means 肘部法则图')
    plt.xlabel('聚类数量 (K)')
    plt.ylabel('簇内平方和 (WCSS)')
    plt.xticks(k_values_wcss)

    if silhouette_scores:
        plt.subplot(1, 2, 2)
        plt.plot(k_values_silhouette, silhouette_scores, marker='o', linestyle='--')
        plt.title('K-Means 轮廓系数图')
        plt.xlabel('聚类数量 (K)')
        plt.ylabel('平均轮廓系数')
        plt.xticks(k_values_silhouette)

    plt.tight_layout();
    plt.savefig('kmeans_evaluation_graphs.png');
    plt.close()
    print("K-Means K值选择辅助图表 (肘部法则和轮廓系数) 已保存。")


# --- 3. 用户行为分析 ---
def analyze_user_behavior(df_with_all_features, features_for_clustering_df):
    if df_with_all_features is None or df_with_all_features.empty:
        print("用户行为分析跳过：原始数据为空。")
        return
    print("\n\n--- 用户行为分析 (增强版：包含聚类和K值选择图) ---")
    print(f"总用户数: {len(df_with_all_features)}")
    purchase_label_counts = df_with_all_features['purchase_label'].value_counts(normalize=True) * 100
    print(f"购买标签分布 (%):\n{purchase_label_counts}")

    plt.figure(figsize=(8, 6));
    sns.histplot(df_with_all_features['age'].dropna(), kde=True, bins=20)
    plt.title('用户年龄分布');
    plt.xlabel('年龄');
    plt.ylabel('用户数量');
    plt.tight_layout()
    plt.savefig('user_age_distribution.png');
    plt.close()
    print("用户年龄分布图已保存。")

    gender_counts = df_with_all_features['gender'].value_counts()
    if not gender_counts.empty:
        plt.figure(figsize=(6, 4));
        sns.barplot(x=gender_counts.index, y=gender_counts.values)
        plt.title('用户性别分布');
        plt.xlabel('性别');
        plt.ylabel('用户数量');
        plt.tight_layout()
        plt.savefig('user_gender_distribution.png');
        plt.close()
        print("用户性别分布图已保存。")
    else:
        print("性别数据为空，无法生成性别分布图。")

    if 'behavior_sequence' in df_with_all_features.columns:
        temp_parsed_behavior = df_with_all_features['behavior_sequence'].apply(parse_rich_action_sequence)
        all_actions_counter = Counter()
        for seq in temp_parsed_behavior:
            for action, item in seq: all_actions_counter[action] += 1
        if all_actions_counter:
            action_freq_df = pd.DataFrame.from_dict(all_actions_counter, orient='index', columns=['次数'])
            action_freq_df = action_freq_df.sort_values(by='次数', ascending=False)
            print("\n--- 各种行为类型总发生次数 ---");
            print(action_freq_df)
            plt.figure(figsize=(10, 6));
            sns.barplot(x=action_freq_df.index, y='次数', data=action_freq_df)
            plt.title('行为类型频率分布');
            plt.xticks(rotation=45, ha='right');
            plt.tight_layout()
            plt.savefig('rich_action_type_distribution.png');
            plt.close()
            print("丰富行为类型分布图已保存。")
        else:
            print("未能从行为序列中解析出任何具体行为。")

        high_intent_actions = ['add_to_cart', 'favorite']
        product_high_intent_counter = Counter()
        for seq in temp_parsed_behavior:
            for action, item in seq:
                if action in high_intent_actions: product_high_intent_counter[item] += 1
        print(f"\n--- Top 10 被执行高意图行为 ({'/'.join(high_intent_actions)}) 最多的商品 ---")
        if product_high_intent_counter:
            print(pd.Series(product_high_intent_counter).sort_values(ascending=False).head(10))
        else:
            print(f"未能发现任何 '{'/'.join(high_intent_actions)}' 行为。")
    else:
        print("警告: 'behavior_sequence' 列不在DataFrame中，无法进行行为类型频率分析。")

    if 'total_actions_in_behavior_seq' in df_with_all_features.columns:
        plt.figure(figsize=(8, 6));
        sns.boxplot(x='purchase_label', y='total_actions_in_behavior_seq', data=df_with_all_features)
        plt.title('购买用户与未购买用户的总行为次数对比');
        plt.xlabel('是否购买 (Purchase Label)');
        plt.ylabel('总行为次数')
        if not df_with_all_features['total_actions_in_behavior_seq'].empty:
            plt.ylim(0, df_with_all_features['total_actions_in_behavior_seq'].quantile(0.98) + 1)
        plt.tight_layout();
        plt.savefig('actions_count_vs_purchase_label.png');
        plt.close()
        print("购买用户与未购买用户的总行为次数对比图已保存。")
    else:
        print("警告: 'total_actions_in_behavior_seq' 列不在DataFrame中，无法生成行为次数对比图。")

    if features_for_clustering_df is not None and not features_for_clustering_df.empty:
        plot_kmeans_evaluation_graphs(features_for_clustering_df, max_k=10)
        print("\n--- 开始进行用户聚类 (K-means) ---");
        num_clusters = 4
        print(f"选定聚类数量 K = {num_clusters}")
        try:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(features_for_clustering_df)
            df_for_cluster_analysis = df_with_all_features.copy()
            if features_for_clustering_df.index.equals(df_for_cluster_analysis.index):
                df_for_cluster_analysis['cluster'] = cluster_labels
            elif features_for_clustering_df.index.isin(df_for_cluster_analysis.index).all() and \
                    len(features_for_clustering_df.index.unique()) == len(cluster_labels):
                temp_labels_series = pd.Series(cluster_labels, index=features_for_clustering_df.index)
                df_for_cluster_analysis['cluster'] = temp_labels_series
            else:
                if len(cluster_labels) == len(df_for_cluster_analysis):
                    df_for_cluster_analysis['cluster'] = cluster_labels
                    print("警告: 聚类标签按顺序赋值，请确保索引一致性以进行准确分析。")
                else:
                    print(
                        f"错误: cluster_labels 长度 ({len(cluster_labels)}) 与目标DataFrame行数 ({len(df_for_cluster_analysis)}) 不匹配。跳过聚类标签添加。")

            if 'cluster' in df_for_cluster_analysis.columns and not df_for_cluster_analysis['cluster'].isnull().all():
                print(f"各用户群数量分布:\n{df_for_cluster_analysis['cluster'].value_counts().sort_index()}")
                cols_for_cluster_analysis = ['age', 'total_actions_in_behavior_seq', 'purchase_history_len',
                                             'count_add_to_cart', 'count_favorite', 'purchase_label']
                existing_cols_for_analysis = [col for col in cols_for_cluster_analysis if
                                              col in df_for_cluster_analysis.columns]
                if existing_cols_for_analysis:
                    cluster_analysis_means = df_for_cluster_analysis.groupby('cluster')[
                        existing_cols_for_analysis].mean()
                    print("\n各用户群特征均值对比:");
                    pd.set_option('display.max_columns', None)
                    print(cluster_analysis_means)
                    if 'purchase_label' in cluster_analysis_means.columns:
                        cluster_purchase_rate = cluster_analysis_means['purchase_label']
                        plt.figure(figsize=(8, 5));
                        sns.barplot(x=cluster_purchase_rate.index, y=cluster_purchase_rate.values)
                        plt.title('各用户群的平均购买率');
                        plt.xlabel('用户群 (Cluster)');
                        plt.ylabel('平均购买率 (Average Purchase Rate)')
                        plt.tight_layout();
                        plt.savefig('cluster_purchase_rates.png');
                        plt.close()
                        print("各用户群购买率对比图已保存。")
                else:
                    print("警告: 用于聚类特性分析的列在DataFrame中未完全找到。")
            else:
                print("未能将聚类标签添加到DataFrame中进行分析或'cluster'列全为NaN。")
        except Exception as e_cluster:
            print(f"用户聚类时发生错误: {e_cluster}")
    else:
        print("未提供用于聚类的特征数据或数据为空，跳过用户聚类和K值选择图。")


# --- 4. 购买预测的特征工程 ---
def feature_engineering_for_prediction(df_orig):
    if df_orig is None or df_orig.empty: return None, None, None, None
    df = df_orig.copy()
    print("\n\n--- 购买预测的特征工程 (基于丰富行为数据) ---")
    df['parsed_behavior'] = df['behavior_sequence'].apply(parse_rich_action_sequence)
    for action_type in DEFINED_ACTION_TYPES:
        df[f'count_{action_type}'] = df['parsed_behavior'].apply(
            lambda seq: sum(1 for action, item in seq if action == action_type))
        df[f'unique_items_{action_type}'] = df['parsed_behavior'].apply(
            lambda seq: len(set(item for action, item in seq if action == action_type)))
    df['total_actions_in_behavior_seq'] = df['parsed_behavior'].apply(len)
    df['unique_items_in_behavior_seq'] = df['parsed_behavior'].apply(lambda seq: len(set(item for action, item in seq)))
    df['has_added_to_cart'] = (df['count_add_to_cart'] > 0).astype(int)
    df['has_favorited'] = (df['count_favorite'] > 0).astype(int)
    df['search_seq_list'] = df['search_sequence'].apply(parse_simple_sequence)
    df['search_seq_len'] = df['search_seq_list'].apply(len)
    df['search_seq_unique_count'] = df['search_seq_list'].apply(lambda x: len(set(x)))
    df['purchase_history_list'] = df['purchase_history'].apply(parse_simple_sequence)
    df['purchase_history_len'] = df['purchase_history_list'].apply(len)

    feature_cols_numerical_raw = ['age', 'total_actions_in_behavior_seq', 'unique_items_in_behavior_seq',
                                  'search_seq_len', 'search_seq_unique_count', 'purchase_history_len']
    for action_type in DEFINED_ACTION_TYPES:
        feature_cols_numerical_raw.append(f'count_{action_type}')
        feature_cols_numerical_raw.append(f'unique_items_{action_type}')
    feature_cols_categorical_raw = ['gender']
    feature_cols_boolean_raw = ['has_added_to_cart', 'has_favorited']
    for col in feature_cols_numerical_raw:
        if col not in df.columns:
            df[col] = 0
        elif not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    for col in feature_cols_boolean_raw:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].astype(int)
    for col in feature_cols_categorical_raw:
        if col not in df.columns: df[col] = 'unknown'
        df[col] = df[col].astype(str).fillna('unknown')
    y = df['purchase_label'].copy()
    final_numerical_features_for_preprocessor = sorted(list(set(feature_cols_numerical_raw + feature_cols_boolean_raw)))
    final_categorical_features_for_preprocessor = sorted(list(set(feature_cols_categorical_raw)))
    preprocessor_definition = ColumnTransformer(
        transformers=[('num', StandardScaler(), final_numerical_features_for_preprocessor),
                      ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'),
                       final_categorical_features_for_preprocessor)],
        remainder='drop', verbose_feature_names_out=False)
    cols_for_X_raw = final_numerical_features_for_preprocessor + final_categorical_features_for_preprocessor
    X_raw_features_df = df[cols_for_X_raw].copy()
    for col in final_numerical_features_for_preprocessor:
        if not pd.api.types.is_numeric_dtype(X_raw_features_df[col]): X_raw_features_df[col] = pd.to_numeric(
            X_raw_features_df[col], errors='coerce').fillna(0)
    for col in final_categorical_features_for_preprocessor:
        X_raw_features_df[col] = X_raw_features_df[col].astype(str).fillna('unknown')
    print(f"原始特征集 X_raw_features_df 的形状 (预处理前): {X_raw_features_df.shape}")
    if y.nunique() < 2: print("严重警告: 目标变量中少于2个唯一值。"); return None, None, None, df
    df_with_all_features = df.drop(columns=['parsed_behavior', 'search_seq_list', 'purchase_history_list'],
                                   errors='ignore')
    return X_raw_features_df, y, preprocessor_definition, df_with_all_features


# --- 5. 购买预测模型训练与评估 ---
def train_and_evaluate_prediction_model(X_raw_features, y_target, preprocessor_definition,
                                        test_size=0.3, random_state=42):
    if X_raw_features is None or y_target is None or X_raw_features.empty:
        print("跳过模型训练，原始特征集(X_raw_features)或目标变量(y_target)为空。")
        return None, {}
    print("\n\n--- 购买预测模型训练、调优与评估 (使用Pipeline和GridSearchCV) ---")
    if not pd.api.types.is_numeric_dtype(y_target): y_target = pd.to_numeric(y_target, errors='coerce')
    y_target = y_target.fillna(0).astype(int)
    if y_target.nunique() < 2: print("目标变量y中唯一值少于2个，跳过模型训练。"); return None, {}
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw_features, y_target, test_size=test_size,
                                                                random_state=random_state, stratify=y_target)

    ## 新增代码开始 ##
    # 保存 y_test，因为它对于所有模型都是一样的
    try:
        np.save('y_test.npy', y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test)  # 确保保存为numpy array
        print("y_test 已成功保存到 y_test.npy")
    except Exception as e_save_ytest:
        print(f"保存 y_test 时出错: {e_save_ytest}")
    ## 新增代码结束 ##

    trained_models_pipelines = {}
    model_performance = {}
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    scoring_metric = 'average_precision'

    def plot_model_evaluation_charts(model_for_plot, model_name_str, y_true, y_pred, y_pred_proba):
        try:
            cm = confusion_matrix(y_true, y_pred, labels=model_for_plot.classes_)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_for_plot.classes_)
            disp.plot(cmap=plt.cm.Blues);
            plt.title(f'{model_name_str} 混淆矩阵')
            plt.savefig(f'confusion_matrix_{model_name_str}.png');
            plt.close()
            print(f"{model_name_str} 混淆矩阵图已保存。")
        except Exception as e_cm:
            print(f"绘制{model_name_str}混淆矩阵失败: {e_cm}")
        if y_pred_proba is not None and len(np.unique(y_true)) > 1:
            try:
                RocCurveDisplay.from_predictions(y_true, y_pred_proba, name=model_name_str)
                plt.title(f'{model_name_str} ROC 曲线');
                plt.savefig(f'roc_curve_{model_name_str}.png');
                plt.close()
                print(f"{model_name_str} ROC 曲线图已保存。")
            except Exception as e_roc:
                print(f"绘制{model_name_str}ROC曲线失败: {e_roc}")
        if y_pred_proba is not None and len(np.unique(y_true)) > 1:
            try:
                PrecisionRecallDisplay.from_predictions(y_true, y_pred_proba, name=model_name_str)
                plt.title(f'{model_name_str} PR 曲线');
                plt.savefig(f'pr_curve_{model_name_str}.png');
                plt.close()
                print(f"{model_name_str} PR 曲线图已保存。")
            except Exception as e_pr:
                print(f"绘制{model_name_str}PR曲线失败: {e_pr}")

    # --- 模型1: 逻辑回归 ---
    print("\n--- 开始训练和调优逻辑回归模型 ---")
    pipeline_lr = Pipeline([('preprocessor', preprocessor_definition), (
    'model', LogisticRegression(random_state=random_state, solver='liblinear', max_iter=2000))])
    param_grid_lr = {'model__C': [0.1, 1.0, 10.0], 'model__penalty': ['l1', 'l2'],
                     'model__class_weight': ['balanced', None]}
    grid_search_lr = GridSearchCV(estimator=pipeline_lr, param_grid=param_grid_lr, scoring=scoring_metric,
                                  cv=cv_strategy, n_jobs=-1, verbose=1)
    grid_search_lr.fit(X_train_raw, y_train)
    best_pipeline_lr = grid_search_lr.best_estimator_
    log_reg_model_actual = best_pipeline_lr.named_steps['model']
    print(f"逻辑回归 - 最佳参数: {grid_search_lr.best_params_}")
    print(f"逻辑回归 - 最佳 {scoring_metric} (CV): {grid_search_lr.best_score_:.4f}")
    trained_models_pipelines['LogisticRegression'] = best_pipeline_lr
    y_pred_log_reg = best_pipeline_lr.predict(X_test_raw)
    y_pred_proba_log_reg = best_pipeline_lr.predict_proba(X_test_raw)[:, 1] if hasattr(best_pipeline_lr,
                                                                                       "predict_proba") and len(
        log_reg_model_actual.classes_) == 2 else None

    ## 新增代码开始 ##
    if y_pred_proba_log_reg is not None:
        try:
            np.save('y_pred_proba_log_reg.npy', y_pred_proba_log_reg)
            print("y_pred_proba_log_reg 已成功保存到 y_pred_proba_log_reg.npy")
        except Exception as e_save_lr:
            print(f"保存 y_pred_proba_log_reg 时出错: {e_save_lr}")
    else:
        print("y_pred_proba_log_reg 为 None，无法保存。")
    ## 新增代码结束 ##

    lr_accuracy = accuracy_score(y_test, y_pred_log_reg)
    lr_precision = precision_score(y_test, y_pred_log_reg, pos_label=1, zero_division=0)
    lr_recall = recall_score(y_test, y_pred_log_reg, pos_label=1, zero_division=0)
    lr_f1 = f1_score(y_test, y_pred_log_reg, pos_label=1, zero_division=0)
    lr_roc_auc = roc_auc_score(y_test, y_pred_proba_log_reg) if y_pred_proba_log_reg is not None else 'N/A'
    lr_pr_auc_val = 'N/A'
    if y_pred_proba_log_reg is not None:
        precision_vals_lr, recall_vals_lr, _ = precision_recall_curve(y_test, y_pred_proba_log_reg, pos_label=1)
        lr_pr_auc_val = auc(recall_vals_lr, precision_vals_lr)
    model_performance['LogisticRegression'] = {'accuracy': lr_accuracy, 'precision': lr_precision, 'recall': lr_recall,
                                               'f1': lr_f1, 'roc_auc': lr_roc_auc, 'pr_auc': lr_pr_auc_val}
    print("\n逻辑回归模型 (调优后) - 真实评估结果:")
    print(
        f"  准确率: {lr_accuracy:.4f}, 精确率(P1): {lr_precision:.4f}, 召回率(R1): {lr_recall:.4f}, F1分数(F1_1): {lr_f1:.4f}")
    print(
        f"  ROC AUC: {lr_roc_auc if isinstance(lr_roc_auc, float) else lr_roc_auc:.4f}, PR AUC: {lr_pr_auc_val if isinstance(lr_pr_auc_val, float) else lr_pr_auc_val:.4f}")
    plot_model_evaluation_charts(log_reg_model_actual, 'LogisticRegression', y_test, y_pred_log_reg,
                                 y_pred_proba_log_reg)

    # --- 模型2: 随机森林 ---
    print("\n--- 开始训练和调优随机森林模型 ---")
    pipeline_rf = Pipeline([('preprocessor', preprocessor_definition),
                            ('model', RandomForestClassifier(random_state=random_state, n_jobs=-1))])
    param_grid_rf = {'model__n_estimators': [100, 150], 'model__max_depth': [10, 20],
                     'model__min_samples_split': [2, 5], 'model__min_samples_leaf': [1, 3],
                     'model__class_weight': ['balanced_subsample', 'balanced']}
    grid_search_rf = GridSearchCV(estimator=pipeline_rf, param_grid=param_grid_rf, scoring=scoring_metric,
                                  cv=cv_strategy, n_jobs=-1, verbose=1)
    grid_search_rf.fit(X_train_raw, y_train)
    best_pipeline_rf = grid_search_rf.best_estimator_
    rf_model_actual = best_pipeline_rf.named_steps['model']
    print(f"随机森林 - 最佳参数: {grid_search_rf.best_params_}")
    print(f"随机森林 - 最佳 {scoring_metric} (CV): {grid_search_rf.best_score_:.4f}")
    trained_models_pipelines['RandomForest'] = best_pipeline_rf
    y_pred_rf = best_pipeline_rf.predict(X_test_raw)
    y_pred_proba_rf = best_pipeline_rf.predict_proba(X_test_raw)[:, 1] if hasattr(best_pipeline_rf,
                                                                                  "predict_proba") and len(
        rf_model_actual.classes_) == 2 else None

    ## 新增代码开始 ##
    if y_pred_proba_rf is not None:
        try:
            np.save('y_pred_proba_rf.npy', y_pred_proba_rf)
            print("y_pred_proba_rf 已成功保存到 y_pred_proba_rf.npy")
        except Exception as e_save_rf:
            print(f"保存 y_pred_proba_rf 时出错: {e_save_rf}")
    else:
        print("y_pred_proba_rf 为 None，无法保存。")
    ## 新增代码结束 ##

    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_precision = precision_score(y_test, y_pred_rf, pos_label=1, zero_division=0)
    rf_recall = recall_score(y_test, y_pred_rf, pos_label=1, zero_division=0)
    rf_f1 = f1_score(y_test, y_pred_rf, pos_label=1, zero_division=0)
    rf_roc_auc = roc_auc_score(y_test, y_pred_proba_rf) if y_pred_proba_rf is not None else 'N/A'
    rf_pr_auc_val = 'N/A'
    if y_pred_proba_rf is not None:
        precision_vals_rf, recall_vals_rf, _ = precision_recall_curve(y_test, y_pred_proba_rf, pos_label=1)
        rf_pr_auc_val = auc(recall_vals_rf, precision_vals_rf)
    model_performance['RandomForest'] = {'accuracy': rf_accuracy, 'precision': rf_precision, 'recall': rf_recall,
                                         'f1': rf_f1, 'roc_auc': rf_roc_auc, 'pr_auc': rf_pr_auc_val}
    print("\n随机森林模型 (调优后) - 真实评估结果:")
    print(
        f"  准确率: {rf_accuracy:.4f}, 精确率(P1): {rf_precision:.4f}, 召回率(R1): {rf_recall:.4f}, F1分数(F1_1): {rf_f1:.4f}")
    print(
        f"  ROC AUC: {rf_roc_auc if isinstance(rf_roc_auc, float) else rf_roc_auc:.4f}, PR AUC: {rf_pr_auc_val if isinstance(rf_pr_auc_val, float) else rf_pr_auc_val:.4f}")
    plot_model_evaluation_charts(rf_model_actual, 'RandomForest', y_test, y_pred_rf, y_pred_proba_rf)

    # --- 模型3: XGBoost ---
    print("\n--- 开始训练和调优 XGBoost 模型 ---")
    scale_pos_weight_val = (y_train.value_counts().get(0, 1) / y_train.value_counts().get(1,
                                                                                          1)) if 1 in y_train.value_counts() and y_train.value_counts().get(
        1, 0) > 0 else 1.0
    pipeline_xgb = Pipeline([('preprocessor', preprocessor_definition), ('model',
                                                                         xgb.XGBClassifier(random_state=random_state,
                                                                                           eval_metric='logloss',
                                                                                           scale_pos_weight=scale_pos_weight_val))])
    param_grid_xgb = {'model__n_estimators': [100, 150], 'model__max_depth': [3, 5, 7],
                      'model__learning_rate': [0.05, 0.1], 'model__subsample': [0.7, 0.9],
                      'model__colsample_bytree': [0.7, 0.9]}
    try:
        grid_search_xgb = GridSearchCV(estimator=pipeline_xgb, param_grid=param_grid_xgb, scoring=scoring_metric,
                                       cv=cv_strategy, n_jobs=-1, verbose=1)
        grid_search_xgb.fit(X_train_raw, y_train)
        best_pipeline_xgb = grid_search_xgb.best_estimator_
        xgb_model_actual = best_pipeline_xgb.named_steps['model']
        print(f"XGBoost - 最佳参数: {grid_search_xgb.best_params_}")
        print(f"XGBoost - 最佳 {scoring_metric} (CV): {grid_search_xgb.best_score_:.4f}")
        trained_models_pipelines['XGBoost'] = best_pipeline_xgb
        y_pred_xgb = best_pipeline_xgb.predict(X_test_raw)
        y_pred_proba_xgb = best_pipeline_xgb.predict_proba(X_test_raw)[:, 1] if hasattr(best_pipeline_xgb,
                                                                                        "predict_proba") and len(
            xgb_model_actual.classes_) == 2 else None

        ## 新增代码开始 ##
        if y_pred_proba_xgb is not None:
            try:
                np.save('y_pred_proba_xgb.npy', y_pred_proba_xgb)
                print("y_pred_proba_xgb 已成功保存到 y_pred_proba_xgb.npy")
            except Exception as e_save_xgb:
                print(f"保存 y_pred_proba_xgb 时出错: {e_save_xgb}")
        else:
            print("y_pred_proba_xgb 为 None，无法保存。")
        ## 新增代码结束 ##

        xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
        xgb_precision = precision_score(y_test, y_pred_xgb, pos_label=1, zero_division=0)
        xgb_recall = recall_score(y_test, y_pred_xgb, pos_label=1, zero_division=0)
        xgb_f1 = f1_score(y_test, y_pred_xgb, pos_label=1, zero_division=0)
        xgb_roc_auc = roc_auc_score(y_test, y_pred_proba_xgb) if y_pred_proba_xgb is not None else 'N/A'
        xgb_pr_auc_val = 'N/A'
        if y_pred_proba_xgb is not None:
            precision_vals_xgb, recall_vals_xgb, _ = precision_recall_curve(y_test, y_pred_proba_xgb, pos_label=1)
            xgb_pr_auc_val = auc(recall_vals_xgb, precision_vals_xgb)
        model_performance['XGBoost'] = {'accuracy': xgb_accuracy, 'precision': xgb_precision, 'recall': xgb_recall,
                                        'f1': xgb_f1, 'roc_auc': xgb_roc_auc, 'pr_auc': xgb_pr_auc_val}
        print("\nXGBoost 模型 (调优后) - 真实评估结果:")
        print(
            f"  准确率: {xgb_accuracy:.4f}, 精确率(P1): {xgb_precision:.4f}, 召回率(R1): {xgb_recall:.4f}, F1分数(F1_1): {xgb_f1:.4f}")
        print(
            f"  ROC AUC: {xgb_roc_auc if isinstance(xgb_roc_auc, float) else xgb_roc_auc:.4f}, PR AUC: {xgb_pr_auc_val if isinstance(xgb_pr_auc_val, float) else xgb_pr_auc_val:.4f}")
        plot_model_evaluation_charts(xgb_model_actual, 'XGBoost', y_test, y_pred_xgb, y_pred_proba_xgb)
    except Exception as e_xgb_tune:
        print(f"训练或调优XGBoost模型时出错: {e_xgb_tune}.")
        trained_models_pipelines['XGBoost'] = None
        model_performance['XGBoost'] = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'roc_auc': 0, 'pr_auc': 0}

    best_model_name = None
    current_best_pr_auc = -1.0;
    current_best_f1 = -1.0;
    current_best_roc_auc = -1.0
    for name, perf in model_performance.items():
        if trained_models_pipelines.get(name) is None: continue
        pr_auc_val = perf.get('pr_auc', 0);
        f1_val = perf.get('f1', 0);
        roc_auc_val = perf.get('roc_auc', 0)
        pr_auc_val = pr_auc_val if isinstance(pr_auc_val, float) else -1.0
        f1_val = f1_val if isinstance(f1_val, float) else -1.0
        roc_auc_val = roc_auc_val if isinstance(roc_auc_val, float) else -1.0
        if pr_auc_val > current_best_pr_auc:
            current_best_pr_auc = pr_auc_val;
            current_best_f1 = f1_val;
            current_best_roc_auc = roc_auc_val
            best_model_name = name
        elif abs(pr_auc_val - current_best_pr_auc) < 1e-9:
            if f1_val > current_best_f1:
                current_best_f1 = f1_val;
                current_best_roc_auc = roc_auc_val;
                best_model_name = name
            elif abs(f1_val - current_best_f1) < 1e-9 and roc_auc_val > current_best_roc_auc:
                current_best_roc_auc = roc_auc_val;
                best_model_name = name
    if not best_model_name and model_performance:
        for name, perf in model_performance.items():
            if trained_models_pipelines.get(name) is None: continue
            f1_val = perf.get('f1', 0);
            f1_val = f1_val if isinstance(f1_val, float) else -1.0
            if f1_val > current_best_f1: current_best_f1 = f1_val; best_model_name = name

    if best_model_name and trained_models_pipelines.get(best_model_name):
        best_pipeline = trained_models_pipelines[best_model_name]
        best_model_actual = best_pipeline.named_steps['model']
        preprocessor_fitted = best_pipeline.named_steps['preprocessor']
        X_train_processed_array = preprocessor_fitted.transform(X_train_raw)
        X_test_processed_array = preprocessor_fitted.transform(X_test_raw)
        try:
            feature_names_processed = preprocessor_fitted.get_feature_names_out()
        except Exception:
            num_cols = X_test_processed_array.shape[1];
            feature_names_processed = [f"feature_{i}" for i in range(num_cols)]
            print(f"警告: 无法从预处理器获取特征名，使用通用名称。SHAP图的可解释性可能受影响。")
        X_train_processed_df = pd.DataFrame(X_train_processed_array, columns=feature_names_processed,
                                            index=X_train_raw.index)
        X_test_processed_df = pd.DataFrame(X_test_processed_array, columns=feature_names_processed,
                                           index=X_test_raw.index)
        print(f"\n--- {best_model_name} 模型被选为最佳模型进行可解释性分析 ---")
        feature_importance_df = None
        if hasattr(best_model_actual, 'feature_importances_'):
            importances = best_model_actual.feature_importances_
            feature_importance_df = pd.DataFrame(
                {'特征 (feature)': feature_names_processed, '重要性 (importance)': importances})
            feature_importance_df.sort_values(by='重要性 (importance)', ascending=False, inplace=True)
            print(feature_importance_df.head(min(20, len(feature_importance_df))))
            plt.figure(figsize=(12, max(8, len(feature_importance_df.head(15)) * 0.5)))
            sns.barplot(x='重要性 (importance)', y='特征 (feature)', data=feature_importance_df.head(15))
            plt.title(f'{best_model_name} 模型特征重要性 (Top 15)');
            plt.tight_layout()
            plt.savefig(f'feature_importances_{best_model_name}.png');
            plt.close()
            print(f"{best_model_name} 特征重要性图已保存。")
        elif best_model_name == 'XGBoost' and hasattr(best_model_actual, 'get_booster'):
            try:
                booster = best_model_actual.get_booster()
                importance_dict = booster.get_score(importance_type='gain')
                if importance_dict:
                    feature_importance_df = pd.DataFrame(list(importance_dict.items()),
                                                         columns=['特征 (feature)', '重要性 (importance)'])
                    feature_importance_df.sort_values(by='重要性 (importance)', ascending=False, inplace=True)
                    print(feature_importance_df.head(min(20, len(feature_importance_df))))
                    plt.figure(figsize=(12, max(8, len(feature_importance_df.head(15)) * 0.5)))
                    sns.barplot(x='重要性 (importance)', y='特征 (feature)', data=feature_importance_df.head(15))
                    plt.title(f'{best_model_name} 模型特征重要性 (gain, Top 15)');
                    plt.tight_layout()
                    plt.savefig(f'feature_importances_gain_{best_model_name}.png');
                    plt.close()
                    print(f"{best_model_name} 特征重要性图 (gain) 已保存。")
            except Exception as e_xgb_plot:
                print(f"生成XGBoost特征重要性图时出错: {e_xgb_plot}")

        if SHAP_INSTALLED:
            print(f"\n--- 开始计算 {best_model_name} 模型的 SHAP 值 ---")
            try:
                explainer = shap.TreeExplainer(best_model_actual, X_train_processed_df)
                shap_values_test = explainer.shap_values(X_test_processed_df)
                shap_values_for_summary = shap_values_test
                if isinstance(shap_values_test, list) and len(shap_values_test) == 2: shap_values_for_summary = \
                shap_values_test[1]
                plt.figure();
                shap.summary_plot(shap_values_for_summary, X_test_processed_df, plot_type="bar", show=False)
                plt.title(f'{best_model_name} SHAP 特征重要性 (Bar)');
                plt.tight_layout();
                plt.savefig(f'shap_summary_bar_{best_model_name}.png');
                plt.close()
                print(f"{best_model_name} SHAP 条形总结图已保存。")
                plt.figure();
                shap.summary_plot(shap_values_for_summary, X_test_processed_df, show=False)
                plt.title(f'{best_model_name} SHAP 特征重要性 (Dot)');
                plt.tight_layout();
                plt.savefig(f'shap_summary_dot_{best_model_name}.png');
                plt.close()
                print(f"{best_model_name} SHAP 点状总结图已保存。")
                if feature_importance_df is not None and not feature_importance_df.empty:
                    vals = np.abs(shap_values_for_summary).mean(0);
                    shap_importance = pd.DataFrame(list(zip(X_test_processed_df.columns, vals)),
                                                   columns=['col_name', 'feature_importance_vals'])
                    shap_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True);
                    top_features_for_dependence = shap_importance['col_name'].head(3).tolist()
                    for feature_name in top_features_for_dependence:
                        if feature_name in X_test_processed_df.columns:
                            plt.figure();
                            shap.dependence_plot(feature_name, shap_values_for_summary, X_test_processed_df, show=False,
                                                 interaction_index="auto")
                            plt.title(f'{best_model_name} SHAP 依赖图: {feature_name}');
                            plt.tight_layout();
                            plt.savefig(f'shap_dependence_plot_{best_model_name}_{feature_name}.png');
                            plt.close()
                            print(f"{best_model_name} SHAP 依赖图 ({feature_name}) 已保存。")
                else:
                    print("feature_importance_df 为空，SHAP 依赖图可能基于通用排序或不生成。")
            except Exception as e_shap:
                print(f"计算或绘制SHAP图时出错: {e_shap}")
    else:
        print("\n未能确定最佳模型或最佳模型训练失败。")
    return trained_models_pipelines


# --- 6. 商品推荐系统 ---
def product_recommendation_item_based_cf(df_users_orig, user_id_to_recommend, top_n=5, min_item_purchases=5):
    if df_users_orig is None or 'purchase_history' not in df_users_orig.columns: print(
        "跳过推荐: 数据为空或缺列。"); return
    df_users = df_users_orig.copy()
    print(f"\n\n--- 商品推荐 (min_item_purchases={min_item_purchases}) ---")
    df_users['purchase_history_parsed_str'] = df_users['purchase_history'].apply(
        lambda x: ' '.join(parse_simple_sequence(x)))
    if user_id_to_recommend not in df_users['user_id'].values: print(
        f"用户 {user_id_to_recommend} 不在数据集中。"); return
    vectorizer = CountVectorizer(binary=True, min_df=min_item_purchases)
    try:
        user_item_matrix_sparse = vectorizer.fit_transform(df_users['purchase_history_parsed_str'])
    except ValueError as ve:
        if "empty vocabulary" in str(ve).lower():
            print(f"推荐矩阵词汇表为空(min_df={min_item_purchases})。"); return
        else:
            print(f"推荐矩阵创建出错: {ve}"); return
    num_users_matrix, num_items_matrix = user_item_matrix_sparse.shape
    print(f"推荐用 用户-物品矩阵形状: ({num_users_matrix} 用户, {num_items_matrix} 商品)")
    if num_items_matrix == 0: print(f"无商品满足推荐 min_df={min_item_purchases} 条件。"); return
    product_ids_vocab = vectorizer.get_feature_names_out()
    user_id_list = df_users['user_id'].tolist()
    try:
        target_user_idx = user_id_list.index(user_id_to_recommend)
    except ValueError:
        print(f"用户 {user_id_to_recommend} 未在ID列表找到(推荐)。"); return
    if target_user_idx >= num_users_matrix: print(f"用户索引 {target_user_idx} 超出矩阵行数(推荐)。"); return
    target_user_purchases_vector_sparse = user_item_matrix_sparse[target_user_idx, :]
    purchased_item_indices = target_user_purchases_vector_sparse.indices
    if purchased_item_indices.size == 0: print(f"用户 {user_id_to_recommend} 在过滤后商品列表无购买记录(推荐)。"); return
    user_purchased_products_names = [product_ids_vocab[i] for i in purchased_item_indices]
    print(f"\n用户 {user_id_to_recommend} 已购商品(推荐计算依据): {user_purchased_products_names}")
    all_items_feature_vectors_sparse = user_item_matrix_sparse.T
    user_purchased_items_feature_vectors_sparse = all_items_feature_vectors_sparse[purchased_item_indices, :]
    similarity_scores_user_vs_all = cosine_similarity(user_purchased_items_feature_vectors_sparse,
                                                      all_items_feature_vectors_sparse)
    aggregated_scores_for_all_items = similarity_scores_user_vs_all.sum(axis=0)
    all_item_scores_series = pd.Series(aggregated_scores_for_all_items, index=product_ids_vocab)
    sorted_recommendations = all_item_scores_series.sort_values(ascending=False)
    final_recommendations = []
    for item_name, score in sorted_recommendations.items():
        if item_name not in user_purchased_products_names: final_recommendations.append((item_name, score))
        if len(final_recommendations) >= top_n: break
    print(f"\n--- 为用户 {user_id_to_recommend} 推荐的前 {top_n} 个商品 ---")
    if not final_recommendations:
        print("未能找到任何新商品推荐。")
    else:
        for i, (product, score) in enumerate(final_recommendations): print(
            f"{i + 1}. 商品ID: {product} (得分: {score:.4f})")


# --- 7. 主程序执行模块 ---
if __name__ == '__main__':
    print("--- 开始执行电商用户分析、预测及推荐实验 (集成Pipeline和GridSearchCV调优) ---")
    main_df_loaded = load_and_preprocess_data(DATA_FILE_PATH)
    if main_df_loaded is not None and not main_df_loaded.empty:
        X_raw_features_df, y_target, preprocessor_definition_obj, df_with_engineered_features = feature_engineering_for_prediction(
            main_df_loaded)
        features_for_clustering = None
        if X_raw_features_df is not None and preprocessor_definition_obj is not None:
            print("\n--- 为聚类和探索性分析准备预处理数据 ---")
            temp_preprocessor_for_clustering = clone(preprocessor_definition_obj)
            try:
                features_for_clustering_array = temp_preprocessor_for_clustering.fit_transform(X_raw_features_df)
                features_for_clustering = pd.DataFrame(features_for_clustering_array,
                                                       columns=temp_preprocessor_for_clustering.get_feature_names_out(),
                                                       index=X_raw_features_df.index)
                print(f"用于聚类的特征数据形状 (预处理后): {features_for_clustering.shape}")
            except Exception as e_cluster_preprocess:
                print(f"为聚类数据进行预处理时出错: {e_cluster_preprocess}")
        analyze_user_behavior(df_with_engineered_features, features_for_clustering)
        trained_prediction_pipelines = None
        if X_raw_features_df is not None and y_target is not None and not X_raw_features_df.empty:
            if y_target.nunique() >= 2:
                print("\n--- 目标变量 y_target 分布 (模型训练前) ---")
                print(y_target.value_counts(normalize=True))
                trained_prediction_pipelines = train_and_evaluate_prediction_model(X_raw_features_df, y_target,
                                                                                   preprocessor_definition_obj)
                if trained_prediction_pipelines: print("\n购买预测模型训练和调优完成。")
            else:
                print("目标变量 y_target 类别不足2个，无法训练。")
        else:
            print("\n特征工程问题或数据不足，跳过模型训练。")
        if 'user_id' in main_df_loaded.columns and not main_df_loaded['user_id'].empty:
            sample_user_id_for_recommendation = main_df_loaded['user_id'].iloc[0]
            print(f"\n尝试为示例用户 ({sample_user_id_for_recommendation}) 进行商品推荐...")
            min_purchases_threshold = 5
            product_recommendation_item_based_cf(main_df_loaded, user_id_to_recommend=sample_user_id_for_recommendation,
                                                 top_n=5, min_item_purchases=min_purchases_threshold)
        print("\n\n--- 实验执行完毕 ---")
    else:
        print("--- 实验失败: 无法加载或预处理数据。---")