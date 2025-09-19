import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# --- 1. 设置 Matplotlib 支持中文 (可选，如果你的图表标题或标签需要中文) ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是一种常用的支持中文的字体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    print("已尝试设置Matplotlib中文字体为SimHei。")
except Exception as e_font:
    print(f"设置Matplotlib中文字体失败: {e_font}。如果图表中文显示乱码，请安装并指定可用中文字体。")

# --- 2. 加载模型预测结果和真实标签 ---
print("正在加载数据...")
try:
    y_test = np.load('y_test.npy', allow_pickle=True) # allow_pickle=True 以防万一y_test中有非数值类型，虽然通常是数值
    y_pred_proba_log_reg = np.load('y_pred_proba_log_reg.npy', allow_pickle=True)
    y_pred_proba_rf = np.load('y_pred_proba_rf.npy', allow_pickle=True)
    y_pred_proba_xgb = np.load('y_pred_proba_xgb.npy', allow_pickle=True)
    print("成功加载模型预测结果和真实标签。")
    print(f"  y_test 形状: {y_test.shape}, 类型: {y_test.dtype}")
    print(f"  y_pred_proba_log_reg 形状: {y_pred_proba_log_reg.shape}, 类型: {y_pred_proba_log_reg.dtype}")
    print(f"  y_pred_proba_rf 形状: {y_pred_proba_rf.shape}, 类型: {y_pred_proba_rf.dtype}")
    print(f"  y_pred_proba_xgb 形状: {y_pred_proba_xgb.shape}, 类型: {y_pred_proba_xgb.dtype}")

    # 确保 y_test 是整数类型，如果不是，尝试转换
    if not np.issubdtype(y_test.dtype, np.integer):
        print("警告: y_test 不是整数类型，将尝试转换为整数。")
        y_test = y_test.astype(int)

    # 确保预测概率是浮点数类型
    if not np.issubdtype(y_pred_proba_log_reg.dtype, np.floating):
        y_pred_proba_log_reg = y_pred_proba_log_reg.astype(float)
    if not np.issubdtype(y_pred_proba_rf.dtype, np.floating):
        y_pred_proba_rf = y_pred_proba_rf.astype(float)
    if not np.issubdtype(y_pred_proba_xgb.dtype, np.floating):
        y_pred_proba_xgb = y_pred_proba_xgb.astype(float)


except FileNotFoundError:
    print("错误：一个或多个 .npy 文件未找到。请确保以下文件与脚本在同一目录下：")
    print("y_test.npy, y_pred_proba_log_reg.npy, y_pred_proba_rf.npy, y_pred_proba_xgb.npy")
    exit()
except Exception as e:
    print(f"加载数据时发生错误: {e}")
    exit()

# 检查 y_test 是否只有一个类别，这会导致 PR/ROC 曲线计算错误
if len(np.unique(y_test)) < 2:
    print("错误: y_test (真实标签) 中只包含一个类别。无法计算PR或ROC曲线。")
    exit()

# --- 3. 计算PR曲线和ROC曲线数据 ---
print("\n正在计算PR和ROC曲线数据...")

models_data = {
    '逻辑回归 (Logistic Regression)': y_pred_proba_log_reg,
    '随机森林 (Random Forest)': y_pred_proba_rf,
    'XGBoost': y_pred_proba_xgb
}

pr_curves = {}
roc_curves = {}

for model_name, y_pred_proba in models_data.items():
    if y_pred_proba is None:
        print(f"警告: 模型 '{model_name}' 的预测概率数据为 None，将跳过此模型。")
        continue
    if len(y_pred_proba) != len(y_test):
        print(f"错误: 模型 '{model_name}' 的预测概率数量 ({len(y_pred_proba)}) 与 y_test 数量 ({len(y_test)}) 不匹配。")
        exit()

    # PR曲线
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    pr_curves[model_name] = {'recall': recall, 'precision': precision, 'auc': pr_auc}
    print(f"  {model_name} - PR AUC: {pr_auc:.4f}")

    # ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    roc_curves[model_name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    print(f"  {model_name} - ROC AUC: {roc_auc:.4f}")


# 基线 (随机猜测)
# 对于PR曲线，基线是 y = positive_class_ratio
num_positives = np.sum(y_test == 1)
num_total = len(y_test)
pr_baseline = num_positives / num_total if num_total > 0 else 0

# --- 4. 绘制PR曲线对比图 ---
print("\n正在绘制PR曲线对比图...")
plt.figure(figsize=(10, 8))
for model_name, data in pr_curves.items():
    plt.plot(data['recall'], data['precision'], label=f"{model_name} (AP = {data['auc']:.3f})")

plt.plot([0, 1], [pr_baseline, pr_baseline], linestyle='--', label=f'随机猜测 (AP ≈ {pr_baseline:.3f})', color='grey')
plt.xlabel('召回率 (Recall)')
plt.ylabel('精确率 (Precision)')
plt.title('不同模型的PR曲线对比 (Precision-Recall Curve Comparison)')
plt.legend(loc='lower left')
plt.grid(True)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
try:
    plt.savefig('combined_pr_curves.png')
    print("PR曲线对比图已保存为 combined_pr_curves.png")
except Exception as e_save:
    print(f"保存PR曲线图失败: {e_save}")
plt.show()

# --- 5. 绘制ROC曲线对比图 ---
print("\n正在绘制ROC曲线对比图...")
plt.figure(figsize=(10, 8))
for model_name, data in roc_curves.items():
    plt.plot(data['fpr'], data['tpr'], label=f"{model_name} (AUC = {data['auc']:.3f})")

plt.plot([0, 1], [0, 1], linestyle='--', label='随机猜测 (AUC = 0.500)', color='grey') # ROC基线
plt.xlabel('假正例率 (False Positive Rate)')
plt.ylabel('真正例率 (True Positive Rate)')
plt.title('不同模型的ROC曲线对比 (ROC Curve Comparison)')
plt.legend(loc='lower right')
plt.grid(True)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
try:
    plt.savefig('combined_roc_curves.png')
    print("ROC曲线对比图已保存为 combined_roc_curves.png")
except Exception as e_save:
    print(f"保存ROC曲线图失败: {e_save}")
plt.show()

print("\n脚本执行完毕。")