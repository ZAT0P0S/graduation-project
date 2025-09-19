一、项目简介 (Project Overview)
本项目是我的本科毕业设计，旨在利用机器学习技术，对电商用户的行为数据进行深入分析、构建用户画像，并精准预测用户未来的购买行为。项目完整地实现了从数据处理、特征工程、模型训练与评估、到模型解释的全链路数据科学流程。

二、项目结构 (Project Structure)
e-commerce-purchase-prediction/
├── data/              # 存放数据样本
├── src/               # 存放核心Python代码
├── results/           # 存放图表等结果
├── README.md          # 本说明文件
└── requirements.txt   # 项目依赖库

三、技术栈 (Tech Stack)
语言: Python 3.x

核心库: Pandas, Scikit-learn, XGBoost, SHAP, Matplotlib

四、安装与运行 (Installation & Usage)
1.克隆本项目
git clone https://github.com/ZAT0P0S/graduation-project.git
2.安装依赖
pip install -r requirements.txt
3.运行项目
python src/model_training.py

五、核心成果展示 (Key Results)
经过系统性评估，最终的XGBoost模型在独立测试集上表现最优，取得了 90.22%的准确率 和 0.8855的PR AUC。

通过SHAP值分析，我们发现“用户是否将商品加入购物车”是预测购买行为的最强驱动因素。
