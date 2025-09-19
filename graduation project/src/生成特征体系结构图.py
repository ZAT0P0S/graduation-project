from graphviz import Digraph

# --- 1. 初始化图表，并指定使用 SimHei 字体 ---
dot = Digraph('FeatureArchitecture', comment='Feature Engineering Architecture')
dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.0')
dot.attr('node', shape='box', style='rounded,filled', fontname='SimHei', fontsize='12')
dot.attr('edge', style='solid', fontname='SimHei', fontsize='10')

# --- 2. 定义图表的各个部分 ---

# **第一层：原始数据源**
with dot.subgraph(name='cluster_raw_data') as c:
    c.attr(label='第一层：原始数据源', style='filled', color='#e6f7ff', fontsize='14',
           fontname='SimHei')
    c.node('raw_data',
           '用户基本信息\n(age, gender)\n\n用户行为序列数据\n(behavior_sequence, search_sequence, purchase_history)',
           shape='folder', fillcolor='#cceeff')

# **第二层：核心处理模块**
with dot.subgraph(name='cluster_processing') as c:
    c.attr(label='第二层：核心处理模块', style='filled', color='#fffbe6', fontsize='14',
           fontname='SimHei')
    c.node('processing_node',
           '特征工程 (Feature Engineering)\n\n- 行为序列解析\n- 特征计算与构建\n(计数, 独立计数, 状态判断)',
           shape='Mrecord', fillcolor='#fff1b8')

# **第三层：结构化的特征体系**
with dot.subgraph(name='cluster_features') as c:
    c.attr(label='第三层：结构化的特征体系', style='filled', color='#f6ffed', fontsize='14',
           fontname='SimHei')
    c.node('align_node', shape='point', width='0')

    c.node('static_features', '1. 用户静态特征\n(age, gender)', fillcolor='#d9f7be')
    c.node('dynamic_features', '2. 动态行为统计特征\n(count_*, unique_items_*, has_*)', fillcolor='#d9f7be')
    c.node('sequence_features', '3. 用户行为序列特征\n(total_actions_*, unique_items_*)', fillcolor='#d9f7be')
    c.node('search_features', '4. 搜索行为特征\n(search_seq_len, unique_search_terms_count)', fillcolor='#d9f7be')
    c.node('history_features', '5. 历史购买特征\n(purchase_history_len)', fillcolor='#d9f7be')

# --- 3. 连接各个部分，表示数据流向 ---
dot.edge('raw_data', 'processing_node', label='输入')
dot.edge('processing_node', 'align_node', label='输出', arrowhead='none')

dot.edge('align_node', 'static_features', style='dashed')
dot.edge('align_node', 'dynamic_features', style='dashed')
dot.edge('align_node', 'sequence_features', style='dashed')
dot.edge('align_node', 'search_features', style='dashed')
dot.edge('align_node', 'history_features', style='dashed')

# --- 4. 生成并保存图表 ---
output_filename = 'feature_architecture_diagram_SimHei'
try:
    dot.render(output_filename, format='png', view=False, cleanup=True)
    print(f"成功！图表已保存为 '{output_filename}.png'")
except Exception as e:
    print(f"生成图表失败，请确保Graphviz已正确安装并已添加到系统PATH中，且'SimHei'字体可用。")
    print(f"错误信息: {e}")