import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import squarify
import numpy as np

# 读取数据
data = pd.read_excel('20250608.xlsx')


# 编码离散特征
data['x2'] = data['x2'].replace({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6})
data['x11'] = data['x11'].replace({'P': 1, 'Q': 2, 'R': 3, 'S': 4})

# 对连续特征进行分段处理并编码
# 对x3进行分段处理
data['x3'] = pd.cut(data['x3'], bins=[0, 3, 6, 9, 12, float('inf')], labels=[1, 2, 3, 4, 5])
# 对x8进行分段处理
data['x8'] = pd.cut(data['x8'], bins=[0, 2, 4, 6, 8, float('inf')], labels=[1, 2, 3, 4, 5])
# 对x12进行分段处理
data['x12'] = pd.cut(data['x12'], bins=[0, 90, 180, 270, 365, float('inf')], labels=[1, 2, 3, 4, 5])
# 对x13进行分段处理
data['x13'] = pd.cut(data['x13'], bins=[0, 100000, 200000, 300000, 500000, float('inf')], labels=[1, 2, 3, 4, 5])
# 对x14进行分段处理
data['x14'] = data['x14'].apply(lambda x: 0 if x == 0 else pd.cut([x], bins=[0, 100000, 200000, 300000, 500000, float('inf')], labels=[1, 2, 3, 4, 5])[0])
# 对x15进行分段处理
data['x15'] = pd.cut(data['x15'], bins=[0, 30, 60, 90, 120, float('inf')], labels=[1, 2, 3, 4, 5])
# 对x16进行分段处理
data['x16'] = pd.cut(data['x16'], bins=[0, 1, 2, 3, 4, float('inf')], labels=[1, 2, 3, 4, 5])

# 将经过处理的数据保存到新的Excel文件
data.to_excel('processed_data.xlsx', index=False)

# 将所有特征都转换为字符串类型，以便构建事务数据
data = data.astype(str)

# FP-Growth需要事务列表作为输入，每个事务是一个项目集
# 这里我们将每行视为一个事务，每个特征视为一个项
transactions = []
for index, row in data.iterrows():
    transaction = row[['x1', 'x2', 'x3',  'x8',  'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x18']].tolist()
    transactions.append(transaction)

# 将事务数据保存到Excel
transactions_df = pd.DataFrame(transactions, columns=['x1', 'x2', 'x3', 'x8', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x18'])
transactions_df.to_excel('transactions.xlsx', index=False)

# 初始化TransactionEncoder
te = TransactionEncoder()

# 使用TransactionEncoder的fit方法学习事务中的项，并使用transform方法将事务转换为独热编码
te_ary = te.fit(transactions).transform(transactions)

# 将独热编码的数组转换为DataFrame，这样mlxtend的fpgrowth函数可以处理它
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# 现在df_encoded是一个包含独热编码事务的DataFrame，我们可以使用它来找频繁项集
frequent_itemsets = fpgrowth(df_encoded, min_support=0.1, use_colnames=True)

# 提取关联规则
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# 将频繁项集结果保存到Excel
frequent_itemsets.to_excel('frequent_itemsets.xlsx', index=False)

# 将关联规则结果保存到Excel
rules.to_excel('association_rules.xlsx', index=False)

# 提取关联规则的前项、后项和支持度
rules_data = []
for index, rule in rules.iterrows():
    antecedents = rule['antecedents']
    consequents = rule['consequents']
    support = frequent_itemsets.loc[frequent_itemsets['itemsets'] == antecedents,'support'].values[0]
    rules_data.append((antecedents, consequents, support))

# 准备热图数据
data_matrix = [[0] * len(frequent_itemsets) for _ in range(len(frequent_itemsets))]
for i, itemset1 in enumerate(frequent_itemsets['itemsets']):
    for j, itemset2 in enumerate(frequent_itemsets['itemsets']):
        common_items = itemset1.intersection(itemset2)
        data_matrix[i][j] = len(common_items)

data_matrix_df = pd.DataFrame(data_matrix, columns=frequent_itemsets['itemsets'], index=frequent_itemsets['itemsets'])

# ================== 修复后的热图部分 ==================
# 设置全局字体和图像分辨率
plt.rcParams['font.size'] = 8
plt.rcParams['font.family'] = 'DejaVu Sans'  # 使用更兼容的字体
plt.rcParams['figure.dpi'] = 300  # 设置高DPI

# 创建热图
plt.figure(figsize=(16, 14))  # 增大图像尺寸
ax = sns.heatmap(
    data_matrix_df,
    annot=True,
    cmap='YlGnBu',
    fmt='d',  # 显示整数
    annot_kws={'fontsize': 6}  # 减小注释字体
)

# 恢复原始坐标标签格式
ax.set_xticklabels(
    [str(itemset) for itemset in frequent_itemsets['itemsets']],
    rotation=45,
    ha='right',
    fontsize=6
)

ax.set_yticklabels(
    [str(itemset) for itemset in frequent_itemsets['itemsets']],
    fontsize=6
)

plt.title('Heatmap of Itemset Overlap', fontsize=10)
plt.tight_layout()  # 自动调整布局

# 保存高质量图像
plt.savefig('itemset_overlap_heatmap.png', dpi=600, bbox_inches='tight')
plt.close()  # 关闭当前图形，避免重叠

# ================== 修复后的树状图部分 ==================
# 确保values变量被正确定义
values = [support for antecedents, consequents, support in rules_data]
labels = []

for antecedents, consequents, support in rules_data:
    # 简化标签显示
    label = f"{','.join(map(str, antecedents))[:15]}... -> {','.join(map(str, consequents))[:10]}..."
    labels.append(label)

# 绘制树状图
plt.figure(figsize=(24, 18), dpi=300)
squarify.plot(
    sizes=values,
    label=labels,
    alpha=.8,
    text_kwargs={
        'fontsize': 7,  # 减小字体
        'wrap': True     # 允许文本换行
    }
)

plt.axis('off')
plt.title('Treemap of Association Rules', fontsize=12)

# 保存高质量图像
plt.savefig('association_rules_treemap.png', dpi=600, bbox_inches='tight')
plt.close()

print("修复后的图像已保存为高分辨率PNG文件")