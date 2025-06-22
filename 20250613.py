import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体以避免警告
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# ===== 1. 数据加载与预处理 =====
data = pd.read_excel('20250608.xlsx')

# 数据预处理
data['x2'] = data['x2'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6})
data['x11'] = data['x11'].map({'P': 1, 'Q': 2, 'R': 3, 'S': 4})

# 连续特征分段
data['x12'] = pd.cut(data['x12'], bins=[0, 30, 60, 90, 120, float('inf')], labels=[1, 2, 3, 4, 5])
data['x13'] = pd.cut(data['x13'], bins=[0, 10000, 30000, 50000, 100000, float('inf')], labels=[1, 2, 3, 4, 5])


# ===== 2. 构建事务数据并挖掘规则 =====
def build_transactions(df):
    transactions = []
    for _, row in df.iterrows():
        transaction = row[['x1', 'x2', 'x3', 'x8', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16']].tolist()
        transactions.append(transaction)
    return transactions


# 构建全量数据事务
transactions_full = build_transactions(data)

# 事务数据编码
te_full = TransactionEncoder()
te_ary_full = te_full.fit_transform(transactions_full)
df_encoded_full = pd.DataFrame(te_ary_full, columns=te_full.columns_)

# 挖掘频繁项集和关联规则
frequent_itemsets_full = fpgrowth(df_encoded_full, min_support=0.1, use_colnames=True)
rules_full = association_rules(frequent_itemsets_full, metric="confidence", min_threshold=0.7)
print(f"全量数据挖掘结果：{len(frequent_itemsets_full)}项频繁项集，{len(rules_full)}条关联规则")

# ===== 3. 数据集划分与泛化能力验证 =====
# 数据集划分
X = data.drop('x18', axis=1)
y = data['x18']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print(f"训练集样本数：{len(X_train)}, 测试集样本数：{len(X_test)}")

# 构建测试集事务数据
transactions_test = build_transactions(X_test)
te_ary_test = te_full.transform(transactions_test)
df_encoded_test = pd.DataFrame(te_ary_test, columns=te_full.columns_)


# 验证规则在测试集上的表现
def evaluate_rules_on_test(rules, df_test, y_test):
    results = []
    for _, rule in rules.iterrows():
        antecedents = list(rule['antecedents'])
        consequents = list(rule['consequents'])

        # 筛选满足前项条件的测试集样本
        mask = np.ones(len(df_test), dtype=bool)
        for item in antecedents:
            if item in df_test.columns:
                mask &= df_test[item].values == 1

        # 计算测试集上满足前项条件的样本数
        antecedent_count = np.sum(mask)

        # 计算测试集置信度
        if antecedent_count > 0:
            actual_consequents = y_test[mask].values
            predicted_consequent = consequents[0]
            test_confidence = np.mean(actual_consequents == predicted_consequent)
            train_confidence = rule['confidence']
            base_rate = np.mean(y_test.values == predicted_consequent)
            lift = test_confidence / base_rate if base_rate > 0 else 0
            offset_rate = test_confidence / train_confidence if train_confidence > 0 else 0

            results.append({
                '规则前项': antecedents,
                '规则后项': consequents,
                '原始置信度': train_confidence,
                '测试集置信度': test_confidence,
                '置信度偏移率': offset_rate,
                '提升度(Lift)': lift,
                '前项匹配样本数': antecedent_count
            })
    return pd.DataFrame(results)


# ===== 4. 特征映射表（基于业务逻辑调整）=====
# 根据业务逻辑调整特征映射
feature_mapping = {
    0.0: "失联模式=0",  # 假设0.0代表某种失联模式
    1.0: "x1=1",
    2.0: "x2=2",  # 可能对应x2=B
    3.0: "x3=3",
    4.0: "x4=4",
    5.0: "x5=5",
    6.0: "x6=6",
    7.0: "x7=7",
    8.0: "x8=8",  # 可能对应x8=高频停机
    9.0: "x9=9",
    10.0: "x10=10",
    11.0: "x11=11",  # 可能对应x11=Q
    12.0: "x12=12",  # 可能对应x12=180-270天
    13.0: "x13=13",  # 可能对应x13=30-50万元
    14.0: "x14=14",
    15.0: "x15=15",  # 可能对应x15=30-60天
    16.0: "x16=16",  # 可能对应x16=0-1人
}


# 将规则中的特征编码转换为实际特征名
def decode_rule(rule, mapping):
    antecedents = [mapping.get(f, f"Unknown_{f}") for f in rule['规则前项']]
    consequents = [mapping.get(f, f"Unknown_{f}") for f in rule['规则后项']]
    return {
        '规则前项(解码)': antecedents,
        '规则后项(解码)': consequents,
        '原始置信度': rule['原始置信度'],
        '测试集置信度': rule['测试集置信度'],
        '置信度偏移率': rule['置信度偏移率'],
        '提升度(Lift)': rule['提升度(Lift)'],
        '前项匹配样本数': rule['前项匹配样本数']
    }


# ===== 5. 核心规则分析 =====
print("\n===== 规则示例（前10条）=====")
print(rules_full.head(10)[['antecedents', 'consequents', 'confidence']])

# 基于业务逻辑调整核心失联模式规则
core_patterns = [
    frozenset({2.0, 4.0, 5.0}),  # 基于规则示例中的常见组合
    frozenset({0.0, 2.0, 4.0}),  # 基于规则示例中的常见组合
    frozenset({8.0, 2.0, 3.0, 5.0})  # 基于规则示例中的常见组合
]

# 查找核心规则
core_rules = pd.DataFrame()
for pattern in core_patterns:
    matching_rules = rules_full[rules_full['antecedents'].apply(lambda x: pattern.issubset(x))]
    if len(matching_rules) > 0:
        best_rule = matching_rules.loc[matching_rules['confidence'].idxmax()]
        core_rules = pd.concat([core_rules, pd.DataFrame([best_rule])], ignore_index=True)

print(f"\n找到{len(core_rules)}条核心规则")

# 执行测试集验证
if len(core_rules) > 0:
    core_evaluation = evaluate_rules_on_test(core_rules, df_encoded_test, y_test)

    if len(core_evaluation) > 0:
        print("\n===== 核心失联模式规则测试集验证结果（原始编码）=====")
        print(core_evaluation[['规则前项', '规则后项', '原始置信度', '测试集置信度', '置信度偏移率', '提升度(Lift)',
                               '前项匹配样本数']])

        # 解码规则并输出
        decoded_core_evaluation = core_evaluation.apply(lambda x: decode_rule(x, feature_mapping), axis=1,
                                                        result_type='expand')
        print("\n===== 核心失联模式规则测试集验证结果（解码后）=====")
        print(decoded_core_evaluation)
    else:
        print("\n警告：核心规则在测试集中没有匹配到任何样本！")
else:
    print("\n警告：未找到与核心模式匹配的规则！使用前10条高置信度规则作为替代")
    # 使用前10条高置信度规则作为替代
    core_rules = rules_full.sort_values('confidence', ascending=False).head(10)
    core_evaluation = evaluate_rules_on_test(core_rules, df_encoded_test, y_test)

    if len(core_evaluation) > 0:
        print("\n===== 高置信度规则测试集验证结果（原始编码）=====")
        print(core_evaluation[['规则前项', '规则后项', '原始置信度', '测试集置信度', '置信度偏移率', '提升度(Lift)',
                               '前项匹配样本数']])

        # 解码规则并输出
        decoded_core_evaluation = core_evaluation.apply(lambda x: decode_rule(x, feature_mapping), axis=1,
                                                        result_type='expand')
        print("\n===== 高置信度规则测试集验证结果（解码后）=====")
        print(decoded_core_evaluation)
    else:
        print("\n警告：高置信度规则在测试集中也没有匹配到任何样本！")

# ===== 6. 整体规则泛化能力评估 =====
all_rules_evaluation = evaluate_rules_on_test(rules_full, df_encoded_test, y_test)

if len(all_rules_evaluation) > 0:
    print(f"\n===== 所有规则泛化能力评估 =====")
    print(
        f"测试集上有匹配的规则数：{len(all_rules_evaluation)}/{len(rules_full)} ({len(all_rules_evaluation) / len(rules_full):.2%})")
    print(f"平均测试集置信度：{all_rules_evaluation['测试集置信度'].mean():.4f}")
    print(f"平均置信度偏移率：{all_rules_evaluation['置信度偏移率'].mean():.4f}")
    print(f"置信度偏移率≥0.9的规则占比：{(all_rules_evaluation['置信度偏移率'] >= 0.9).mean():.2%}")

    # 输出表现最好的5条规则（解码后）
    top_rules = all_rules_evaluation.sort_values('测试集置信度', ascending=False).head(5)
    decoded_top_rules = top_rules.apply(lambda x: decode_rule(x, feature_mapping), axis=1, result_type='expand')
    print("\n===== 表现最好的5条规则（解码后）=====")
    print(decoded_top_rules)

    # 输出泛化能力最强的5条规则
    robust_rules = all_rules_evaluation.sort_values('置信度偏移率', ascending=False).head(5)
    decoded_robust_rules = robust_rules.apply(lambda x: decode_rule(x, feature_mapping), axis=1, result_type='expand')
    print("\n===== 泛化能力最强的5条规则（解码后）=====")
    print(decoded_robust_rules)
else:
    print("\n警告：所有规则在测试集中均无匹配样本！")

# ===== 7. 规则优化建议 =====
print("\n===== 规则优化建议 =====")
# 计算规则在测试集上的支持度
rule_supports = []
for _, rule in rules_full.iterrows():
    antecedents = list(rule['antecedents'])
    mask = np.ones(len(df_encoded_test), dtype=bool)
    for item in antecedents:
        if item in df_encoded_test.columns:
            mask &= df_encoded_test[item].values == 1
    support = np.sum(mask) / len(df_encoded_test)
    rule_supports.append(support)

# 分析不同支持度阈值下的规则质量
support_thresholds = [0.05, 0.1, 0.15, 0.2]
for threshold in support_thresholds:
    filtered_rules = rules_full[rules_full['support'] >= threshold]
    filtered_evaluation = evaluate_rules_on_test(filtered_rules, df_encoded_test, y_test)
    if len(filtered_evaluation) > 0:
        avg_confidence = filtered_evaluation['测试集置信度'].mean()
        avg_offset = filtered_evaluation['置信度偏移率'].mean()
        print(
            f"支持度阈值 {threshold}: {len(filtered_rules)}条规则, 平均测试置信度: {avg_confidence:.4f}, 平均偏移率: {avg_offset:.4f}")
    else:
        print(f"支持度阈值 {threshold}: 无有效规则")

# ===== 8. 最终规则筛选建议 =====
print("\n===== 最终规则筛选建议 =====")
if len(all_rules_evaluation) > 0:
    # 筛选高质量规则
    high_quality_rules = all_rules_evaluation[
        (all_rules_evaluation['测试集置信度'] >= 0.5) &
        (all_rules_evaluation['置信度偏移率'] >= 0.8)
        ]

    if len(high_quality_rules) > 0:
        print(
            f"符合高质量标准的规则数：{len(high_quality_rules)}/{len(rules_full)} ({len(high_quality_rules) / len(rules_full):.2%})")
        decoded_high_quality_rules = high_quality_rules.apply(lambda x: decode_rule(x, feature_mapping), axis=1,
                                                              result_type='expand')
        print("\n===== 高质量规则列表（解码后）=====")
        print(decoded_high_quality_rules)
    else:
        print("未找到符合高质量标准的规则，请降低筛选条件")

        # 降低标准再次筛选
        medium_quality_rules = all_rules_evaluation[
            (all_rules_evaluation['测试集置信度'] >= 0.3) &
            (all_rules_evaluation['置信度偏移率'] >= 0.5)
            ]

        if len(medium_quality_rules) > 0:
            print(
                f"符合中等质量标准的规则数：{len(medium_quality_rules)}/{len(rules_full)} ({len(medium_quality_rules) / len(rules_full):.2%})")
            decoded_medium_quality_rules = medium_quality_rules.apply(lambda x: decode_rule(x, feature_mapping), axis=1,
                                                                      result_type='expand')
            print("\n===== 中等质量规则列表（解码后）=====")
            print(decoded_medium_quality_rules)

            # 特别关注包含关键特征的规则
            key_features = [2.0, 8.0]  # x2=2 和 x8=8
            key_rules = medium_quality_rules[
                medium_quality_rules['规则前项'].apply(lambda x: any(f in x for f in key_features))
            ]

            if len(key_rules) > 0:
                print("\n===== 包含关键特征的中等质量规则（解码后）=====")
                decoded_key_rules = key_rules.apply(lambda x: decode_rule(x, feature_mapping), axis=1,
                                                    result_type='expand')
                print(decoded_key_rules)
            else:
                print("\n未找到包含关键特征的中等质量规则")
        else:
            print("未找到符合中等质量标准的规则，建议重新调整挖掘参数")
else:
    print("无法提供规则筛选建议，所有规则在测试集中均无匹配样本")

# ===== 9. 业务规则可视化 =====
if len(all_rules_evaluation) > 0:
    # 创建规则质量评估图
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='测试集置信度',
        y='置信度偏移率',
        size='前项匹配样本数',
        hue='提升度(Lift)',
        data=all_rules_evaluation,
        sizes=(50, 200),
        palette='viridis'
    )

    # 添加参考线
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=0.3, color='r', linestyle='--', alpha=0.3)

    # 添加标题和标签
    plt.title('规则质量评估散点图')
    plt.xlabel('测试集置信度')
    plt.ylabel('置信度偏移率')
    plt.grid(True, alpha=0.3)

    # 保存图像
    plt.savefig('rule_quality_evaluation.png', dpi=300, bbox_inches='tight')
    print("\n已生成规则质量评估图: rule_quality_evaluation.png")

    # 提取关键规则的特征
    if len(decoded_medium_quality_rules) > 0:
        # 统计特征在中等质量规则中的出现频率
        feature_counts = {}
        for _, rule in decoded_medium_quality_rules.iterrows():
            for feature in rule['规则前项(解码)'] + rule['规则后项(解码)']:
                if feature in feature_counts:
                    feature_counts[feature] += 1
                else:
                    feature_counts[feature] = 1

        # 排序并显示前10个特征
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        print("\n===== 中等质量规则中出现频率最高的特征 =====")
        for feature, count in sorted_features[:10]:
            print(f"{feature}: {count}次")

# ===== 10. 保存关键规则结果 =====
if 'decoded_medium_quality_rules' in locals():
    # 保存中等质量规则到Excel文件
    decoded_medium_quality_rules.to_excel('medium_quality_rules.xlsx', index=False)
    print("\n已保存中等质量规则到: medium_quality_rules.xlsx")

    # 保存包含关键特征的规则到Excel文件
    if 'decoded_key_rules' in locals():
        decoded_key_rules.to_excel('key_rules.xlsx', index=False)
        print("\n已保存包含关键特征的规则到: key_rules.xlsx")