import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import squarify
import numpy as np
import warnings
from matplotlib import MatplotlibDeprecationWarning
from sklearn.model_selection import train_test_split
import pydotplus
from io import BytesIO, StringIO
from PIL import Image

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=MatplotlibDeprecationWarning)

# Set font to avoid warnings
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['font.size'] = 6
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

# ===== 1. Data loading and preprocessing  =====
# Load data
data = pd.read_excel('lost_linking_processed_10f.xlsx')

# Encode categorical features (aligned with Table 1)
data['x7'] = data['x7'].replace({'A': 'x7=1', 'B': 'x7=2', 'C': 'x7=3', 'D': 'x7=4', 'E': 'x7=5', 'F': 'x7=6'})
data['x11'] = data['x11'].replace({'P': 'x11=1', 'Q': 'x11=2', 'R': 'x11=3', 'S': 'x11=4'})
data['x1'] = data['x1'].apply(lambda x: f'x1={x}')
data['x18'] = data['x18'].apply(lambda x: f'x18={x}')  # Lost-linking modes (HS/FM/FD)

# Discretize continuous features
data['x3'] = pd.cut(
    data['x3'],
    bins=[0, 3, 6, 9, 12, float('inf')],
    labels=['x3=1', 'x3=2', 'x3=3', 'x3=4', 'x3=5']
)

data['x8'] = pd.cut(
    data['x8'],
    bins=[0, 2, 4, 6, 8, float('inf')],
    labels=['x8=1', 'x8=2', 'x8=3', 'x8=4', 'x8=5']
)

data['x12'] = pd.cut(  # Critical for HS mode
    data['x12'],
    bins=[0, 90, 180, 270, 365, float('inf')],
    labels=['x12=1', 'x12=2', 'x12=3', 'x12=4', 'x12=5']
)

data['x13'] = pd.cut(  # Key indicator in core rules
    data['x13'],
    bins=[0, 100000, 200000, 300000, 500000, float('inf')],
    labels=['x13=1', 'x13=2', 'x13=3', 'x13=4', 'x13=5']
)


def discretize_x14(x):
    if x == 0:
        return 'x14=0'
    else:
        return pd.cut([x], bins=[0, 100000, 200000, 300000, 500000, float('inf')],
                      labels=['x14=1', 'x14=2', 'x14=3', 'x14=4', 'x14=5'])[0]


data['x14'] = data['x14'].apply(discretize_x14)

data['x15'] = pd.cut(  # Relevant for FD mode
    data['x15'],
    bins=[0, 30, 60, 90, 120, float('inf')],
    labels=['x15=1', 'x15=2', 'x15=3', 'x15=4', 'x15=5']
)

data['x16'] = pd.cut(  # Indicator for FM mode
    data['x16'],
    bins=[0, 2, 4, 6, 10, float('inf')],
    labels=['x16=1', 'x16=2', 'x16=3', 'x16=4', 'x16=5']
)

# Construct transactions
transactions = []
for _, row in data.iterrows():
    transaction = [
        row['x1'], row['x3'], row['x7'], row['x8'],
        row['x11'], row['x12'], row['x13'],
        row['x14'], row['x15'], row['x16'], row['x18']
    ]
    transactions.append(transaction)

# Save processed data
data.to_excel('processed_data.xlsx', index=False)
transactions_df = pd.DataFrame(transactions,
                               columns=['x1', 'x3', 'x7', 'x8', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x18'])
transactions_df.to_excel('transactions.xlsx', index=False)

# Encode transactions for FP-Growth
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Mine frequent itemsets (min_support=0.1)
frequent_itemsets = fpgrowth(df_encoded, min_support=0.1, use_colnames=True)

# Filter core itemsets (containing x18)
core_itemsets = frequent_itemsets[
    frequent_itemsets['itemsets'].apply(lambda x: any('x18=' in item for item in x))
].sort_values(by='support', ascending=False)
core_itemsets = core_itemsets[core_itemsets['support'] > 0.2]  # High-support itemsets

# Mine association rules (min_confidence=0.7)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Filter core rules (x18 in consequents, lift>1.2 as per significant correlations)
core_rules = rules[
    rules['consequents'].apply(lambda x: any('x18=' in item for item in x)) &
    (rules['lift'] > 1.2)
    ].sort_values(by='confidence', ascending=False)

# Save results
frequent_itemsets.to_excel('frequent_itemsets.xlsx', index=False)
core_itemsets.to_excel('core_itemsets.xlsx', index=False)
rules.to_excel('association_rules.xlsx', index=False)
core_rules.to_excel('core_rules.xlsx', index=False)

# Generate heatmap for top core itemsets
if not core_itemsets.empty:
    top_core_itemsets = core_itemsets.head(50)  # Focus on top 50
    itemset_list = top_core_itemsets['itemsets'].tolist()

    data_matrix = np.zeros((len(itemset_list), len(itemset_list)))
    for i, itemset1 in enumerate(itemset_list):
        for j, itemset2 in enumerate(itemset_list):
            data_matrix[i][j] = len(itemset1.intersection(itemset2))

    # Simplify labels
    simplified_labels = [
        str(its).replace("frozenset({'", "").replace("'})", "").replace("', '", ", ")[:20]
        for its in itemset_list
    ]
    data_matrix_df = pd.DataFrame(
        data_matrix,
        columns=simplified_labels,
        index=simplified_labels
    )

    plt.figure(figsize=(14, 12))
    # Use a more informative colormap with clear color gradient
    cmap = sns.color_palette("viridis", as_cmap=True)

    # Create heatmap with annotations
    ax = sns.heatmap(
        data_matrix_df,
        annot=True,
        cmap='YlGnBu',
        fmt='.0f',
        annot_kws={'fontsize': 5, 'alpha': 0.7},
        cbar=True,
        linewidths=0.5,
        vmin=0,  # Set minimum value for color scale
        vmax=data_matrix.max()  # Set maximum value for color scale
    )

    # Configure color bar with clear labels
    cbar = ax.collections[0].colorbar
    cbar.set_label('Number of Common Items', fontsize=9)
    cbar.ax.tick_params(labelsize=6)

    # Set axis labels and ticks
    ax.set_xticks(range(len(itemset_list)))
    ax.set_xticklabels(simplified_labels, rotation=45, ha='right', fontsize=6)
    ax.set_yticks(range(len(itemset_list)))
    ax.set_yticklabels(simplified_labels, fontsize=6)

    # Add title and explanatory text
    plt.title('Overlap of Top Core Itemsets', fontsize=12)
    plt.figtext(0.5, 0.01,
                'Color indicates the number of common items between itemset pairs (darker = more common items)',
                ha='center', fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for figtext
    plt.savefig('top_core_itemset_heatmap.png', dpi=600, bbox_inches='tight')
    plt.close()

# Generate rule network structure using Graphviz
if not core_rules.empty:
    # Select top rules for visualization
    top_core_rules = core_rules.sort_values(by='confidence', ascending=False).head(30)  # Top 20 rules

    # Create a directed graph
    graph = pydotplus.Dot(graph_type='digraph', dpi=300)
    graph.set_rankdir('LR')  # Left to right layout

    # Create a dictionary to store nodes
    nodes = {}

    # Define color schemes
    antecedent_color = '#5DA5DA'  # Blue for antecedents
    consequent_color = '#FAA43A'  # Orange for consequents
    edge_colors = {
        'high': '#60BD68',  # Green for high confidence
        'medium': '#FAA43A',  # Orange for medium confidence
        'low': '#F17CB0'  # Pink for low confidence
    }

    # Add nodes and edges
    for _, rule in top_core_rules.iterrows():
        antecedents = [item for item in rule['antecedents'] if 'x18' not in item]
        consequents = [item for item in rule['consequents'] if 'x18' in item]

        if not antecedents or not consequents:
            continue

        # Add antecedent nodes
        for ant in antecedents:
            if ant not in nodes:
                node = pydotplus.Node(
                    ant,
                    style='filled',
                    fillcolor=antecedent_color,
                    shape='box',
                    fontsize=10
                )
                graph.add_node(node)
                nodes[ant] = True

        # Add consequent nodes
        for con in consequents:
            if con not in nodes:
                node = pydotplus.Node(
                    con,
                    style='filled',
                    fillcolor=consequent_color,
                    shape='ellipse',
                    fontsize=10
                )
                graph.add_node(node)
                nodes[con] = True

        # Determine edge color based on confidence
        confidence = rule['confidence']
        if confidence >= 0.8:
            edge_color = edge_colors['high']
        elif confidence >= 0.6:
            edge_color = edge_colors['medium']
        else:
            edge_color = edge_colors['low']

        # Add edges with confidence and lift information
        for ant in antecedents:
            for con in consequents:
                edge = pydotplus.Edge(
                    ant,
                    con,
                    label=f"Conf: {confidence:.2f}\nLift: {rule['lift']:.2f}",
                    color=edge_color,
                    fontsize=8,
                    penwidth=0.5
                )
                graph.add_edge(edge)

    # Add a legend
    legend = pydotplus.Cluster('legend', label='Legend', fontsize=12)

    # Legend nodes
    legend.add_node(
        pydotplus.Node('ant_legend', label='Antecedents', style='filled', fillcolor=antecedent_color, shape='box',
                       fontsize=10))
    legend.add_node(
        pydotplus.Node('con_legend', label='Consequents', style='filled', fillcolor=consequent_color, shape='ellipse',
                       fontsize=10))
    legend.add_node(
        pydotplus.Node('high_legend', label='High Confidence (≥0.8)', color=edge_colors['high'], fontsize=10))
    legend.add_node(
        pydotplus.Node('med_legend', label='Medium Confidence (0.6-0.8)', color=edge_colors['medium'], fontsize=10))
    legend.add_node(pydotplus.Node('low_legend', label='Low Confidence (<0.6)', color=edge_colors['low'], fontsize=10))

    graph.add_subgraph(legend)

    # Save the graph
    graph.write_png('top_core_rules_network.png')
    print("Rule network graph generated: top_core_rules_network.png")

# ===== 2. Dataset splitting and generalization capability verification  =====
# Split dataset
X = data.drop('x18', axis=1)
y = data['x18']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print(f"Training set samples: {len(X_train)}, Test set samples: {len(X_test)}")


# Build test set transaction data
def build_transactions(df):
    transactions = []
    for _, row in df.iterrows():
        transaction = [str(val) for val in
                       row[['x1', 'x3', 'x7', 'x8', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16']].tolist()]
        transactions.append(transaction)
    return transactions


transactions_test = build_transactions(X_test)
te_ary_test = te.transform(transactions_test)
df_encoded_test = pd.DataFrame(te_ary_test, columns=te.columns_)


# Evaluate rules on the test set
def evaluate_rules_on_test(rules, df_test, y_test):
    results = []
    for _, rule in rules.iterrows():
        antecedents = list(rule['antecedents'])
        consequents = list(rule['consequents'])

        # Filter test set samples that meet antecedent conditions
        mask = np.ones(len(df_test), dtype=bool)
        for item in antecedents:
            if item in df_test.columns:
                mask &= df_test[item].values == 1

        # Calculate the number of samples in the test set that meet the antecedent conditions
        antecedent_count = np.sum(mask)

        # Calculate test set confidence
        if antecedent_count > 0:
            actual_consequents = y_test[mask].values
            predicted_consequent = consequents[0] if consequents else None
            if predicted_consequent:
                test_confidence = np.mean(actual_consequents == predicted_consequent)
                train_confidence = rule['confidence']
                base_rate = np.mean(y_test.values == predicted_consequent)
                lift = test_confidence / base_rate if base_rate > 0 else 0
                offset_rate = test_confidence / train_confidence if train_confidence > 0 else 0

                results.append({
                    'Antecedents': antecedents,
                    'Consequents': consequents,
                    'Training Confidence': train_confidence,
                    'Test Confidence': test_confidence,
                    'Confidence Offset Rate': offset_rate,
                    'Lift': lift,
                    'Antecedent Match Count': antecedent_count
                })
    return pd.DataFrame(results)


# ===== 3. Feature mapping table =====
feature_mapping = {
    'x1=1': "x1=1 (Permanent address type 1)",
    'x1=2': "x1=2 (Permanent address type 2)",
    'x3=1': "x3=1 (Mobile dialing records: 0-3)",
    'x3=2': "x3=2 (Mobile dialing records: 3-6)",
    'x3=3': "x3=3 (Mobile dialing records: 6-9)",
    'x3=4': "x3=4 (Mobile dialing records: 9-12)",
    'x3=5': "x3=5 (Mobile dialing records: >12)",
    'x7=1': "x7=1 (Emergency contact mobile status A)",
    'x7=2': "x7=2 (Emergency contact mobile status B)",
    'x7=3': "x7=3 (Emergency contact mobile status C)",
    'x7=4': "x7=4 (Emergency contact mobile status D)",
    'x7=5': "x7=5 (Emergency contact mobile status E)",
    'x7=6': "x7=6 (Emergency contact mobile status F)",
    'x8=1': "x8=1 (Emergency contact dialing records: 0-2)",
    'x8=2': "x8=2 (Emergency contact dialing records: 2-4)",
    'x8=3': "x8=3 (Emergency contact dialing records: 4-6)",
    'x8=4': "x8=4 (Emergency contact dialing records: 6-8)",
    'x8=5': "x8=5 (Emergency contact dialing records: >8)",
    'x11=1': "x11=1 (Loan type P)",
    'x11=2': "x11=2 (Loan type Q)",
    'x11=3': "x11=3 (Loan type R)",
    'x11=4': "x11=4 (Loan type S)",
    'x12=1': "x12=1 (Overdue days: 0-90)",
    'x12=2': "x12=2 (Overdue days: 90-180)",
    'x12=3': "x12=3 (Overdue days: 180-270)",
    'x12=4': "x12=4 (Overdue days: 270-365)",
    'x12=5': "x12=5 (Overdue days: >365)",
    'x13=1': "x13=1 (Overdue debt: 0-100000)",
    'x13=2': "x13=2 (Overdue debt: 100000-200000)",
    'x13=3': "x13=3 (Overdue debt: 200000-300000)",
    'x13=4': "x13=4 (Overdue debt: 300000-500000)",
    'x13=5': "x13=5 (Overdue debt: >500000)",
    'x14=0': "x14=0 (Estimated recoverable amount: 0)",
    'x14=1': "x14=1 (Estimated recoverable amount: 0-100000)",
    'x14=2': "x14=2 (Estimated recoverable amount: 100000-200000)",
    'x14=3': "x14=3 (Estimated recoverable amount: 200000-300000)",
    'x14=4': "x14=4 (Estimated recoverable amount: 300000-500000)",
    'x14=5': "x14=5 (Estimated recoverable amount: >500000)",
    'x15=1': "x15=1 (Lost-linking days: 0-30)",
    'x15=2': "x15=2 (Lost-linking days: 30-60)",
    'x15=3': "x15=3 (Lost-linking days: 60-90)",
    'x15=4': "x15=4 (Lost-linking days: 90-120)",
    'x15=5': "x15=5 (Lost-linking days: >120)",
    'x16=1': "x16=1 (Number of valid contacts: 0-1)",
    'x16=2': "x16=2 (Number of valid contacts: 1-2)",
    'x16=3': "x16=3 (Number of valid contacts: 2-3)",
    'x16=4': "x16=4 (Number of valid contacts: 3-4)",
    'x16=5': "x16=5 (Number of valid contacts: >4)",
    'x18=0': "x18=0 (Lost-linking Mode: Hide and Seek)",
    'x18=1': "x18=1 (Lost-linking Mode: Flee with the Money)",
    'x18=2': "x18=2 (Lost-linking Mode: False Disappearance)",
}


# Convert feature encodings in rules to actual feature names
def decode_rule(rule, mapping):
    antecedents = [mapping.get(f, f"Unknown_{f}") for f in rule['Antecedents']]
    consequents = [mapping.get(f, f"Unknown_{f}") for f in rule['Consequents']]
    return {
        'Decoded Antecedents': antecedents,
        'Decoded Consequents': consequents,
        'Training Confidence': rule['Training Confidence'],
        'Test Confidence': rule['Test Confidence'],
        'Confidence Offset Rate': rule['Confidence Offset Rate'],
        'Lift': rule['Lift'],
        'Antecedent Match Count': rule['Antecedent Match Count']
    }


# ===== 4. Core rule analysis=====
print("\n===== Rule examples (first 10) =====")
print(rules.head(10)[['antecedents', 'consequents', 'confidence']])

# Adjust core lost-linking mode rules based on business logic
core_patterns = [
    frozenset({'x1=1', 'x3=3', 'x11=1'}),  # Common combinations from rule examples
    frozenset({'x8=2', 'x12=3', 'x15=2'}),  # Common combinations from rule examples
    frozenset({'x16=2', 'x15=3'})  # Common combinations from rule examples
]

# Find core rules
core_rules_selected = pd.DataFrame()
for pattern in core_patterns:
    matching_rules = rules[rules['antecedents'].apply(lambda x: pattern.issubset(x))]
    if len(matching_rules) > 0:
        best_rule = matching_rules.loc[matching_rules['confidence'].idxmax()]
        core_rules_selected = pd.concat([core_rules_selected, pd.DataFrame([best_rule])], ignore_index=True)

print(f"\nFound {len(core_rules_selected)} core rules")

# Perform test set verification
if len(core_rules_selected) > 0:
    core_evaluation = evaluate_rules_on_test(core_rules_selected, df_encoded_test, y_test)

    if len(core_evaluation) > 0:
        print("\n===== Core lost-linking mode rule test set verification results (original encoding) =====")
        print(core_evaluation[
                  ['Antecedents', 'Consequents', 'Training Confidence', 'Test Confidence', 'Confidence Offset Rate',
                   'Lift',
                   'Antecedent Match Count']])

        # Decode rules and output
        decoded_core_evaluation = core_evaluation.apply(lambda x: decode_rule(x, feature_mapping), axis=1,
                                                        result_type='expand')
        print("\n===== Core lost-linking mode rule test set verification results (decoded) =====")
        print(decoded_core_evaluation)
    else:
        print("\nWarning: No samples matched core rules in the test set!")
else:
    print("\nWarning: No rules matching core patterns found! Using top 10 high-confidence rules as a substitute")
    # Use top 10 high-confidence rules as a substitute
    core_rules_selected = rules.sort_values('confidence', ascending=False).head(10)
    core_evaluation = evaluate_rules_on_test(core_rules_selected, df_encoded_test, y_test)

    if len(core_evaluation) > 0:
        print("\n===== High-confidence rule test set verification results (original encoding) =====")
        print(core_evaluation[
                  ['Antecedents', 'Consequents', 'Training Confidence', 'Test Confidence', 'Confidence Offset Rate',
                   'Lift',
                   'Antecedent Match Count']])

        # Decode rules and output
        decoded_core_evaluation = core_evaluation.apply(lambda x: decode_rule(x, feature_mapping), axis=1,
                                                        result_type='expand')
        print("\n===== High-confidence rule test set verification results (decoded) =====")
        print(decoded_core_evaluation)
    else:
        print("\nWarning: No samples matched high-confidence rules in the test set!")

# ===== 5. Overall rule generalization capability evaluation=====
all_rules_evaluation = evaluate_rules_on_test(rules, df_encoded_test, y_test)

if len(all_rules_evaluation) > 0:
    print(f"\n===== All rules generalization capability evaluation =====")
    print(
        f"Number of rules with matches in test set: {len(all_rules_evaluation)}/{len(rules)} ({len(all_rules_evaluation) / len(rules):.2%})")
    print(f"Average test set confidence: {all_rules_evaluation['Test Confidence'].mean():.4f}")
    print(f"Average confidence offset rate: {all_rules_evaluation['Confidence Offset Rate'].mean():.4f}")
    print(
        f"Proportion of rules with confidence offset rate ≥ 0.9: {(all_rules_evaluation['Confidence Offset Rate'] >= 0.9).mean():.2%}")

    # Output the top 5 performing rules (decoded)
    top_rules = all_rules_evaluation.sort_values('Test Confidence', ascending=False).head(5)
    decoded_top_rules = top_rules.apply(lambda x: decode_rule(x, feature_mapping), axis=1, result_type='expand')
    print("\n===== Top 5 performing rules (decoded) =====")
    print(decoded_top_rules)

    # Output the top 5 most robust rules
    robust_rules = all_rules_evaluation.sort_values('Confidence Offset Rate', ascending=False).head(5)
    decoded_robust_rules = robust_rules.apply(lambda x: decode_rule(x, feature_mapping), axis=1, result_type='expand')
    print("\n===== Top 5 most robust rules (decoded) =====")
    print(decoded_robust_rules)
else:
    print("\nWarning: No rules matched samples in the test set!")

# ===== 6. Rule optimization suggestions=====
print("\n===== Rule optimization suggestions =====")
# Calculate rule support in the test set
rule_supports = []
for _, rule in rules.iterrows():
    antecedents = list(rule['antecedents'])
    mask = np.ones(len(df_encoded_test), dtype=bool)
    for item in antecedents:
        if item in df_encoded_test.columns:
            mask &= df_encoded_test[item].values == 1
    support = np.sum(mask) / len(df_encoded_test)
    rule_supports.append(support)

# Analyze rule quality under different support thresholds
support_thresholds = [0.05, 0.1, 0.15, 0.2]
for threshold in support_thresholds:
    filtered_rules = rules[rules['support'] >= threshold]
    filtered_evaluation = evaluate_rules_on_test(filtered_rules, df_encoded_test, y_test)
    if len(filtered_evaluation) > 0:
        avg_confidence = filtered_evaluation['Test Confidence'].mean()
        avg_offset = filtered_evaluation['Confidence Offset Rate'].mean()
        print(
            f"Support threshold {threshold}: {len(filtered_rules)} rules, average test confidence: {avg_confidence:.4f}, average offset rate: {avg_offset:.4f}")
    else:
        print(f"Support threshold {threshold}: No valid rules")

# ===== 7. Final rule screening suggestions=====
print("\n===== Final rule screening suggestions =====")
if len(all_rules_evaluation) > 0:
    # Screen high-quality rules
    high_quality_rules = all_rules_evaluation[
        (all_rules_evaluation['Test Confidence'] >= 0.5) &
        (all_rules_evaluation['Confidence Offset Rate'] >= 0.8)
        ]

    if len(high_quality_rules) > 0:
        print(
            f"Number of rules meeting high-quality standards: {len(high_quality_rules)}/{len(rules)} ({len(high_quality_rules) / len(rules):.2%})")
        decoded_high_quality_rules = high_quality_rules.apply(lambda x: decode_rule(x, feature_mapping), axis=1,
                                                              result_type='expand')
        print("\n===== High-quality rule list (decoded) =====")
        print(decoded_high_quality_rules)
    else:
        print("No rules meeting high-quality standards found, please lower screening criteria")

        # Screen again with lower standards
        medium_quality_rules = all_rules_evaluation[
            (all_rules_evaluation['Test Confidence'] >= 0.3) &
            (all_rules_evaluation['Confidence Offset Rate'] >= 0.5)
            ]

        if len(medium_quality_rules) > 0:
            print(
                f"Number of rules meeting medium-quality standards: {len(medium_quality_rules)}/{len(rules)} ({len(medium_quality_rules) / len(rules):.2%})")
            decoded_medium_quality_rules = medium_quality_rules.apply(lambda x: decode_rule(x, feature_mapping), axis=1,
                                                                      result_type='expand')
            print("\n===== Medium-quality rule list (decoded) =====")
            print(decoded_medium_quality_rules)

            # Pay special attention to rules containing key features
            key_features = ['x8=2', 'x11=1']
            key_rules = medium_quality_rules[
                medium_quality_rules['Antecedents'].apply(lambda x: any(f in x for f in key_features))
            ]

            if len(key_rules) > 0:
                print("\n===== Medium-quality rules containing key features (decoded) =====")
                decoded_key_rules = key_rules.apply(lambda x: decode_rule(x, feature_mapping), axis=1,
                                                    result_type='expand')
                print(decoded_key_rules)
            else:
                print("\nNo medium-quality rules containing key features found")
        else:
            print("No rules meeting medium-quality standards found, suggest readjusting mining parameters")
else:
    print("Cannot provide rule screening suggestions, all rules have no matching samples in the test set")

# ===== 8. Business rule visualization =====
if len(all_rules_evaluation) > 0:
    # Rename columns
    all_rules_evaluation = all_rules_evaluation.rename(columns={
        'Lift': 'Lift',
        'Antecedent Match Count': 'Antecedent Match Count'
    })

    # Create rule quality evaluation plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='Test Confidence',
        y='Confidence Offset Rate',
        size='Antecedent Match Count',
        hue='Lift',
        data=all_rules_evaluation,
        sizes=(50, 200),
        palette='viridis'
    )

    # Add reference lines
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=0.3, color='r', linestyle='--', alpha=0.3)

    # Add title and labels
    plt.title('Rule Quality Evaluation Scatter Plot')
    plt.xlabel('Test Set Confidence')
    plt.ylabel('Confidence Deviation Rate')
    plt.grid(True, alpha=0.3)

    # Add color bar legend explanation
    plt.figtext(0.5, 0.01,
                'Color indicates Lift value (darker = higher lift), Size indicates Antecedent Match Count',
                ha='center', fontsize=7)

    # Save the image
    plt.savefig('rule_quality_evaluation.png', dpi=300, bbox_inches='tight')
    print("\nRule quality evaluation plot generated: rule_quality_evaluation.png")

    # Extract features of key rules
    if 'decoded_medium_quality_rules' in locals() and len(decoded_medium_quality_rules) > 0:
        # Count the occurrence frequency of features in medium-quality rules
        feature_counts = {}
        for _, rule in decoded_medium_quality_rules.iterrows():
            for feature in rule['Decoded Antecedents'] + rule['Decoded Consequents']:
                if feature in feature_counts:
                    feature_counts[feature] += 1
                else:
                    feature_counts[feature] = 1

        # Sort and display the top 10 features
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        print("\n===== Most frequently occurring features in medium-quality rules =====")
        for feature, count in sorted_features[:10]:
            print(f"{feature}: {count} times")

# ===== 9. Save key rule results =====
if 'decoded_medium_quality_rules' in locals():
    # Save medium-quality rules to Excel file
    decoded_medium_quality_rules.to_excel('medium_quality_rules.xlsx', index=False)
    print("\nMedium-quality rules saved to: medium_quality_rules.xlsx")

    # Save rules containing key features to Excel file
    if 'decoded_key_rules' in locals():
        decoded_key_rules.to_excel('key_rules.xlsx', index=False)
        print("\nRules containing key features saved to: key_rules.xlsx")

print("All analyses completed. Results and visualizations have been saved.")