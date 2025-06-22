import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set font to avoid warnings
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# ===== 1. Data loading and preprocessing =====
data = pd.read_excel('20250608.xlsx')

# Data preprocessing
data['x2'] = data['x2'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6})
data['x11'] = data['x11'].map({'P': 1, 'Q': 2, 'R': 3, 'S': 4})

# Discretize continuous features
data['x12'] = pd.cut(data['x12'], bins=[0, 30, 60, 90, 120, float('inf')], labels=[1, 2, 3, 4, 5])
data['x13'] = pd.cut(data['x13'], bins=[0, 10000, 30000, 50000, 100000, float('inf')], labels=[1, 2, 3, 4, 5])


# ===== 2. Build transaction data and mine rules =====
def build_transactions(df):
    transactions = []
    for _, row in df.iterrows():
        transaction = row[['x1', 'x2', 'x3', 'x8', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16']].tolist()
        transactions.append(transaction)
    return transactions


# Build full dataset transactions
transactions_full = build_transactions(data)

# Encode transaction data
te_full = TransactionEncoder()
te_ary_full = te_full.fit_transform(transactions_full)
df_encoded_full = pd.DataFrame(te_ary_full, columns=te_full.columns_)

# Mine frequent itemsets and association rules
frequent_itemsets_full = fpgrowth(df_encoded_full, min_support=0.1, use_colnames=True)
rules_full = association_rules(frequent_itemsets_full, metric="confidence", min_threshold=0.7)
print(f"Full dataset mining results: {len(frequent_itemsets_full)} frequent itemsets, {len(rules_full)} association rules")

# ===== 3. Dataset splitting and generalization capability verification =====
# Split dataset
X = data.drop('x18', axis=1)
y = data['x18']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print(f"Training set samples: {len(X_train)}, Test set samples: {len(X_test)}")

# Build test set transaction data
transactions_test = build_transactions(X_test)
te_ary_test = te_full.transform(transactions_test)
df_encoded_test = pd.DataFrame(te_ary_test, columns=te_full.columns_)


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
            predicted_consequent = consequents[0]
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


# ===== 4. Feature mapping table (adjusted based on business logic) =====
# Adjust feature mapping based on business logic
feature_mapping = {
    0.0: "Lost-linking Mode=0",
    1.0: "x1=1",
    2.0: "x2=2",
    3.0: "x3=3",
    4.0: "x4=4",
    5.0: "x5=5",
    6.0: "x6=6",
    7.0: "x7=7",
    8.0: "x8=8",
    9.0: "x9=9",
    10.0: "x10=10",
    11.0: "x11=11",
    12.0: "x12=12",
    13.0: "x13=13",
    14.0: "x14=14",
    15.0: "x15=15",
    16.0: "x16=16",}


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


# ===== 5. Core rule analysis =====
print("\n===== Rule examples (first 10) =====")
print(rules_full.head(10)[['antecedents', 'consequents', 'confidence']])

# Adjust core lost-linking mode rules based on business logic
core_patterns = [
    frozenset({2.0, 4.0, 5.0}),  # Common combinations from rule examples
    frozenset({0.0, 2.0, 4.0}),  # Common combinations from rule examples
    frozenset({8.0, 2.0, 3.0, 5.0})  # Common combinations from rule examples
]

# Find core rules
core_rules = pd.DataFrame()
for pattern in core_patterns:
    matching_rules = rules_full[rules_full['antecedents'].apply(lambda x: pattern.issubset(x))]
    if len(matching_rules) > 0:
        best_rule = matching_rules.loc[matching_rules['confidence'].idxmax()]
        core_rules = pd.concat([core_rules, pd.DataFrame([best_rule])], ignore_index=True)

print(f"\nFound {len(core_rules)} core rules")

# Perform test set verification
if len(core_rules) > 0:
    core_evaluation = evaluate_rules_on_test(core_rules, df_encoded_test, y_test)

    if len(core_evaluation) > 0:
        print("\n===== Core lost-linking mode rule test set verification results (original encoding) =====")
        print(core_evaluation[['Antecedents', 'Consequents', 'Training Confidence', 'Test Confidence', 'Confidence Offset Rate', 'Lift',
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
    core_rules = rules_full.sort_values('confidence', ascending=False).head(10)
    core_evaluation = evaluate_rules_on_test(core_rules, df_encoded_test, y_test)

    if len(core_evaluation) > 0:
        print("\n===== High-confidence rule test set verification results (original encoding) =====")
        print(core_evaluation[['Antecedents', 'Consequents', 'Training Confidence', 'Test Confidence', 'Confidence Offset Rate', 'Lift',
                               'Antecedent Match Count']])

        # Decode rules and output
        decoded_core_evaluation = core_evaluation.apply(lambda x: decode_rule(x, feature_mapping), axis=1,
                                                        result_type='expand')
        print("\n===== High-confidence rule test set verification results (decoded) =====")
        print(decoded_core_evaluation)
    else:
        print("\nWarning: No samples matched high-confidence rules in the test set!")

# ===== 6. Overall rule generalization capability evaluation =====
all_rules_evaluation = evaluate_rules_on_test(rules_full, df_encoded_test, y_test)

if len(all_rules_evaluation) > 0:
    print(f"\n===== All rules generalization capability evaluation =====")
    print(
        f"Number of rules with matches in test set: {len(all_rules_evaluation)}/{len(rules_full)} ({len(all_rules_evaluation) / len(rules_full):.2%})")
    print(f"Average test set confidence: {all_rules_evaluation['Test Confidence'].mean():.4f}")
    print(f"Average confidence offset rate: {all_rules_evaluation['Confidence Offset Rate'].mean():.4f}")
    print(f"Proportion of rules with confidence offset rate ≥ 0.9: {(all_rules_evaluation['Confidence Offset Rate'] >= 0.9).mean():.2%}")

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

# ===== 7. Rule optimization suggestions =====
print("\n===== Rule optimization suggestions =====")
# Calculate rule support in the test set
rule_supports = []
for _, rule in rules_full.iterrows():
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
    filtered_rules = rules_full[rules_full['support'] >= threshold]
    filtered_evaluation = evaluate_rules_on_test(filtered_rules, df_encoded_test, y_test)
    if len(filtered_evaluation) > 0:
        avg_confidence = filtered_evaluation['Test Confidence'].mean()
        avg_offset = filtered_evaluation['Confidence Offset Rate'].mean()
        print(
            f"Support threshold {threshold}: {len(filtered_rules)} rules, average test confidence: {avg_confidence:.4f}, average offset rate: {avg_offset:.4f}")
    else:
        print(f"Support threshold {threshold}: No valid rules")

# ===== 8. Final rule screening suggestions =====
print("\n===== Final rule screening suggestions =====")
if len(all_rules_evaluation) > 0:
    # Screen high-quality rules
    high_quality_rules = all_rules_evaluation[
        (all_rules_evaluation['Test Confidence'] >= 0.5) &
        (all_rules_evaluation['Confidence Offset Rate'] >= 0.8)
        ]

    if len(high_quality_rules) > 0:
        print(
            f"Number of rules meeting high-quality standards: {len(high_quality_rules)}/{len(rules_full)} ({len(high_quality_rules) / len(rules_full):.2%})")
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
                f"Number of rules meeting medium-quality standards: {len(medium_quality_rules)}/{len(rules_full)} ({len(medium_quality_rules) / len(rules_full):.2%})")
            decoded_medium_quality_rules = medium_quality_rules.apply(lambda x: decode_rule(x, feature_mapping), axis=1,
                                                                      result_type='expand')
            print("\n===== Medium-quality rule list (decoded) =====")
            print(decoded_medium_quality_rules)

            # Pay special attention to rules containing key features
            key_features = [2.0, 8.0]  # x2=2 and x8=8
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

# ===== 9. Business rule visualization =====
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

    # Save the image
    plt.savefig('rule_quality_evaluation.png', dpi=300, bbox_inches='tight')
    print("\nRule quality evaluation plot generated: rule_quality_evaluation.png")

    # Extract features of key rules
    if len(decoded_medium_quality_rules) > 0:
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

# ===== 10. Save key rule results =====
if 'decoded_medium_quality_rules' in locals():
    # Save medium-quality rules to Excel file
    decoded_medium_quality_rules.to_excel('medium_quality_rules.xlsx', index=False)
    print("\nMedium-quality rules saved to: medium_quality_rules.xlsx")

    # Save rules containing key features to Excel file
    if 'decoded_key_rules' in locals():
        decoded_key_rules.to_excel('key_rules.xlsx', index=False)
        print("\nRules containing key features saved to: key_rules.xlsx")