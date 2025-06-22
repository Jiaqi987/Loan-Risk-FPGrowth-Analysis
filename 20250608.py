import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import squarify
import numpy as np

# Load data
data = pd.read_excel('20250608.xlsx')

# Encode categorical features
data['x2'] = data['x2'].replace({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6})
data['x11'] = data['x11'].replace({'P': 1, 'Q': 2, 'R': 3, 'S': 4})

# Discretize continuous features into bins
# Discretize x3
data['x3'] = pd.cut(data['x3'], bins=[0, 3, 6, 9, 12, float('inf')], labels=[1, 2, 3, 4, 5])
# Discretize x8
data['x8'] = pd.cut(data['x8'], bins=[0, 2, 4, 6, 8, float('inf')], labels=[1, 2, 3, 4, 5])
# Discretize x12
data['x12'] = pd.cut(data['x12'], bins=[0, 90, 180, 270, 365, float('inf')], labels=[1, 2, 3, 4, 5])
# Discretize x13
data['x13'] = pd.cut(data['x13'], bins=[0, 100000, 200000, 300000, 500000, float('inf')], labels=[1, 2, 3, 4, 5])
# Discretize x14
data['x14'] = data['x14'].apply(lambda x: 0 if x == 0 else pd.cut([x], bins=[0, 100000, 200000, 300000, 500000, float('inf')], labels=[1, 2, 3, 4, 5])[0])
# Discretize x15
data['x15'] = pd.cut(data['x15'], bins=[0, 30, 60, 90, 120, float('inf')], labels=[1, 2, 3, 4, 5])
# Discretize x16
data['x16'] = pd.cut(data['x16'], bins=[0, 1, 2, 3, 4, float('inf')], labels=[1, 2, 3, 4, 5])

# Save processed data to new Excel file
data.to_excel('processed_data.xlsx', index=False)

# Convert all features to string type for transaction data construction
data = data.astype(str)

# FP-Growth requires a list of transactions, where each transaction is a set of items
# Here we treat each row as a transaction and each feature as an item
transactions = []
for index, row in data.iterrows():
    transaction = row[['x1', 'x2', 'x3',  'x8',  'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x18']].tolist()
    transactions.append(transaction)

# Save transaction data to Excel
transactions_df = pd.DataFrame(transactions, columns=['x1', 'x2', 'x3', 'x8', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x18'])
transactions_df.to_excel('transactions.xlsx', index=False)

# Initialize TransactionEncoder
te = TransactionEncoder()

# Use TransactionEncoder's fit method to learn items from transactions and transform to one-hot encoding
te_ary = te.fit(transactions).transform(transactions)

# Convert one-hot encoded array to DataFrame for mlxtend's fpgrowth function
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Now df_encoded is a DataFrame containing one-hot encoded transactions, which we can use to find frequent itemsets
frequent_itemsets = fpgrowth(df_encoded, min_support=0.1, use_colnames=True)

# Extract association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Save frequent itemsets to Excel
frequent_itemsets.to_excel('frequent_itemsets.xlsx', index=False)

# Save association rules to Excel
rules.to_excel('association_rules.xlsx', index=False)

# Extract antecedents, consequents, and support from association rules
rules_data = []
for index, rule in rules.iterrows():
    antecedents = rule['antecedents']
    consequents = rule['consequents']
    support = frequent_itemsets.loc[frequent_itemsets['itemsets'] == antecedents,'support'].values[0]
    rules_data.append((antecedents, consequents, support))

# Prepare heatmap data
data_matrix = [[0] * len(frequent_itemsets) for _ in range(len(frequent_itemsets))]
for i, itemset1 in enumerate(frequent_itemsets['itemsets']):
    for j, itemset2 in enumerate(frequent_itemsets['itemsets']):
        common_items = itemset1.intersection(itemset2)
        data_matrix[i][j] = len(common_items)

data_matrix_df = pd.DataFrame(data_matrix, columns=frequent_itemsets['itemsets'], index=frequent_itemsets['itemsets'])

# ================== Heatmap ==================
# Set global font and image resolution
plt.rcParams['font.size'] = 8
plt.rcParams['font.family'] = 'DejaVu Sans'  # Use more compatible font
plt.rcParams['figure.dpi'] = 300  # Set high DPI

# Create heatmap
plt.figure(figsize=(16, 14))  # Increase image size
ax = sns.heatmap(
    data_matrix_df,
    annot=True,
    cmap='YlGnBu',
    fmt='d',  # Display integers
    annot_kws={'fontsize': 6}  # Reduce annotation font size
)

# Restore original coordinate label format
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
plt.tight_layout()  # Automatically adjust layout

# Save high-quality image
plt.savefig('itemset_overlap_heatmap.png', dpi=600, bbox_inches='tight')
plt.close()  # Close current figure to avoid overlaps

# ================== Treemap ==================
# Ensure values variable is properly defined
values = [support for antecedents, consequents, support in rules_data]
labels = []

for antecedents, consequents, support in rules_data:
    # Simplify label display
    label = f"{','.join(map(str, antecedents))[:15]}... -> {','.join(map(str, consequents))[:10]}..."
    labels.append(label)

# Plot treemap
plt.figure(figsize=(24, 18), dpi=300)
squarify.plot(
    sizes=values,
    label=labels,
    alpha=.8,
    text_kwargs={
        'fontsize': 7,  # Reduce font size
        'wrap': True    # Allow text wrapping
    }
)

plt.axis('off')
plt.title('Treemap of Association Rules', fontsize=12)

# Save high-quality image
plt.savefig('association_rules_treemap.png', dpi=600, bbox_inches='tight')
plt.close()

print("Processed images saved as high-resolution PNG files")