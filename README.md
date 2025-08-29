# Loan Customer Lost-Linking Risk Association Rule Mining Project

## I. Project Overview

### 1.1 Research Background
This project focuses on the lost-linking behavior of loan customers, leveraging data mining techniques to uncover association rules between lost-linking features and three typical modes ("Hide and Seek," "Flee with the Money," and "False Disappearance"). The goal is to provide financial institutions with quantitative risk warning foundations and support precise risk control.

### 1.2 Data/Code Purpose
This project includes 256 desensitized samples of loan lost-linking customers (original data: `Lost-linking Dataset.csv`) and supporting analysis code. Using the FP-Growth algorithm, it mines association rules between lost-linking features and modes, ultimately constructing a risk warning system framework to enable financial institutions to monitor and respond to lost-linking risks in a graded and real-time manner. The code is divided into two parts:

- `fp_growth_lost_linking_association_mining_loan_risk.py`: This code uses the FP-Growth algorithm to mine association rules related to loan risk, focusing on analyzing lost-linking modes (x18) by preprocessing data, identifying frequent itemsets and core rules, evaluating their generalization ability, and visualizing results.
- `lost_linking_association_rule_network_visualization.py`: This code visualizes an association rule network between customer features and lost-linking modes using Graphviz, creating a PNG image with nodes representing features/modes and edges indicating confidence levels of associations.


## II. Data Description

### 2.1 Data Source and Processing
The original data is stored in `Lost-linking Dataset.csv`, containing 18 features (including 1 target variable: "lost-linking mode"). To enhance model robustness, based on prior research, 7 low-contribution features were removed. The removed features are:
- x4: SMS signaling
- x5: email features
- x6: other signaling features (APPs/mini-programs/official accounts)
- x2: mobile number network status
- x9: emergency contact SMS signaling
- x10: emergency contact email features
- x17: valid contact relationship

The processed data, which retains 11 core features, is stored in `lost_linking_processed_10f.xlsx`.

### 2.2 Key Retained Features
The processed data includes 11 features (including 1 target variable: "lost-linking mode"). Key feature details are as follows (full details in `lost_linking_processed_10f.xlsx`):

| Feature Code | Feature Name | Description |
|--------------|--------------|-------------|
| x1 | Permanent Address Matching | 1=registered/business address, 0=other (discrete) |
| x3 | Mobile Call Record Count | Non-negative integer (continuous, segmented and encoded) |
| x7 | Emergency Contact Mobile Number Network Status | A=normal, B=out of service, C=active but unreachable, D=unassigned, E=unactivated, F=abnormal (discrete) |
| x8 | Emergency Contact Mobile Call Record Count | Non-negative integer (continuous, segmented and encoded) |
| x11 | Loan Type | P=guaranteed, Q=mortgage, R=credit, S=discount (discrete) |
| x12 | Overdue Days | Non-negative real number (continuous, segmented into intervals like [0,90), [90,180), etc., encoded) |
| x13 | Overdue Amount | Non-negative real number (continuous, segmented into intervals like [0,100,000), [100,000,200,000), etc., encoded) |
| x14 | Estimated Recoverable Amount | Non-negative real number (continuous, segmented and encoded) |
| x15 | Lost-Linking Days | Non-negative real number (continuous, segmented into intervals like [0,30), [30,60), etc., encoded) |
| x16 | Valid Contact Count | Non-negative integer (continuous, segmented and encoded) |
| x18 | Lost-Linking Mode | 0="Hide and Seek" (active mobile but evasive), 1="Flee with the Money" (malicious asset transfer), 2="False Disappearance" (fabricated loss of contact) (target variable) |

### 2.3 Desensitization Method
The data has been stripped of real user privacy information (e.g., names, IDs, contact details), retaining only structured features related to communication behavior (call records, mobile status) and loan qualifications (overdue days, amounts) to comply with privacy protection requirements.


## III. Code Dependencies
The project’s code is developed in Python. Required libraries (with recommended versions) are listed below (using Anaconda for environment management is advised):

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.10+ | Base runtime environment |
| pandas | 2.1.0+ | Data reading, cleaning, and preprocessing |
| matplotlib | 3.7.0+ | Visualization (heatmaps, treemaps) |
| seaborn | 0.12.2+ | Visualization support |
| mlxtend | 0.21.0+ | FP-Growth algorithm implementation |
| numpy | 1.24.0+ | Numerical computations |
| scikit-learn | 1.2.2+ | Dataset partitioning (train_test_split) |


## IV. Execution Steps

### 4.1 Data Preprocessing and Rule Mining (`fp_growth_lost_linking_association_mining_loan_risk.py`)
Run `fp_growth_lost_linking_association_mining_loan_risk.py` to perform the following operations:
1. Read processed data (`lost_linking_processed_10f.xlsx`, with 7 low-contribution features removed).
2. Encode discrete features (e.g., x7: A-F → 1-6).
3. Segment continuous features (e.g., x12: overdue days into intervals like [0,90), [90,180), etc., and encode).
4. Generate cleaned data (`processed_data.xlsx`) and transactional data (`transactions.xlsx`).
5. Mine frequent itemsets (output: `frequent_itemsets.xlsx`) and association rules (output: `association_rules.xlsx`).
6. Filter core itemsets (containing x18, support>0.2, output: `core_itemsets.xlsx`) and core rules (x18 in consequents, lift>1.2, output: `core_rules.xlsx`).

### 4.2 Parameter Selection and Generalization Verification (`fp_growth_lost_linking_association_mining_loan_risk.py`)
Run `fp_growth_lost_linking_association_mining_loan_risk.py` to independently execute the "Dataset Partitioning and Generalization Capability Verification" step described in the paper, including:
1. Dataset partitioning (70% training, 30% test, stratified by lost-linking mode).
2. Validate rule performance on the test set (confidence, lift, and offset rate).
3. Screen core rules (e.g., top high-confidence rules containing x18) and analyze their generalization (output: `rule_quality_evaluation.png`; core rules are derived from `core_rules.xlsx`).
4. Provide rule optimization suggestions (e.g., adjusting support thresholds) and quality assessments.

### 4.3 Result Visualization
#### From `fp_growth_lost_linking_association_mining_loan_risk.py`:
- `top_core_itemset_heatmap.png`: Heatmap of overlaps between top 50 core itemsets (annotated with counts of common items, using a YlGnBu colormap).
- `top_core_rules_network.png`: Directed graph of top 30 core rules (nodes styled by type, edges colored by confidence levels with embedded confidence/lift labels and a legend).
- `rule_quality_evaluation.png`: Scatterplot of rule quality (x: test confidence, y: offset rate, size: antecedent match count, color: lift, with reference lines for quality thresholds).

#### From `lost_linking_association_rule_network_visualization.py`:
- `Loan_LostLinking_Network.png`: Association rule network between customer features and lost-linking modes.


## V. Data/Code Citation
The project’s data and code are stored in a GitHub repository (link: https://github.com/Jiaqi987/Loan-Risk-FPGrowth-Analysis) and have been assigned a persistent DOI via Zenodo (DOI: 10.5281/zenodo.15714220). For citations, use:

> Data and code are from: v1.1: Loan Risk FPGrowth Analysis (Data & Code), accessible via DOI: 10.5281/zenodo.15714220 or the GitHub repository.


## Notes
1. Ensure data files (`lost_linking_processed_10f.xlsx`) and scripts (`fp_growth_lost_linking_association_mining_loan_risk.py`) are in the same directory before execution.
2. Adjust algorithm parameters (e.g., support, confidence) directly in the `fpgrowth` and `association_rules` functions within the scripts.
3. `lost_linking_association_rule_network_visualization.py` is an independent verification script; use `core_rules_filtered_mode0.xlsx` directly.
4. **Visualization Dependencies**: To generate the "Lost-linking Mode=0" network graph, install the Graphviz software (not just the Python library) and add its path to the system environment variables (download: https://graphviz.org/download/), as required by the paper’s visualization methodology.
