# Loan Customer Lost-Linking Risk Association Rule Mining Project  

## I. Project Overview  

### 1.1 Research Background  
This project focuses on the lost-linking behavior of loan customers, leveraging data mining techniques to uncover association rules between lost-linking features and three typical modes ("Hide and Seek," "Flee with the Money," and "False Disappearance"). The goal is to provide financial institutions with quantitative risk warning foundations and support precise risk control.  

### 1.2 Data/Code Purpose  
This project includes 256 desensitized samples of loan lost-linking customers (original data: `Lost-linking Dataset.csv`) and supporting analysis code. Using the FP-Growth algorithm, it mines association rules between lost-linking features and modes, ultimately constructing a risk warning system framework to enable financial institutions to monitor and respond to lost-linking risks in a graded and real-time manner.  

The code is divided into two parts:  
- `20250608.py`: Main workflow for data preprocessing and association rule mining.  
- `20250613.py`: Dedicated to "Dataset Partitioning and Generalization Capability Verification" (parameter selection and sensitivity analysis) as described in the paper, validating the model’s generalization performance.  


## II. Data Description  

### 2.1 Data Source and Processing  
The original data is stored in `Lost-linking Dataset.csv`, containing 18 features (including 1 target variable: "lost-linking mode"). To enhance model robustness, based on prior research, 7 low-contribution features were removed (details in the paper). The processed data, retaining 11 core features, is stored in `20250608.xlsx`.  

### 2.2 Key Retained Features  
The processed data includes 11 features (including 1 target variable: "lost-linking mode"). Key feature details are as follows:  

| Feature Code | Feature Name | Description |  
|--------------|--------------|-------------|  
| x1 | Permanent Address Matching | 1=registered/business address, 0=other (discrete) |  
| x2 | Mobile Number Network Status | A=normal, B=out of service, C=active but unreachable, D=unassigned, E=unactivated, F=abnormal (discrete) |  
| x3 | Mobile Call Record Count | Non-negative integer (continuous, segmented and encoded) |  
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

| Library          | Version  | Purpose                                  |  
|-------------------|----------|------------------------------------------|  
| Python            | 3.10+    | Base runtime environment                 |  
| pandas            | 2.1.0+   | Data reading, cleaning, and preprocessing |  
| matplotlib        | 3.7.0+   | Visualization (heatmaps, treemaps)       |  
| seaborn           | 0.12.2+  | Visualization support                    |  
| mlxtend           | 0.21.0+  | FP-Growth algorithm implementation       |  
| numpy             | 1.24.0+  | Numerical computations                   |  
| scikit-learn      | 1.2.2+   | Dataset partitioning (train_test_split)  |  


## IV. Execution Steps  

### 4.1 Data Preprocessing and Rule Mining (`20250608.py`)  
Run `20250608.py` to perform:  
- Read processed data (`20250608.xlsx`, with 7 low-contribution features removed).  
- Encode discrete features (e.g., x2: A-F → 1-6).  
- Segment continuous features (e.g., x12: overdue days into intervals like [0,90), [90,180), etc., and encode).  
- Generate cleaned data (`processed_data.xlsx`) and transactional data (`transactions.xlsx`).  
- Mine frequent itemsets (output: `frequent_itemsets.xlsx`) and association rules (output: `association_rules.xlsx`).  

### 4.2 Parameter Selection and Generalization Verification (`20250613.py`)  
Run `20250613.py` to independently execute the "Dataset Partitioning and Generalization Capability Verification" step described in the paper, including:  
- Dataset partitioning (70% training, 30% test, stratified by lost-linking mode).  
- Validate rule performance on the test set (confidence, lift, and offset rate).  
- Screen core rules and analyze their generalization (output: `rule_quality_evaluation.png`, `medium_quality_rules.xlsx`, etc.).  
- Provide rule optimization suggestions (e.g., adjusting support thresholds) and quality assessments.  

### 4.3 Result Visualization  
Both scripts generate the following visualizations:  
- `itemset_overlap_heatmap.png` (from `20250608.py`): Heatmap of frequent itemset overlaps.  
- `association_rules_treemap.png` (from `20250608.py`): Treemap of association rules.  
- `rule_quality_evaluation.png` (from `20250613.py`): Scatterplot of rule quality assessment.  


## V. Data/Code Citation  
The project’s data and code are stored in a GitHub repository (link: https://github.com/Jiaqi987/Loan-Risk-FPGrowth-Analysis) and have been assigned a persistent DOI via Zenodo (DOI: 10.5281/zenodo.15714220). For citations, use:  
> Data and code are from: v1.0: Loan Risk FPGrowth Analysis (Data & Code), accessible via DOI: 10.5281/zenodo.15714220 or the GitHub repository.  


### Notes  
- Ensure data files (`20250608.xlsx`) and scripts (`20250608.py`, `20250613.py`) are in the same directory before execution.  
- Adjust algorithm parameters (e.g., support, confidence) directly in the `fpgrowth` and `association_rules` functions within the scripts.  
- `20250613.py` is an independent verification script; run it after `20250608.py` generates transactional data (or use `20250608.xlsx` directly).
