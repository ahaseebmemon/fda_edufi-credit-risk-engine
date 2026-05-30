# fda_edufi-credit-risk-engine
machine learning credit risk engine that predicts loan defaults by evaluating traditional financial metrics against alternative educational and geographic data.

# ⚡ EduFi: Alternative Data Engineering Pipeline

A Python-based data augmentation and feature engineering pipeline designed to bridge traditional financial data with alternative educational and geographic risk indicators.

## Overview
This repository contains the data preparation architecture for the EduFi credit risk engine. Traditional credit scoring relies on historical financial ledgers. This notebook takes a base dataset of borrowers and engineers a robust set of alternative metrics based on real-world institutional prestige, program employability, and geographic affluence to build a more comprehensive risk profile.

## Pipeline Architecture
The pipeline is contained entirely within a Jupyter Notebook and executes the following sequence:

### 1. Categorical Simulation & Cleaning
To simulate a real-world loan application process, the pipeline first cleans the base data (`primary_selected_with_customer_id.csv`) and maps borrowers to specific categories:
- **Universities:** LUMS, IBA, NUST, FAST, COMSATS, UOL
- **Programs:** Computer Science, Business, Engineering, Psychology, Media, Fashion
- **Cities:** Karachi, Lahore, Islamabad, Multan, Quetta

### 2. Real-Based Feature Engineering
The core of the notebook translates raw categories into quantitative risk indices:
- **`University_Prestige_Score`**: Mapped using 2025 EduRank national rankings for Pakistani universities (scaled 50-100).
- **`Program_Employability_Score`**: Derived from reported starting salaries for fresh graduates by field.
- **`City_Income_Index`**: A geographic affluence index based on UN-Habitat city-level income and poverty patterns.
- **`EduFi_Alt_Risk_Index`**: A composite engineered metric calculating overall alternative risk via weighted averages: `(0.4 * Uni Risk) + (0.35 * Program Risk) + (0.25 * City Risk)`.

### 3. Alternative Satellite Data Integration
The pipeline introduces non-traditional macro-economic data by merging the borrower dataset with `City_NTL_Score_Pakistan_2023.csv`. This appends a **Night-Time Lights (NTL) Score** (`City_NTL_Score`) for each borrower's city, acting as a proxy for regional economic activity.

### 4. Exploratory Data Analysis (EDA)
Before exporting the final dataset for machine learning modeling, the pipeline runs initial checks:
- Visualizes the target distribution (`default`) to assess class imbalances.
- Generates a Seaborn Correlation Heatmap for all numeric features to identify multicollinearity between traditional metrics and the newly engineered alternative data.

## Tech Stack
- **Data Processing:** `pandas`, `numpy`
- **Data Visualization:** `matplotlib.pyplot`, `seaborn`
- **Environment:** Google Colab / Jupyter 

## Output
The pipeline exports a finalized, enriched matrix (`final_augmented_dataset.csv`) containing 22 columns, ready to be ingested by balancing algorithms (like SMOTETomek) and classification models (XGBoost, LightGBM).
