# Gen Z Career Insights - ML Analysis

Comprehensive analysis of Gen Z career aspirations with machine learning models for prediction and insights.

## What this project does

- **Exploratory Data Analysis**: Country distribution, education levels, work preferences, salary importance
- **Career Satisfaction Analysis**: Trends by education level and demographics
- **ML Model 1**: Predict who will pursue higher education (Random Forest)
- **ML Model 2**: Predict job loyalty - who stays 3+ years (Gradient Boosting)
- **ML Model 3**: Predict career satisfaction levels (Logistic Regression)

## Dataset

- File: `data/gen_z_career.csv`
- Records: 5,000 Gen Z professionals
- Columns: 11 (Age, Country, Education, Work Environment, Career Satisfaction, etc.)

## Project structure

- `notebooks/gen_z_career_analysis.ipynb`: Main analysis & ML models
- `data/gen_z_career.csv`: Dataset file
- `requirements.txt`: Python dependencies

## Run locally

1. Create and activate virtual environment
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebook in Jupyter and run cells interactively:

```bash
jupyter notebook notebooks/gen_z_career_analysis.ipynb
```

## Key outputs (latest run)

- **Dataset**: 5,000 records, 11 features
- **Model 1 (Higher Education Prediction)**: Accuracy=58.80%, ROC-AUC=49.71%
- **Model 2 (Job Loyalty Prediction)**: Accuracy=52.00%, ROC-AUC=50.29%
- **Model 3 (Career Satisfaction)**: Accuracy=21.10%

## Technologies

- Python 3.13
- pandas, numpy: data manipulation
- scikit-learn: ML models (RandomForest, GradientBoosting, LogisticRegression)
- matplotlib, plotly, seaborn: visualization
- jupyter: interactive notebooks

## Features

- High-end ML models with multiple algorithms
- Simple, readable variable names (info, q1-q5, graph_1-5, model_1-3)
- Clean data pipeline
- Comprehensive EDA visualizations

## Upcoming Features

- **Power BI Dashboard**: Interactive Power BI dashboard for Gen Z career insights coming soon!
