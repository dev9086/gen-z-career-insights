import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Load data
info = pd.read_csv('data/gen_z_career.csv')
print(f"Gen Z project: Loaded {info.shape[0]} rows x {info.shape[1]} cols")

# Prepare for ML
ml_info = info.copy()
for col in ml_info.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    ml_info[col] = le.fit_transform(ml_info[col])

# Model 1: Higher Education
if 'PursueHigherEd' in info.columns:
    target_1 = info['PursueHigherEd'].astype(int)
    features_1 = ml_info.drop('PursueHigherEd', axis=1)
    X1_train, X1_test, y1_train, y1_test = train_test_split(features_1, target_1, test_size=0.2, random_state=42, stratify=target_1)
    model_1 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model_1.fit(X1_train, y1_train)
    acc_1 = accuracy_score(y1_test, model_1.predict(X1_test))
    try:
        roc_1 = roc_auc_score(y1_test, model_1.predict_proba(X1_test)[:, 1])
    except Exception:
        roc_1 = float('nan')
    print(f"Model 1 - HigherEd accuracy: {acc_1:.4f}, ROC-AUC: {roc_1:.4f}")

# Model 2: Job Loyalty
if 'StayWithEmployer3Years' in info.columns:
    target_2 = info['StayWithEmployer3Years'].astype(int)
    features_2 = ml_info.drop('StayWithEmployer3Years', axis=1)
    X2_train, X2_test, y2_train, y2_test = train_test_split(features_2, target_2, test_size=0.2, random_state=42, stratify=target_2)
    model_2 = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model_2.fit(X2_train, y2_train)
    acc_2 = accuracy_score(y2_test, model_2.predict(X2_test))
    try:
        roc_2 = roc_auc_score(y2_test, model_2.predict_proba(X2_test)[:, 1])
    except Exception:
        roc_2 = float('nan')
    print(f"Model 2 - Job Loyalty accuracy: {acc_2:.4f}, ROC-AUC: {roc_2:.4f}")

# Model 3: Career Satisfaction
if 'CareerSatisfaction' in info.columns:
    target_3 = info['CareerSatisfaction'].astype(int)
    features_3 = ml_info.drop('CareerSatisfaction', axis=1)
    scaler = StandardScaler()
    features_3_scaled = scaler.fit_transform(features_3)
    X3_train, X3_test, y3_train, y3_test = train_test_split(features_3_scaled, target_3, test_size=0.2, random_state=42, stratify=target_3)
    model_3 = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    model_3.fit(X3_train, y3_train)
    acc_3 = accuracy_score(y3_test, model_3.predict(X3_test))
    print(f"Model 3 - Career Satisfaction accuracy: {acc_3:.4f}")

print('\nAll models executed.')
