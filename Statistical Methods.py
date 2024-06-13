import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 設置非交互式後端
import matplotlib
matplotlib.use('Agg')

# 加載原始數據
from ucimlrepo import fetch_ucirepo

# 獲取資料集
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)

# 數據（作為 pandas DataFrame）
X = predict_students_dropout_and_academic_success.data.features
y = predict_students_dropout_and_academic_success.data.targets

# 刪除 "在學" 的資料
data = pd.concat([X, y], axis=1)
data = data[data['Target'] != 'enrolled']
X = data.drop(columns=['Target'])
y = data['Target'].astype('category').cat.codes

# 變數名稱
categorical_features = [
    'Mother\'s qualification', 'Father\'s qualification', 'Mother\'s occupation', 'Father\'s occupation',
    'Marital Status', 'Application mode', 'Course', 'Daytime/evening attendance', 'Nacionality', 'Displaced',
    'Previous qualification', 'Gender', 'International', 'Educational special needs',
    'Debtor', 'Tuition fees up to date', 'Scholarship holder'
]

continuous_features = [
    'Unemployment rate', 'Inflation rate', 'GDP', 'Previous qualification (grade)', 'Admission grade',
    'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)'
]

count_features = [
    'Age at enrollment', 'Application order', 'Curricular units 1st sem (credited)', 
    'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)', 
    'Curricular units 1st sem (approved)', 'Curricular units 1st sem (without evaluations)', 
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)', 
    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)', 
    'Curricular units 2nd sem (without evaluations)'
]

alpha = 0.05  # 顯著性水平

# 標準化連續型資料
scaler = StandardScaler()
X[continuous_features] = scaler.fit_transform(X[continuous_features])

# 計算Cramér's V值
def cramers_v(confusion_matrix):
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k-1, r-1))

cramers_v_values = []
chi2_values = []
p_values = []
valid_categorical_features = []

# 對類別型特徵進行假設檢定並計算Cramér's V值
for feature in categorical_features:
    if feature in X.columns:
        contingency_table = pd.crosstab(X[feature], y)
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        
        cv = cramers_v(contingency_table)
        cramers_v_values.append(cv)
        chi2_values.append(chi2)
        p_values.append(p)
        valid_categorical_features.append(feature)
        
        print(f'類別型特徵: {feature}')
        print(f'Chi-square statistic: {chi2}, P-value: {p}, Cramér\'s V: {cv}')
        if p < alpha:
            print(f'拒絕 H0: {feature} 與 Target 之間不是獨立的')
        else:
            print(f'沒有證據拒絕 H0: {feature} 與 Target 之間是獨立的')
        print('\n')
    else:
        print(f'特徵 {feature} 不存在於數據集中')

# 繪製Cramér's V值柱狀圖
plt.figure(figsize=(10, 6))
sns.barplot(x=cramers_v_values, y=valid_categorical_features, hue=valid_categorical_features, palette='coolwarm', dodge=False, legend=False)
plt.title('Cramér\'s V Values for Categorical Features')
plt.xlabel('Cramér\'s V Value')
plt.ylabel('Features')
plt.yticks(fontsize=8)  # 調整字體大小
plt.tight_layout()
plt.savefig("cramers_v_values.png")

# 繪製Chi-square值柱狀圖
plt.figure(figsize=(10, 6))
sns.barplot(x=chi2_values, y=valid_categorical_features, hue=valid_categorical_features, palette='coolwarm', dodge=False, legend=False)
plt.title('Chi-square Values for Categorical Features')
plt.xlabel('Chi-square Value')
plt.ylabel('Features')
plt.yticks(fontsize=8)  # 調整字體大小
plt.tight_layout()
plt.savefig("chi_square_values.png")

# 繪製P值柱狀圖
plt.figure(figsize=(10, 6))
sns.barplot(x=p_values, y=valid_categorical_features, hue=valid_categorical_features, palette='coolwarm', dodge=False, legend=False)
plt.axvline(x=alpha, color='red', linestyle='--')
plt.title('P-values for Categorical Features')
plt.xlabel('P-value')
plt.ylabel('Features')
plt.yticks(fontsize=8)  # 調整字體大小
plt.tight_layout()
plt.savefig("p_values.png")

# 計算連續變數與 Target 的 Pearson 相關係數
pearson_corr_continuous = []
for feature in continuous_features:
    corr, p_value = stats.pearsonr(X[feature], y)
    pearson_corr_continuous.append((feature, corr, p_value))

# 計算計數變數與 Target 的 Pearson 相關係數
pearson_corr_count = []
for feature in count_features:
    corr, p_value = stats.pearsonr(X[feature], y)
    pearson_corr_count.append((feature, corr, p_value))

# 打印連續變數與 Target 的 Pearson 相關係數
print("連續變數與 Target 的 Pearson 相關係數:")
for feature, corr, p_value in pearson_corr_continuous:
    print(f"{feature}: Pearson correlation = {corr}, P-value = {p_value}")

# 打印計數變數與 Target 的 Pearson 相關係數
print("計數變數與 Target 的 Pearson 相關係數:")
for feature, corr, p_value in pearson_corr_count:
    print(f"{feature}: Pearson correlation = {corr}, P-value = {p_value}")

# 繪製連續變數和計數變數與 Target 的 Pearson 相關係數柱狀圖
continuous_corrs = [x[1] for x in pearson_corr_continuous]
count_corrs = [x[1] for x in pearson_corr_count]
features = [x[0] for x in pearson_corr_continuous] + [x[0] for x in pearson_corr_count]
correlations = continuous_corrs + count_corrs

plt.figure(figsize=(12, 8))
sns.barplot(x=correlations, y=features, palette='coolwarm')
plt.title('Pearson Correlation with Target')
plt.xlabel('Pearson Correlation Coefficient')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig("pearson_corr_with_target.png")



