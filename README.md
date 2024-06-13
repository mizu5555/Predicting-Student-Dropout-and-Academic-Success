# Predicting-Student-Dropout-and-Academic-Success
Final Project of Statistical Methods and Data Mining in 2024 Spring.

## 資料介紹
### Dataset Description
- **Title**: [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- **UCI Donated on**: 12/12/2021
- **Authors**: Valentim Realinho, Mónica V. Martins, Jorge Machado, Luís Baptista
- **Records**: 4424
- **Attributes**: 36+1
- **Missing Values**: None
- **Description**: 每筆記錄代表一名學生，資料來源涵蓋了從2008/2009學年（波隆那進程應用於歐洲高等教育之後）到2018/2019學年的數據，涉及來自不同知識領域的17個本科學位。

## 環境設置：
```
Python: 3.10.6
pandas: 1.5.3
numpy: 1.24.3
scikit-learn: 1.4.2
matplotlib: 3.7.1
seaborn: 0.12.2
conda: 23.7.4
jupyter: 1.0.0
scipy: 1.13.1
```
## EDA

## 統計方法與假設

### 卡方的獨立性檢定(Chi-squared test)
檢查每個類別特徵（如母親的學歷、父親的職業等）與目標變量（Target）之間是否
存在獨立性，即這些特徵與學生的輟學或畢業狀態之間是否有關聯。
#### 假設
- **虛無假設(H0)**: 類別特徵與目標變量之間是獨立的(沒有相關性)
- **對立假設(H1)**: 類別特徵與目標變量之間不是獨立的(有相關性)
- **顯著水準(significant level)**: 0.05
#### 結果
- **Reject H0** (categorical features are significantly related to the target variable):
  - Mother's qualification
  - Father's qualification
  - Mother's occupation
  - Father's occupation
  - Marital Status
  - Application mode
  - Course
  - Previous qualification
  - Daytime/evening attendance
  - Displaced
  - Gender
  - Debtor
  - Tuition fees up to date
  - Scholarship holder

- **Fail to Reject H0** (no significant evidence to suggest a relationship between categorical features and the target variable):
  - Nationality
  - International
  - Educational special needs

### 皮爾森相關係數 (Pearson Correlation)
我們為了要比較各種數值型資料與目標變量（Target）的關係，
所以利用皮爾森相關係數來衡量兩個變數之間線性關係強度和方向，
其值範圍從 -1 到 1，其中：
- 1 表示完全正相關
- -1 表示完全負相關
- 0 表示無線性關係

![Pearson](https://raw.githubusercontent.com/mizu5555/Predicting-Student-Dropout-and-Academic-Success/d99722f785615ffe1c90e70be475841808f82720/Pearson.png)

## 資料探勘

###  任務描述
- **Task**: 預測學生的輟學和學業成功
- **Input**:
  - 學生入學資料
  - 學生的學業表現
- **Output**: 
  - 學生是否輟學或成功

### 假設
1. **Complete dataset**: 包含37個屬性
2. **Reject H0 dataset**: 包含21個屬性

### 設定
- **5-fold cross validation**
- **數據分割**: 80%訓練數據, 20%測試數據

### 模型結果

### 支持向量機 (SVM) 結果
※ kernel = rbf
|            | First   | Second  | Third   | Fourth  | Fifth   | Mean    |
|------------|---------|---------|---------|---------|---------|---------|
| Complete   | 91.04 % | 90.63 % | 91.73 % | 91.59 % | 91.04 % | 91.21 % |
| Reject H0  | 90.08 % | 89.53 % | 91.04 % | 91.18 % | 90.90 % | 90.55 % |

### K最近鄰 (KNN) 結果
※ n_neighbors = 3
|            | First   | Second  | Third   | Fourth  | Fifth   | Mean    |
|------------|---------|---------|---------|---------|---------|---------|
| Complete   | 84.29 % | 85.53 % | 84.84 % | 85.12 % | 85.67 % | 85.09 % |
| Reject H0  | 85.81 % | 84.84 % | 86.50 % | 87.46 % | 87.46 % | 86.41 % |

### 決策樹 (Decision Tree) 結果
|            | First   | Second  | Third   | Fourth  | Fifth   | Mean    |
|------------|---------|---------|---------|---------|---------|---------|
| Complete   | 83.74 % | 83.74 % | 87.87 % | 86.36 % | 85.81 % | 85.50 % |
| Reject H0  | 85.67 % | 82.78 % | 84.43 % | 84.84 % | 83.88 % | 84.32 % |

### Citation
Realinho, Valentim, Vieira Martins, Mónica, Machado, Jorge, and Baptista, Luís. (2021). Predict Students' Dropout and Academic Success. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
