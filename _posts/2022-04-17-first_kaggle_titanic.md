---
layout: single
title:  "2019250058_추희정"
categories: coding
tag: [python, blog, jekyll]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


# 타이타닉 생존자 예측


## 데이터 불러오기



```python
# 기본설정
import pandas as pd
import numpy as np
```


```python
# 데이터 로드
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_titanic_data():
    tarball_path = Path("datasets/titanic.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/titanic.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as titanic_tarball:
            titanic_tarball.extractall(path="datasets")
    return [pd.read_csv(Path("datasets/titanic") / filename)
            for filename in ("train.csv", "test.csv")]
```


```python
train, test = load_titanic_data()
```

데이터는 이미 훈련 세트와 테스트 세트로 분할되어 있다. 그러나 테스트 데이터에는 레이블이 포함되어 있지 않다. 목표는 훈련 데이터를 사용하여 최상의 모델을 훈련한 다음 테스트 데이터에 대한 예측을 하고 Kaggle에 업로드하여 최종 점수를 확인하는 것이다.

<br><br>

훈련 세트의 상위 몇 행을 살펴보겠다.



```python
train.head()
```

<pre>
   PassengerId  Survived  Pclass  \
0            1         0       3   
1            2         1       1   
2            3         1       3   
3            4         1       1   
4            5         0       3   

                                                Name     Sex   Age  SibSp  \
0                            Braund, Mr. Owen Harris    male  22.0      1   
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
2                             Heikkinen, Miss. Laina  female  26.0      0   
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
4                           Allen, Mr. William Henry    male  35.0      0   

   Parch            Ticket     Fare Cabin Embarked  
0      0         A/5 21171   7.2500   NaN        S  
1      0          PC 17599  71.2833   C85        C  
2      0  STON/O2. 3101282   7.9250   NaN        S  
3      0            113803  53.1000  C123        S  
4      0            373450   8.0500   NaN        S  
</pre>
## 데이터 분석



```python
# 데이터값의 분포를 보기위한 라이브러리 불러오기
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
# categorical featrue의 분포를 보기위한 piechart를 만드는 함수
def pie_chart(feature):
    feature_ratio = train[feature].value_counts(sort=False)
    feature_size = feature_ratio.size
    feature_index = feature_ratio.index
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    
    
    plt.plot(aspect='auto')
    plt.pie(feature_ratio, labels=feature_index, autopct='%1.1f%%')
    plt.title(feature + '\'s ratio in total')
    plt.show()
    for i, index in enumerate(feature_index):
        plt.subplot(1, feature_size + 1, i + 1, aspect='equal')
        plt.pie([survived[index], dead[index]], labels=['Survivied', 'Dead'], autopct='%1.1f%%')
        plt.title(str(index) + '\'s ratio')
    plt.show()
```


```python
# 'sex'에 대한 pie chart
pie_chart('Sex')
```

<pre>
<Figure size 432x288 with 1 Axes>
</pre>
<pre>
<Figure size 432x288 with 2 Axes>
</pre>
남성이 여성보다 배에 많이 탔으며, 남성보다 여성의 비율이 더 높다



```python
# 사회경제적 지위인 pclass에 대해 그리기
pie_chart('Pclass')
```

<pre>
<Figure size 432x288 with 1 Axes>
</pre>
<pre>
<Figure size 432x288 with 3 Axes>
</pre>
위와 같이 Pclass가 3인 사람들의 수가 가장 많았으며, Pclass가 높을수록(숫자가 작을수록; 사회경제적 지위가 높을수록) 생존 비율이 높다는 것을 알 수 있다.



마지막으로 어느 곳에서 배를 탔는지를 나타내는 Embarked에 대해서 살펴보자.



```python
pie_chart('Embarked')
```

<pre>
<Figure size 432x288 with 1 Axes>
</pre>
<pre>
<Figure size 432x288 with 3 Axes>
</pre>
이번에는 아래의 특성들에 대해서 Bar chart를 정의해서 데이터를 시각화 해보자.



SibSp ( # of siblings and spouse) Parch ( # of parents and children)



```python
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
```


```python
# sibsp에 대한 barchat 그리기
bar_chart("SibSp")
```

<pre>
<Figure size 720x360 with 1 Axes>
</pre>
위와 같이 2명 이상의 형제나 배우자와 함께 배에 탔을 경우 생존한 사람의 비율이 컸다는 것을 볼 수 있고, 그렇지 않을 경우에는 생존한 사람의 비율이 적었다는 것을 볼 수 있다.



```python
# Parch에 대해서도 Bar chart를 그려보자.
bar_chart("Parch")
```

<pre>
<Figure size 720x360 with 1 Axes>
</pre>
## 데이터 전처리 및 특성 추출


데이터 전처리를 하는 과정에서는 train과 test 데이터를 같은 방법으로 한 번에 처리를 해야하므로 먼저 두 개의 데이터를 합쳐보도록하자.



```python
train.head(5)
```

<pre>
   PassengerId  Survived  Pclass  \
0            1         0       3   
1            2         1       1   
2            3         1       3   
3            4         1       1   
4            5         0       3   

                                                Name     Sex   Age  SibSp  \
0                            Braund, Mr. Owen Harris    male  22.0      1   
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
2                             Heikkinen, Miss. Laina  female  26.0      0   
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
4                           Allen, Mr. William Henry    male  35.0      0   

   Parch            Ticket     Fare Cabin Embarked  
0      0         A/5 21171   7.2500   NaN        S  
1      0          PC 17599  71.2833   C85        C  
2      0  STON/O2. 3101282   7.9250   NaN        S  
3      0            113803  53.1000  C123        S  
4      0            373450   8.0500   NaN        S  
</pre>

```python
train_and_test = [train, test]
```


```python
for dataset in train_and_test:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')

train.head(5)
```

<pre>
   PassengerId  Survived  Pclass  \
0            1         0       3   
1            2         1       1   
2            3         1       3   
3            4         1       1   
4            5         0       3   

                                                Name     Sex   Age  SibSp  \
0                            Braund, Mr. Owen Harris    male  22.0      1   
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
2                             Heikkinen, Miss. Laina  female  26.0      0   
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
4                           Allen, Mr. William Henry    male  35.0      0   

   Parch            Ticket     Fare Cabin Embarked Title  
0      0         A/5 21171   7.2500   NaN        S    Mr  
1      0          PC 17599  71.2833   C85        C   Mrs  
2      0  STON/O2. 3101282   7.9250   NaN        S  Miss  
3      0            113803  53.1000  C123        S   Mrs  
4      0            373450   8.0500   NaN        S    Mr  
</pre>

```python
# 추출한 Title을 가진 사람이 몇 명이 존재하는지 성별과 함께 표현을 해보자.
pd.crosstab(train['Title'], train['Sex'])
```

<pre>
Sex       female  male
Title                 
Capt           0     1
Col            0     2
Countess       1     0
Don            0     1
Dr             1     6
Jonkheer       0     1
Lady           1     0
Major          0     2
Master         0    40
Miss         182     0
Mlle           2     0
Mme            1     0
Mr             0   517
Mrs          125     0
Ms             1     0
Rev            0     6
Sir            0     1
</pre>

```python
# 흔하지 않은 Title은 Other로 대체하고 중복되는 표현을 통일하자.
for dataset in train_and_test:
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Countess', 'Don','Dona', 'Dr', 'Jonkheer',
                                                 'Lady','Major', 'Rev', 'Sir'], 'Other')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
```

<pre>
    Title  Survived
0  Master  0.575000
1    Miss  0.702703
2      Mr  0.156673
3     Mrs  0.793651
4   Other  0.347826
</pre>

```python
# 추출한 Title 데이터를 학습하기 알맞게 String Data로 변형
for dataset in train_and_test:
    dataset['Title'] = dataset['Title'].astype(str)
```

승객의 성별을 나타내는 Sex Feature를 처리할 것인데 이미 male과 female로 나뉘어져 있으므로 String Data로만 변형해주면 된다.



```python
for dataset in train_and_test:
    dataset['Sex'] = dataset['Sex'].astype(str)
```

데이터 정보에 따르면 train 데이터에서 Embarked feature에는 NaN 값이 존재하며, 다음을 보면 잘 알 수 있다.



```python
train.Embarked.unique()
```

<pre>
array(['S', 'C', 'Q', nan], dtype=object)
</pre>

```python
train.Embarked.value_counts(dropna=False)
```

<pre>
S      644
C      168
Q       77
NaN      2
Name: Embarked, dtype: int64
</pre>

```python
for dataset in train_and_test:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
```

<pre>
  Embarked  Survived
0        C  0.553571
1        Q  0.389610
2        S  0.339009
</pre>

```python
for dataset in train_and_test:
    dataset['Embarked'] = dataset['Embarked'].astype(str)
```

pd.cut()을 이용해 같은 길이의 구간을 가지는 다섯 개의 그룹을 만들어 보자.



```python
for dataset in train_and_test:
    dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
    dataset['Age'] = dataset['Age'].astype(int)
    train['AgeBand'] = pd.cut(train['Age'], 5)
print (train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean()) # Survivied ratio about Age Band
```

<pre>
         AgeBand  Survived
0  (-0.08, 16.0]  0.550000
1   (16.0, 32.0]  0.344762
2   (32.0, 48.0]  0.403226
3   (48.0, 64.0]  0.434783
4   (64.0, 80.0]  0.090909
</pre>

```python
train.head()
```

<pre>
   PassengerId  Survived  Pclass  \
0            1         0       3   
1            2         1       1   
2            3         1       3   
3            4         1       1   
4            5         0       3   

                                                Name     Sex  Age  SibSp  \
0                            Braund, Mr. Owen Harris    male   22      1   
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female   38      1   
2                             Heikkinen, Miss. Laina  female   26      0   
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female   35      1   
4                           Allen, Mr. William Henry    male   35      0   

   Parch            Ticket     Fare Cabin Embarked Title       AgeBand  
0      0         A/5 21171   7.2500   NaN        S    Mr  (16.0, 32.0]  
1      0          PC 17599  71.2833   C85        C   Mrs  (32.0, 48.0]  
2      0  STON/O2. 3101282   7.9250   NaN        S  Miss  (16.0, 32.0]  
3      0            113803  53.1000  C123        S   Mrs  (32.0, 48.0]  
4      0            373450   8.0500   NaN        S    Mr  (32.0, 48.0]  
</pre>

```python
# Age에 들어 있는 값을 위에서 구한 구간에 속하도록 바꾸기
for dataset in train_and_test:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
    dataset['Age'] = dataset['Age'].map( { 0: 'Child',  1: 'Young', 2: 'Middle', 3: 'Prime', 4: 'Old'} ).astype(str)
```

Test 데이터 중에서 Fare Feature에도 NaN 값이 하나 존재하는데, Pclass와 Fare가 어느 정도 연관성이 있는 것 같아 Fare 데이터가 빠진 값의 Pclass를 가진 사람들의 평균 Fare를 넣어주는 식으로 처리를 해보자.



```python
for dataset in train_and_test:
    dataset['Fare'] = dataset['Fare'].fillna(13.675) # The only one empty fare data's pclass is 3.
```


```python
train['FareBand'] = pd.qcut(train['Fare'], 5)
print (train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())
```

<pre>
            FareBand  Survived
0    (-0.001, 7.854]  0.217877
1      (7.854, 10.5]  0.201087
2     (10.5, 21.679]  0.424419
3   (21.679, 39.688]  0.444444
4  (39.688, 512.329]  0.642045
</pre>

```python
for dataset in train_and_test:
    dataset.loc[ dataset['Fare'] <= 7.854, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 10.5), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.679), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 39.688), 'Fare']   = 3
    dataset.loc[ dataset['Fare'] > 39.688, 'Fare'] = 4
    dataset['Fare'] = dataset['Fare'].astype(int)
```

형제, 자매, 배우자, 부모님, 자녀의 수가 많을 수록 생존한 경우가 많았는데, 두 개의 Feature를 합쳐서 Family라는 Feature로 만들자.



```python
for dataset in train_and_test:
    dataset["Family"] = dataset["Parch"] + dataset["SibSp"]
    dataset['Family'] = dataset['Family'].astype(int)
```

특성 추출 및 나머지 전처리



```python
features_drop = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId', 'AgeBand', 'FareBand'], axis=1)

print(train.head())
print(test.head())
```

<pre>
   Survived  Pclass     Sex     Age  Fare Embarked Title  Family
0         0       3    male   Young     0        S    Mr       1
1         1       1  female  Middle     4        C   Mrs       1
2         1       3  female   Young     1        S  Miss       0
3         1       1  female  Middle     4        S   Mrs       1
4         0       3    male  Middle     1        S    Mr       0
   PassengerId  Pclass     Sex     Age  Fare Embarked Title  Family
0          892       3    male  Middle     0        Q    Mr       0
1          893       3  female  Middle     0        S   Mrs       1
2          894       2    male   Prime     1        Q    Mr       0
3          895       3    male   Young     1        S    Mr       0
4          896       3  female   Young     2        S   Mrs       2
</pre>

```python
# One-hot-encoding for categorical variables
train = pd.get_dummies(train)
test = pd.get_dummies(test)

train_label = train['Survived']
train_data = train.drop('Survived', axis=1)
test_data = test.drop("PassengerId", axis=1).copy()

print(train_data.shape, train_label.shape, test_data.shape)
```

<pre>
(891, 18) (891,) (418, 18)
</pre>
## 모델 설계 및 학습


사용할 예측 모델은 다음과 같이 5가지가 있다.



Logistic Regression Support Vector Machine (SVM) k-Nearest Neighbor (kNN) Random Forest Naive Bayes



```python
# scikit-learn 라이브러리를 불러오자.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.utils import shuffle
```


```python
# 학습시키기 전에는 주어진 데이터가 정렬되어있어 학습에 방해가 될 수도 있으므로 섞어주도록 하자.
train_data, train_label = shuffle(train_data, train_label, random_state = 5)
```


```python
def train_and_test(model):
    model.fit(train_data, train_label)
    prediction = model.predict(test_data)
    accuracy = round(model.score(train_data, train_label) * 100, 2)
    print("Accuracy : ", accuracy, "%")
    return prediction
```

이 함수에 다섯가지 모델을 넣어주면 학습과 평가가 완료된다.



```python
# Logistic Regression
log_pred = train_and_test(LogisticRegression())
# SVM
svm_pred = train_and_test(SVC())
#kNN
knn_pred_4 = train_and_test(KNeighborsClassifier(n_neighbors = 4))
# Random Forest
rf_pred = train_and_test(RandomForestClassifier(n_estimators=100))
# Navie Bayes
nb_pred = train_and_test(GaussianNB())
```

<pre>
Accuracy :  82.72 %
Accuracy :  83.5 %
Accuracy :  84.51 %
Accuracy :  88.55 %
Accuracy :  79.8 %
</pre>
## 마무리


위에서 볼 수 있듯 4번째 모델인 Random Forest에서 가장 높은 정확도(88.55%)를 보였는데, 이 모델을 채택해서 submission 해보자.



```python
# 아래 코드는 kaggle에 제출 할 내용임.
# submission = pd.DataFrame({
# ​    "PassengerId": test["PassengerId"],
# ​    "Survived": rf_pred
# })

# submission.to_csv('submission_rf.csv', index=False)
```

이 파일을 kaggle에 업로드 하면 다음과 같은 결과가 나온다.





![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABT8AAAEECAYAAADuwkBHAAAgAElEQVR4nOzdd1STZxsG8CuEJRsZDlyouAUcgLhH3Rttq3XUuveuW2vds466FfeedVQ/Z1VQBDeuKiouXIDMACEJ+f6gvBITIEEClF6/c3qOvPN5Q6AnF/fz3CKlUqkEERERERERERERUQFjkNcDICIiIiIiIiIiItIHhp9ERERERERERERUIDH8JCIiIiIiIiIiogKJ4ScREREREREREREVSAw/iYiIiIiIiIiIqEBi+ElEREREREREREQFEsNPIiIiIiIiIiIiKpAYfhIREREREREREVGBxPCTiIiIiIiIiIiICiSGn0RERERERERERFQgMfwkIiIiIiIiIiKiAonhJxERERERERERERVIDD+JiIiIiIiIiIioQGL4SURERERERERERAUSw08iIiIiIiIiIiIqkBh+EhERERERERERUYHE8JOIiIiIiIiIiIgKJIafREREREREREREVCAx/CQiIiIiIiIiIqICieEnERERERERERERFUgMP4mIiIiIiIiIiKhAYvhJREREREREREREBRLDTyIiIiIiIiIiIiqQGH4SERERERERERFRgcTwk4iIiIiIiIiIiAokhp9ERERERERERERUIDH8JCIiIiIiIiIiogKJ4ScREREREREREREVSAw/iYiIiIiIiIiIqEBi+ElEREREREREREQFEsNPIiIiIiIiIiIiKpAYfhIREREREREREVGBZKjvG0RFRaFvvwG4c+euynZ3dzds9t0IAOjbbwAAYLPvRtja2mp9jS+lXVPTNbIr7d4D+vfDxk2+mY4zqzHrY3xERERERERERESkmd7DT1tbWxw5fDDD/VFRUV99jbTr9O03AH37DdA5YMwsoF2yeJHW19F1zERERERERERERKQ/eg0/s6rY/JpwMSelBZVp4y1dqhTmz5+LQoUKaRXOptG2QhVgFSgREREREREREZG+6TX8TF/9mBYMAqrTxnUJF/O7L6s904ehDDuJiIiIiIiIiIhyFxse5aCoqCh09ukK57IucC7rgpq1PAEA586eBgDUrOUp7Ovs07VABb9ERERERERERET5jd7X/ATUmwZlZ11Oba4PaNeMKCsvXr5E8xatERYWptN5ma3zyfU/iYiIiIiIiIiIcleudXsHAG/vOvD2rqPSmEjb87VZRxNIra7MzhTz9Pfp2KE99uzeKaz5mTb+nBwnwHU/iYiIiIiIiIiI9Elv4Wdm611u9t2Ivv0GoGYtT6xetTLT6+R21/TVq1Zi4yZf/NCjl1bhbJqMxpnTValERERERERERESkHb2Fn5qa/3T26QogNQRM3whp4yZffQ0jV2VV/Zm2BigrPomIiIiIiIiIiPTvX9HwKC041VeToC+n5qdVfPbtNwCfPml/v7TAN/R5SIb/3boZJFybDY+IiIiIiIiIiIj0J1caHv2XaLP2Jys/iYiIiIiIiIiI9C9X1vz8Utr0bwBZrvmpb5rW5MxuRWb6a926GcRwk4iIiIiIiIiIKA/l2pqfGdFmzU9bW1uVJkmZ0bWqUtM402/TJQjV5ziJiIiIiIiIiIhINyKlUqnM60EQERERERERERER5bR/RcMjIiIiIiIiIiIiIl0x/CQiIiIiIiIiIqICieEnERERERERERERFUgMP4mIiIiIiIiIiKhAYvhJREREREREREREBRLDTyIiIiIiIiIiIiqQGH4SERERERERERFRgcTwk4iIiIiIiIiIiAokhp9ERERERERERERUIDH8JCIiIiIiIiIiogKJ4ScREREREREREREVSAw/iYiIiIiIiIiIqEBi+ElEREREREREREQFEsNPIiIiIiIiIiIiKpAYfhIREREREREREVGBxPCTiIiIiIiIiIiICiSGn0RERERERERERFQgMfwkIiIiIiIiIiKiAonhJxERERERERERERVIDD+JiIiIiIiIiIioQGL4SURERERERERERAUSw08iIiIiIiIiIiIqkBh+EhERERERERERUYHE8JOIiIiIiIiIiIgKJIafREREREREREREVCAx/CQiIiIiIiIiIqICieEnERERERERERERFUgMP4mIiIiIiIiIiKhAYvhJREREREREREREBRLDTyIiIiIiIiIiIiqQGH4SERERERERERFRgcTwk4iIiIiIiIiIiAokhp9ERERERERERERUIDH8JCIiIiIiIiIiogKJ4ScREREREREREels4aLFcC7rAueyLggMDMrr4RBpZJjXA8gJMoUMTz+9xJXXtxD0JhgPI54iQhKFRHmS2rGFDE1hb26LKvbl4VnCFfVK1kT5wqVhJDbKg5ETEREREREREeWNp0+foc9P/RAWFqbV8e7ubtjsuxG2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vndP+8UOxhPiq6oq09mxtdhjAAAAAP2sYPGzq7s7FRUVhbr8K+rt7c2Pf3xn7rjjzlx11f/IpEmT+vy1PT09+fa3v5O1Tz+dr/3DxRk5cmQBJy2sVzpkafHih/KBD/5dn69RiAOP7rr7nj7Fs309HGfIkCGZPXt2LvvG13PZN76+r2O+TE9PT279l9uyZs2aJNv30/zhP//fzJ0zpyCHN/WHkSNHZtSo2tTX12fdunUve72zszNtbW0ZPXp0EaYbfCoqKtLVvf97xQIAAACDS8HiZ09PT8qGDi3U5V/Rbx98MF/80lfS2dmZz13w97nmW9/sUwDt6enJdf/z+lx33fVJkokTJuSCC87f5xOwX8lgPu2a3fuPxYtz0823pLy8PF+48PO5/fY78q8//Vnectxbctp7F+zTY/Q77E2Q3ptH5GtqRqauri5Lly7LypWr0tvb+5I5ly5dliSZNHFixo8fv8frDR8+PNde+61ce+23+jzvgaRs6ND09PQUewwAAACgnxUsfhbDm974xnzta1/Nl798cRYvfqhPAfTPw+d73nNKPvnJT/Rr+DxQvFLgevHjzrs65Xxvr9mfNm/enDPPOjt//OMf9+rrrrji8j697+GHH8nChV9MU1NTPvvZ83LWWWemrq4u//38C/KNb1yeCYcemuOPP26/AmghVFdXZ/78ebn33vvy77/5TU4//UM7/ztobGzKfffdn2R7lK+pOXBXOQMAAADsTsGWZpaVlaVngE/EHjp0aN532ntz6aWXpLy8fGcAXb9+/Su+/5XC59cv+YcD5pH3u+6+J3OPnJdp02e94sdxx78tK1euKvaYBVVVVZWJEyfs8vXKysoceeTczJgx/SWfb2tt2+11t23bljvvujtnn3Nu6uvrs2DBqfnEOWenrKws73znX+bii7+ctra2nPOJT+amm2/J1q193y9y5swZeWDRr7Nm9ZM7P37/u4dy9NHzkyTnnnvOS17b8bG3ByOddNLfZPasWXniiaW58qqr09jYlC1b2nPDd27Mbx98MBMmHJpTTnn3Pof+lpaW3HHHj3PWWWfnhLe+feff3azZc/JX7zopCxdelMWLHzogVlT2bNt2UP4PDwAAACh1BVv5WVFenq6urpRVDuyhRzsCaJKXrQB9sc7Orbn2H6/L9dffkOTAC59sV11dne999zsv+/yOg6bmzHn1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)


**참고** : <a href="https://www.kaggle.com/code/chuchoo/notebook-titanic/edit"
   title="chuchoo/notebook-titanic">내가 제출한 캐글주피터 노트북 주소</a>




