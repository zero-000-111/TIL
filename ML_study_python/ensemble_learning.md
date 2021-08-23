# Ensemble Learning
### 여러 개의 분류기(Classifier)를 생성하고 그 예측을 결합함으로써 보다 정확한 최종 예측을 도출하는 기법

## 앙상블 학습의 유형
1. Voting(보팅)
    - 여러 개의 분류기가 투표를 통해 최종 예측 결과를 결정하는 방식
    - 일반적으로 서로 다른 알고리즘을 가진 분류기를 결합
2. Bagging(배깅)
    - 보팅과 유사하게 여러 개의 분류기가 투표를 통해 최종 예측 결과를 결정
    - 각각의 분류기가 모두 같은 유형의 알고리즘 기반이지만, 데이터 셈플링을 서로 다르게 가져가면서 학습을 수행함
    - 부트스트래핑(Bootstrapping) : 개별 Classifier에게 데이터를 샘플링해서 추출하는 방식 사용
    - 예) 랜덤 포레스트 알고리즘
3. Boosting(부스팅)
    - 여러 개의 분류기가 순차적으로 학습을 수행하되, 앞으로 학습한 분류기가 예측이 틀린 데이터에 대해서는 올바르게 예측할 수 있도록 다음 분류기에게는 가중치를 부여하면서 학습과 예측을 진행하는 것
    - 위처럼 계속해서 분류기에게 가중치를 부스팅함
    - 예) 그래디언트 부스트, XGBoost(eXtra Gradient Boost), LightGBM(Light Gradient Boost)
4. 스태킹
    - 여러가지 다른 모델의 예측 결과값을 다시 학습데이터로 만들어서 다른 모델로 재학습시켜 결과를 예측하는 방법

## 보팅 유형
1. 하드 보팅(Hard Voting)
    - 다수결 원칙: 다수의 분류기가 결정한 예측값을 최종 보팅 결괏값으로 선정
2. 소프트 보팅(Soft Voting)
    - 분류기들의 레이블 값 결정 확률을 모두 더하고 이를 평균해서 이들 중 확률이 가장 높은 레이블 값을 최종 보팅 결괏값으로 선정
    - 일반적으로 많이 적용되는 방법
## 보팅 분류기(Voting Classifier)
- 보팅 방식 앙상블(VotingClassifier)을 통해 위스콘신 유방암 데이터 세트 분석하기
```
# 데이터 로드
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cancer=load_breast_cancer()

data_df = pd.DataFrame(cancer.data,columns=cancer.feature_names)
```
```
# 로지스틱 회귀와 KNN을 기반으로 하여 소프트 보팅 방식으로 보팅 분류기 생성하기
lr_clf=LogisticRegression()
knn_clf=KneighborsClassifier(n_neighbors=8)

#개별 모델을 소프트 보팅 기반의 앙상블 모델로 구현한 분류기
vo_clf=VotingClassifier(estimators=[('LR',lr_clf),('KNN',knn_clf)],voting='soft')
## estimators: 리스트 값으로 보팅에 사용도리 여러 개의 Classifier 객체들을 튜플 형식으로 입력받음
## voting: 'hard'(default)/'soft'

#VotingClassifier 학습/예측/평가
vo_clf.fit(X_train,y_train)
pred=vo_clf.predict(X_test)
```