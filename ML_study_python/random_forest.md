# Random Forest
### 여러 개의 결정 트리 분류기가 전체 전체 데이터에서 배깅 방식으로 각자의 데이터를 샘플링해 개별적으로 학습을 수행한 뒤 최종적으로 모든 분류기가 보팅을 통해 예측 결정
- 배깅의 대표적인 알고리즘
- 앙상블 알고리즘 중 비교적 빠른 수행 속도
- 다양한 영역에서 높은 예측 성능
### 부트스트래핑 
- 여러 개의 데이터 세트를 중첩되게 분리하는 것 (Bagging: Bootstrap aggregating)
- 통계학: 여러 개의 작은 데이터 세트를 임의로 만들어 개별 평균의 분포도를 측정하는 등의 목적을 위한 샘플링 방식
### RandomForestClassifier
- n_estimators: 랜덤 포레스트에서 결정 트리의 개수를 지정합니다. 많이 설정할수록 좋은 성능을 기대할 수 있지만 확정적이지 않을 뿐만 아니라 학습 수행 시간 증가, default=10
- max_features: 참조할 피처 개수, default='sqrt'(전체 피처의 제곱근)
```
# 사용자 행동 인식 데이터 세트 예측
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 결정 트리에서 사용한 get_human_dataset()를 이용해 학습/테스트용 DataFrame 반환
X_train,X_test,y_train,y_test=get_human_dataset()

# 랜덤 포레스트 학습 및 별도의 테스트 세트로 예측 성능 평가
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train,y_train)
pred=rf_clf.predict(X_test)
accuracy = accuracy_score(y_test,pred)
print('랜덤 포레스트 정확도:{0:.4f}'.format(accuracy))
```