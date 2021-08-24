# GBM(Gradient Boosting Machine)
### 부스팅 알고리즘: 여러 개의 약한 학습기를 순차적으로 학습/예측하면서 잘못 예측한 데이터에 가중치 부여를 통해 오류를 개선해 나가면서 학습하는 방식
- GBM: 반복 수행을 통해 오류값을 최소화할 수 있도록 가중치의 업데이트 값을 도출하는 기법 (경사 하강법 사용)
## GradientBoostingClassifier
```
gb_clf=GradientBoostingClassifier()
gb_clf.fit(X_train,y_train)
pred=gb_clf.predict(X_test)
```
- 일반적으로 GBM이 랜덤 포레스트보다는 예측 성능이 뛰어난 경우가 많음
- 그러나 수행 시간이 오래 걸리고, 하이퍼 파라미터 튜닝 노력도 더 필요하다
- 예측 오류 보정을 통해 학습을 수행하기 때문에 병렬 처리 지원이 안돼 대용량 데이터의 경우 학습 시간이 길다
### 하이퍼 파라미터
- loss: 경사 하강법에서 사용할 비용함수 지엉, default='deviance'
- learning_rate: GBM이 학습을 진행할 때마다 적용하는 학습률, weak learner가 순차적으로 오류 값을 보정해 나가는 데 적용하는 계수, n_estimators와 상호 보완적으로 조합해 사용 (learning_rate를 작게하고 n_estimator를 크게 하면 한계점까지 예측 성능 개선 가능, but 수행시간 유의)
- n_estimators: weak learner의 개수, 순차적으로 오류를 수정하기 때문에 많을 수록 예측 성능이 일정 수준까지 개선 가능(but, 수행시간 유의), default=100
- subsample: weak learner가 학습에 사용하는 데이터 샘플리의 비율, default=1(전체 데이터 사용), 과적합 우려시 낮추는 것 고려
## XGBoost(eXtra Gradient Boost)
- 뛰어난 예측 성능: 일반적으로 분류와 회귀 영역에서 뛰어난 예측 성능 발휘
- GBM 대비 빠른 수행 시간: 다른 알고리즘에 비해 느릴 수 있음
- GBM과 달리 과적합 규제 기능 존재
- Tree pruning(나무 가지치기): 더 이상 긍정 이득이 없는 분할을 가지치기 해서 분할 수를 줄일 수 있음
- 자체 내장된 교차 검증
- 결손값 자체 처리 기능
### 하이퍼 파라미터 (사이킷런 파이썬 래퍼 XGBoost 기준)

- 일반 파라미터: 일반적으로 실행 시 스레드의 개수나 silent 모드 등의 선택을 위한 파라미터로 주로 디폴트 파라미터 값 사용   
    
    - booster: gbtree(tree based model) 또는 gblinear(linear model)  선택, default = gbtree
    - silent: 출력 메시지를 나타내고 싶지 않을 경우 1로 설정, defualt=0
    - nthread: CPU 실행 스레드 개수 조정, default= 전체 사용   
- 부스터 파라미터: 트리 최적화, 부스팅, regularization 관련 파라미터
    - eta[default=0.3, alias: learning_rate]: GBM의 학습률과 같은 파라미터, 사이킷런 클래스 사용시 learning_rate 파라미터로 대체되며 디폴트는 0.1이다, 보통 0.01~0.2 사이의 값 선호
    - num_bost_rounds: GBM의 n_estimators와 같은 파라미터
    - min_child_weight[default=1]: 트리에서 추가적으로 가지를 나눌지를 결정하기 위해 필요한 데이터들의 weight총합, 클수록 분할 자제, 과적합 조절을 위해 사용
    - gamma[default=0,alias: min_split_loss]: 트리의 리프 노드를 추가적으로 나눌지를 결정할 ㅣ최소 손실 감소 값, 해당 값보다 큰 손실이 감소된 경우에 리프 노드를 분리, 클수록 과적합 감소
    - max_depth[default=6]: 트리 기반 알고리즘의 max_depth와 같음, 0 지정하면 깊이에 제한이 없음, 높을 수록 과적합 가능성이 높아 보통 3~10 사이의 값 적용
    - sub_sample[default=1]: GBM의 subsample과 동일, 데이터를 샘플링하는 비율 지정, 과적합 제어, 통 0.5~1 사이의 값 사용
    -colsample_bytree[default=1]: GBM의 max_features와 유사, 트리 생성에 필요한 피처를 임의로 샘플링 하는데 사용
    -lambda[default=1,alias:reg_lambda]:L2 Regularization 적용 값, 피처 개수가 많을 경우 적용을 검토하며 값이 클수록 과적합 감소
    -alpha[dfault=0,alias:reg_alpha]:L1 Regularization 적용 값, 깊이 클수록 과적합 감소 효과
    -scale_pos_weight[default=1]: 특정 값으로 치우친 비대칭한 클래스로 구성된 데이터 세트의 균형을 유지하기 위한 파라미터
- 학습 테스크 파라미터: 학습 수행 시의 객체 함수, 평가를 위한 지표 등을 설정하는 파라미터
    - objective: 최솟값을 가져야할 손실 함수 정의, XGBoost는 많은 유형의 손실함수 사용 가능
    - bianary:logistic: 이진 분류일 때 적용
    - multi:softmax: 다중 분류일 때 적용, 이 때 레이블 클래스의 개수인 num_class 파라미터 지정 필요
    - multi:softprob: 위와 유사하나 개별 레이블 클래스의 해당되는 예측 확률 반환
    - eval_metric: 검증에 사용되는 함수 정의, default=회귀: rmse,분류: error
* 조기 중단 파라미터: 50으로 설정하면 학습 수행 중 50회 반복 중에 학습 오류가 감소하지 않으면 부스팅을 조기 중단함

## LightGBM
- XGBoost에 비해 학습 시간 훨씬 짧음
- 예측 성능에는 별다른 차이가 없음
- 적은 데이터 세트에 적용할 경우 과적합이 발생하기 쉽다 (일반적으로 10,000건 이하의 데이터 세트)
- 리프 중심 분할(Leaf Wise) 방식 사용: 트리의 균형을 맞추지 않고, 최대 손실 값을 가지는 리프 노드를 지속적으로 분할 (비대칭적ㅇ니 규칙 트리), 예측 오류 최소화 가능
    - 균형 트리 분할(Level Wise): 최대한 균형 잡힌 트리를 유지하면서 분할하기 때문에 트리의 깊이가 최소화 될 수 있다. but, 균형을 맞추기 위한 시간이 필요하다

### 하이퍼 파라미터
: 리프 토드가 계속 분할되면서 트리의 깊이가 깊어지므로 이 특성에 맞는 하이퍼 파라미터 설정이 필요함

- num_iterations[default=100]: 반복 수행하려는 트리의 개수(n_estimators와 같음, 사이킷런 호환 클래스에서 이 파라미터 사용 가능)
- learning_rate[default=0.1]: 업데이트되는 학습률 값(learning rate와 같음)
- max_depth[default=1]: 0보다 작은 값을 지정하면 깊이에 제한이 없다, leaf wise 기반이므로 깊이가 상대적으로 더 깊다
- min_data_in_leaf[default=20]: 결정 트리의 min_samples_leaf와 같은 파라미터. 최종 결정 클래스인 리프 노드가 되기 위해서 최소한으로 필요한 레코드 수, 과적함 제어, impurity가 꼭 0이 아니더라도 분할을 멈출 수 있도록 하는 방식
- num_leaves[default=31]: 하나의 트리가 가질 수 있는 최대 리프 개수
- boosting[default=gbdt]: 부스팅의 트리를 생성하는 알고리즘 기술
    - gbdt: 일반적인 그래디언트 부스팅 결정 트리
    - rf: 랜덤 포레스트
- bagging_fraction[default=1.0]: 트리가 커져서 과적합되는 것을 제어하기 위해서 데이터를 샘플링하는 비율 지정,(subsample과 같음)
- feature_fraction[default=1.0]: 개별 트리를 학습할 때마다 무작위로 선택하는 피처의 비율, 과적합 방지를 위해 사용
- lambda_l2[default=0.0]: L2 regulation 제어를 위한 값
- lambda_l1[default=0.0]: L1 regulation 제어를 위한 값

- Learning Task 파라미터
    - objective: 최솟값을 가져야 할 손실함수 정의 (회귀, 다중 클래스 분류, 이진 분류에 따라서 지정)
### 튜닝 방안
- num_leaves, min_data_in_leaf, max_depth 조절을 통해 과적합 방지
- 과적합 방지를 위해 reg_labda, reg_alpha(lambda_l1,lambda_l2)와 같은 regulation을 적용하거나 학습 샘플 수를 줄이기 위해 subsample, colsample_bytree (bagging_fraction, feature_function) 파라미터를 적용하는 방안 검토
- learning_rate 작게, n_estimators 크게 함으로써 예측 성능 향상 (과적합 유의)