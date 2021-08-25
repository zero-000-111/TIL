# Regression
: 데이터 값이 평균 키로 회귀하려는 경향을 이용한 통계학 기법
- 주어진 피처와 결정 값 데이터 기반에서 학습을 통해 최적의 회귀 계수를 찾아내는 것
## 유형
1. 독립 변수 개수
    - 단일 회귀: 1개
    - 다중 회귀: 여러개
2. 회귀 계수의 결합
    - 선형 회귀: 선형
    - 비선형 회귀: 비선형
## 지도학습
1. Classification: 예측값이 카테고리와 같은 이산형 클래스 값
2. Regression: 예측 값이 연속형 숫자 값
## 선형 회귀 모델
    - 일반 선형 회귀: RSS를 최소화할 수 있는 회귀 계수 최적화, 규제 적용하지 않음
    - 릿지(Ridge): 선형 회귀에 L2 규제를 추가한 회귀 모델. L2 규제는 상대적으로 큰 회귀 계수 값의 예측 영향도를 감소시키기 위해서 회귀 계숫값을 더 작게 만드는 규제 모델
    - 라쏘(Lasso): 선형 회귀에 L1 규제를 함께 적용한 방식. L2 규제가 회귀 계수 값의 크기를 줄이는 데 반해, L1 규제는 예측 영향력이 작은 피처의 회귀 계수를 0으로 만들어 회귀 예측 시 피처가 선택되지 않게 함(피처 선택 기능)
    - 엘라스티넷(ElasticNet): L2, L1 규제를 함께 결합한 모델. 주로 피처가 많은 데이터 세트에서 적용되며, L1 규제로 피처의 개수를 줄임과 동시에 L2 규제로 계수 값의 크기를 조정
    - 로지스틱 회귀(Logistic REgression): 분류에 사용되는 선형 모델. 이진 분류뿐만 아니라 희소 영역의 분류(텍스트 분류)와 같은 영역에서 뛰어난 예측 성능을 보임

## 비용 함수(손실 함수)
RSS(Residual Sum of Square) = Error^2의 합의 평균
- 경사 하강법: 점진적으로 W 파라미터를 업데이트하면서 오류 값이 최소가 되는 W 파라미터를 구함
    1. w1, w0를 임의의 값으로 설정하고 첫 비용 함수의 값을 계산
    2. w1 -학습률(손실함수의 편미분 값)로 업데이트
    3. 비용 함수의 값이 감소한다면 반복. 감소하지 않으면 중단

## 사용 예시
```
def get_cost(y, y_pred):
    N=len(y)
    cost=np.sum(np.square(y-y_pred))/N
    return cost
def get_weight_updates(w1, w0, X, y, learning_rate=0.01):
    N=len(y)
    w1_update=np.zeros_like(w1)
    w0_update=np.zeors_like(w0)

    y_pred = w0 + np.dot(w1.T,X)
    diff = y-y_pred

    w0_factors= np.ones((N,1))
    
    w1_update=-(2/N)*learning_rate*np.dot(X.T,diff) # diff: n행 1열, X.T: 1행 N열 ~ Transpose 하여 내적 가능하게 함
    w2_update=-(2/N)*learning_rate*np.dot(w0_facotrs.T,diff)

    return w1_update, w0_update

def gradient_descent_steps(X, y, iters=10000):
    w0 = np.zeros((1,1))
    w1 = np.zeros((1,1))

    for ind in range(iters):
        w1_update, w0_update = get_weight_updates(w1, w0, X, y, learning_rate=0.01)
        w1 = w1 - w1_update
        w0 = w0 - wo_update
    return w1, w0
```
## LinearRegression 클래스
- class sklearn.linearmodel.LinearREgression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
### 파라미터
- 입력 파라미터
    - fit_intercept[default=True]: 불리언 값, intercept 값을 계산할지 말지 지정
    - normalize[default=False]: True일 시, 회귀 수행하기 전에 입력 데이터 세트를 정규화
- 속성
    - coef_: fit() 메서드를 수행했을 때 회귀 계수가 배열 형태로 저장하는 ㄴ속성
    - intercept_: intercept 값
## 회귀 평가 지표
- MAE(Mean Absolute Error): 실제 값과 예측값의 차이를 절댓값으로 변환해 평균한 것
    - API: metrics.mean_absolute_error 
    - scoring 함수 파라미터 값: 'neg_mean_absolute_error' #  평가 지표 값에서 음수로 만들어 반환함, 오류가 적을수록 더 좋은 평가를 할 수 있도록 함(사이킷런의 Scoring 함수는 값이 클수록 좋은 평가로 인식하기 때문)
- MSE(Mean Squared Error): 실제 값과 예측값의 차이를 제곱해 평균한 것
    - API: metrics.mean_sqared_error
    - scoring 함수 파라미터 값: 'neg_mean_squared_error'
- RMSE(Root Mean Squared Error): MSE에 루트를 씌운 것
- R^2: 실제 값의 분산 대비 예측값의 분산 비율, 1에 가까울수록 예측 정확도가 높음
    - API: metrics.r2_score
    scoring 함수 파라미터 값: 'r2'