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

# 다항 회귀
- 독립변수의 2차, 3차 방정식과 같은 다항식으로 표현되는 회귀
- 선형 회귀: 회귀 계수의 선형/ 비선형 여부가 기준. 독립변수의 선형/비선형 여부가 기준이 아님
- 사이킷런은 다항 회귀를 위한 클래스를 명시적으로 제공하지는 않지만 다항 회귀는 선형 회귀이기 때문에 비선형 함수를 선형 모델에 적용시키는 방법을 사용해 구현한다
-> PolynomialFeatures 클래스를 통해  피처를 Polynomial 피처로 변환
```
# Polynomial 활용 예제
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# 다항식으로 변환한 단항식 생성, [[0,1],[2,3]]dml 2X2 행렬 생성
X=np.arange(4).reshape(2,2)
output:[[0,1],[2,3]]

poly=PolynomialFeatures(degree=2)
poly.fit(X)
poly_ftr = poly.transform(X)
output:[[1,0,1,0,0,1],[1,2,3,4,6,9]]

# 피처 [x1,x2]를 [1,x1,x2,x1^2,x1x2,x2^2]으로 변환해줌
0 차항 만들기 싫으면, PolynommialFeatures(degree=2,include_bias=False) # include_bias=False로 지정해주면 됨
```
```
y= 삼차 다항식 결정값
poly_ftr=PolynomialFeatures(degree=3).fit_transform(X)

model = LinearRegression()
model.fit(poly_ftr,y)
회귀 계수: np.round(model.coef_,2)
회귀 shape: model.coef_.shape
```
-Pipeline 객체를 활용하면 피처 변환(PolynomialFeatures)과 선형 회귀(LinearRegression) 적용을 한 번에 구현할 수 있음
```
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

def polynomial_func(X):
    y=1+2*X[:,0]+3*X[:,0]**2+4X[:,1]**3
    return y
# Pipeline 객체로 Streamline하게 Polynomial Feature 변환과 Linear Regression을 연결
model = Pipeline([('poly',PolynomialFeatures(degree=3)),('linear',LinearRegression())])
X=np.arange(4).reshape(2,2)
y=polynomial_func(X)

model = model.fit(X,y)
회귀 계수: np.round(model.named_steps['linear'].coef_,2)
```
## 다항 회귀와 과적합/과소적합
- 다항식의 차수가 높을수록 복잡한 피처 간의 관계까지 모델링 가능,
그러나 과적합 문제 발생 가능
## 편향-분산 트레이드오프(Bias-Variance Trade off)
- Degree 1과 같은 모델은 매우 단순화된 모델: 고편향성을 가짐, 정확한 결과에서 벗어나면서도 예측이 특정 부분에 집중돼 있음
- Degree 15와 같이 매우 복잡한 모델: 고분산성을 가짐(높은 변동성), 실제 결과와 비교적 근접하지만, 예측 결과과 실제 결과 중심으로 넓은 부분에 분포되어 있음
- 추가: 단순한 모델의 경우 데이터를 모델에 충분히 포함시키지 못해 분산이 낮게 나오는 경향이 있으며 복잡한 모델의 경우 커다란 노이즈까지 반영하여 덜 정확한 추론을 하게 되는 경향이 있다

- 일반적으로 편황과 분산은 한 쪽이 높아지면 한쪽이 낮아지는 경향이 있음, 편향을 낮추고 분산을 높이면서 전체 오류가 가장 낮아지는 '골디락스' 지점을 지나 분산이 지속적으로 높아져 전체 오류가 다시 높아짐
- 따라서 편향과 분산을 트레이드오프하면서 전체 오류 값이 최저로 하는 모델을 구축할 필요가 있다