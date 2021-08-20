# 분류
### : 기존 데이터가 어떤 레이블에 속하는지 패턴을 알고리즘으로 인지한 뒤에 새롭게 관측된 데이터에 대한 레이블을 판별하는 것
<br>

## 결정 트리 (Decision Tree)
: if, else를 자동으로 찾아내 예측을 위한 규칙을 만드는 알고리즘 (스무고개 알고리즘)
* 규칙 노드(Decision Node): 규칙조건, 이 때마다 새로운 서브 트리 생성(Sub Tree)
* 리프 노드(Leaf Node): 결정된 클라스 값
*  많은 규칙이 있다는 것은 분류 결정방식이 복잡해진 다는 것 == 과적합으로 이어지기 쉽다
### 균일도
* 균일도가 높을 수록 적은 규칙(정보)으로 분류 가능 => 균일도가 높은 데이터 세트를 먼저 선택할 수 있도록 하는 것이 중요
* 균일도를 측정하는 대표적인 방법: 정보이득   
<br>
엔트로피는 주어진 데이터 집합의 혼잡도, 서로 다른 값이 섞여 있으면 엔트로피가 높고, 같은 값이 섞여 있으면 엔트로피가 낮다   
<정보이득 = 1- 엔트로피지수>
### 결정 트리 모델의 특징
* 장점
1. 균일도라는 룰을 기반으로 하고 있어서 알고리즘이 쉽고 직관적
2. 균일도만 신경 쓰면 되기 때문에 특별한 경우를 제외하고는 각 피처의 스케일링과 정규화 같은 전처리 작업이 필요 없음 
*단점
1. 과적합으로 정확도가 떨어짐 : 서브트리를 계속 만들다 보면 피처가 많고 균일도가 다양하게 존재할 수록 트리의 깊이가 커지고 복잡해짐

### DecisionTreeClassifier()
```
  DecisionTreeClassifier()
Parameter
  1. min_samples_split: 노드를 분할하기 위한 최소한의 샘플 데이터 수 / 과적합 제어에 사용
  2. min_samples_leaf: 말단 노드가 되기 위한 최소한의 샘플 데이터 수 / 과적합 제어 but, 비대칭적 데이터의 경우 작게 설정할 필요가 있을 수 있음
  3. max_features: 최적의 분할을 위해 고려할 최대 피처 개수,default=None (모든 피처를 사용해 분할 수행)
  4. max_depth: 트리의 최대 깊이 규정 / default=None(계속 분할/min_samples_split보다 작아질 때까지 수행)
  5. max_leaf_nodes: 말단 노드(leaf)의 최대 개수
```
### 결정 트리 모델의 시각화

[결정모델 시각화](../jupyter_notebook/classification_viz.png)


### 결정 트리 과적합
* DecisionTree_overfitting.ipynb
* TIL(2021-8-20)
```
# 중복된 column_name 얼마나 있는지 확인 가능
## column_index의 건수가 feature_dup_df에 적힘
feature_dup_df = feature_name_df.groupby('column_name').count()
## 적힌 개수가 2 이상, 즉 중복된 row의 개수 확인 
feature_dup_df[feature_dup_df['column_index']>1].count()

# 중복 피처명 대체하기
def get_new_feature_name_df(old_feature_name_df):
  feature_dup_df = pd.DataFrame(data=old_feature_name_df.groupby('column_name').cumcount(),columns=['dup_cnt']) # cumcount() : 중복된 컬럼을 축적하여 셈 0,1,2... 순으로
  feature_dup_df=feature_dup_df.reset_index()
  new_feature_name_df = pd.merge(old_feature_name_df.reset_index(), feature_dup_df,how='outer')
  new_feature_name_df['column_name] = new_feature_name_df[['column_name','dup_cnt']].apply(lambda x: x[0]+'_'+str(x[1]) if x[1]>0 else x[0], axis=1)
  new_feature_name_df =new_feature_name_df.drop(['index'],axis=1)
  return new_reature_name_df
```