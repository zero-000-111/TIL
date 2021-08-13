# scikit-learn basics

## iris datasets 활용
* DataFrame 만들기
```
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

iris=load_iris() # 붓꽃 데이터 세트 로드
iris_data=iris.data # 데이터 세트 중 {data:numpy array} 부분 추출
iris_label=iris.target # 데이터 세트 중 {target:numpy array} 부분 추출

iris_df=pd.DataFrame(data=iris_data,columns=iris.feature_names) # feature_names를 컬럼 명으로 사용
iris_df['label']=iris.target # label column 추가
```
* 사이킷런의 기반 프레임워크
> fit() : 입력 데이터의 형태에 맞춰 데이터를 변환하기 위한 사전 구조를 맞추는 작업

> 실제 작업은 transform() / predict() 를 통해 이루어짐 # fit_transform()으로 한번에 작업하기도 함

* Model Selection 모듈

    1. train_test_split()
    ```
        train_test_plit(feature_dataset,label_dataset,test_size,shuffle_random_state)
        test_size : 테스트 데이터 세트 크기를 얼마나 샘플링할 것인가 (default : 0.25)
        shuffle : 데이터를 분리하기 전에 데이터를 미리 섞을지 결정
        random_state : 호출할때마다 동일한 학습/데이터 세트를 생성하기 위해 주어지는 난수 값
    ```
    ```
    ## 실제 예시

    dt_clf=DecisionTreeClassifier()
    iris_data=load_iris()

    X_train,X_test,y_train,y_test = train_test_split(iris_data.data,iris_data.target,test_size=0.3, random_state=121)

    dt_clf.fit(X_train,y_train)
    pred=dt_clf.predict(X_test)
    ```

    2. 교차 검증 : 여러번 학습과 검증 세트에서 알고리즘 수행하고 평가하는 것
    >  장점 1) 과적합 방지 : 고정된 학습 데이터와 테스트 데이터는 편향을 발생시킬 수 있음

    >       2) 여러 특성을 가진 데이터 세트를 활용함으로써 분포도, 이상치, 피쳐 중요도 등의 편중 방지
    * K-Fold 교차검증 : K개의 데이터 폴드 세트를 만들어서 K번 만큼 각 폴드 세트에 학습과 검증 평가를 반복적으로 수행하는 방법
    ```
        iris=load_iris()
        features=iris.data
        label=iris.target
        dt_clf=DecisionTreeClassifier(random_state=156)

        Kfold=KFold(n_splits=5) # 5개의 폴드 세트로 분리하는 KFold 객체 생성
        cv_accuracy=[] # 폴드 세트별 정확도를 담을 리스트 객체

        n_iter=0
         # KFold 객체에 split()를 호출하면 폴드 별 학습용, 검증용 테스트의 로우 인덱스를 반환
        for train_index,test_index in kfold.split(features):
            X_train,X_test=features[train_index],features[test_index]
            y_train,y_test=label[train_index],label[test_index]

        dt_clf.fit(X_train,y_train)
        pred=dt_clf.predict(X_test)
        n_iter+=1

        accuracy=np.round(accuracy_score(y_test,pred),4)
        cv_accuracy.append(accuracy) # 반복 시마다 정확도 측정

        np.mean(cv_accuracy) # 평균 정확도 계산
    ```
    * Stratified K-Fold : 불균형한 레이블 데이터 집합을 위한 K-Fold 방식
        작동 방식은 K-Fold와 비슷
    ```
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=3)
        n_iter=0
        # 피쳐 데이터 세트 뿐만 아니라 레이블 데이터 세트도 반드시 필요함
        for train_index, test_index in skf.split(iris_df,iris_df['label']):
            n_iter+=1
            label_train=iris_df['label'].iloc[train_index]
            label_test=iris_df['label'].iloc[test_index]
    ```
     * cross_val_score : K-Fold를 한번에
     ```
        cross_val_score(estimator,X,y=None,cv=None, n_jobs=1,verbose=0,fit_params=None, pre_dispatch='2*n_jobs')
            # Stratified K-fold 방식으로 수행하지만 회귀는 불가하기 때문에 K-Fold 방식으로 분할
            # scoring 파라미터로 지정된 성능 지표 측정값을 배열 형태로 반환
        estimator : Classifier 또는 Regressor
        X : 피쳐 데이터 세트
        y : 레이블 데이터 세트
        scoring : 예측 성능 평가 지표
        cv : 교차 검증 폴드 수
     ```
     예제:
     ```
        iris_data=load_iris()
        dt_clf=DecisionTreeClassifier()

        feature=iris_data.data
        label = iris_data.target

        scores=cross_val_score(dt_clf,feature,label,scoring='accuracy',cv=3)
        np.round(scores,4) # 교차 검증별 정확도 소숫점 4자리에서 반올림
        np.mean(scores) # 평균 검증 정확도
     ```

     * GridSearchCV : 교차 검증과 최적 하이퍼 파라미터 튜닝을 한번에
     > 알고리즘에 적용되는 파라미터들의 조합을 순차적으로 적용해보면서 최고 성능을 가지는 파라미터 조합을 찾음
     ```
        GridSearch(estimator, param_grid, scoring, cv, refit=True)

        estimator : classifier, regressor, pipeline 사용 가능
        param_grid : key + 리스트 값을 가지는 딕셔너리가 주어짐. 튜닝을 위해{파라미터 명: [여러 파라미터 값], ...}
        scoring : 선능 평가 방법
        cv : 교차 검증을 위해 분할되는 학습/테스트 세트의 갯수
        refit : True로 생성 시 가장 최적의 하이퍼 파라미터를 찾은 뒤 입력된 estimator 객체를 해당 하이퍼 파라미터로 재학습시킴
     ```
     예제
     ```
        iris_data=load_iris()
        X_train,X_test,y_train,y_test = train_test_split(iris_data.data,iris_data.target,test_size=0.2)

        dtree=DecisionTreeClassifier()

        parameters={'max_depth':[1,2,3],'min_sample_split':[2,3]}
        
        grid_dtree=GridSearchCV(dtree,param_grid=parameters,cv=3,refit=True)    # 객체 생성

        grid_dtree.fit(X_train,y_train) # 하이퍼 파라미터를 순차적으로 평가

        scores_df=pd.DataFrame(grid_dtree.cv_results_) # 총 6개의 결과 출추해 DataFrame으로 변환
        scores_df[['params','mean_test_score','rank_test_score','split0_test_score','split1_test_score','split2_test_score']]

        # params 컬럼: 적용된 개별 하이퍼 파라미터 조합 표현
          rank_test_score : 성능이 좋은 score 순위를 나타냄. 1이 가장 뛰어남
          mean_test_score : 개별 하이퍼 파라미터별로 CV의 폴딩 테스트 세트에 대해서 총 수행한 평가 평균값

        # refit = True, 최적 성능의 하이퍼 파라미터 조합으로 Estimator를 학습해 best_estimator_로 조합
        estimator = grid_dtree.best_estimator_
        pred = estimator.predict(X_test)
     ```
* 데이터 전처리
    1. 데이터 인코딩
    * 레이블 인코딩 : 카테고리 피러를 코드형 숫자 값으로 변환
    ```
        from sklearn.preprocessing import LabelEncoder

        items=['TV','냉장고','전자레인지','컴퓨터'.'선풍기','믹서','믹서']
        # LabelEncoder 객체 생성 후 fit(), transform()으로 레이블 인코딩 수행
        encoder=LabelEncoder()
        encoder.fit(items)
        labels=encoder.transform(items)

        encoder.classes_ : 인코딩 값에 대한 원본값 확인 가능

        # 디코딩
        encoder.inverse_transform([4,5,2,0,1,1,3,3])

        # 숫자로 인코딩 되기 때문에 순서나 중요도를 인식하는 회귀와 같은 알고리즘에는 부적합
    ``` 
    * 원-핫 인코딩(One-Hot Encoding) : 피처 값의 유형에 따라 새로운 피처를 추가해 고유 값에 해당하는 칼럼만 1 부여
    ```
        from sklearn.preprocessing import OneHotEncoder
        import numpy as np

        items=['TV','냉장고','전자레인지','컴퓨터'.'선풍기','믹서','믹서']
        encoder=LabelEncoder()
        encoder.fit(items)
        labels = encoder.transform(items)

        labels= labels.reshape(-1,1)

        oh_encoder = OneHotEncoder()
        oh_encoder.fit(labels)
        oh_labels = oh_encoder.transform(lables)
        # 숫자로 인코딩된 순서대로 피처 컬럼이 새로 생성됨
    ```
    2. 피처 스케일링 : 서로 다른 변수의 값 범위를 일정한 수준으로 맞추는 작업
    > 표준화 : 데이터의 피처 각각이 평균 0이고 분산이 1인 가우시안 정규 분포를 가진 값으로 변환

    > 정규화: 서로 다른 피처의 크기를 통일하기 위해 크기를 변환해주는 개념

    * StandardScaler : 표준화를 쉽게 지원하기 위한 클래스, 데이터가 가우시안 분포를 가지고 있다고 가정
    ```
        from sklearn.datasets import load_iris
        from pandas as pd

        iris = load_iris()
        iris_data = iris.data
        iris_df = pd.DataFrame(data=iris_data,columns=iris.feature_names)

        scaler = StandardScaler()
        scaler.fit(iris_df)
        iris_scaled = scaler.transform(iris_df)

        # transform() 시 스케일 변환된 데이터 세트가 ndarray로 변환돼 이를 DataFrame으로 변환
        iris_df_scaled=pd.DataFrame(data=iris_scaled,columns=iris.feature_names)
    ```
    * MinMaxScaler : 데이터 값을 0과 1 사이의 범위 값으로 변환
    ```
        from sklearn.preprocessing import MinMaxScaler

        scaler=MinMaxScaler()
        scaler.fit(iris_df)
        iris_scaled=scaler.transform(iris_df)

        iris_df_scaled = pd.DataFrame(data=iris_scaled,columns=iris.feature_names)
    ```
    **유의점**   
    - fit()을 학습 데이터와 테스트 데이터에 각각 적용 시 서로 다른 스케일이 적용 될 수 있음 => 전체 데이터에 대해서 스케일링 변환을 한후 학습/테스트 분리 권장 , 여의치 않으면 fit()된 scaler객체를 이용해 바로 transform
