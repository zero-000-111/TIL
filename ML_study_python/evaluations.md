# Evaluation

## 분류의 성과지표
1) 정확도(Accuracy)
2) 오차행렬(Confusion Matrix)
3) 정밀도(Precision)
4) 재현율(Recall)
5) F1 스코어
6) ROC AUC

* Accuracy : 예측 성공 데이터 건수/ 전체 예측 데이터 건수  
**유의:**  
불균형한 레이블 값 분포에서 모든 데이터를 True로 판단하는 이상한 모델들도 높은 성능이 나올 수 있음

* Confusion Matrix : 학습된 분류 모델이 예측을 수행하면서 얼마나 헷갈리고 있는지도 함께 보여주는 지표
```
    TN,FP,FN,TP 형태로 오차 행렬이 나타남
    # 귀무가설이 Negative로 가정한 듯 함
    True Negative : 예측과 실제 값이 같음, 이때 실제 값은 Negative
    False Posivtive : Positive로 예측했지만 틀림 (1종 오류)
    False Negative : Negative로 예측했지만 틀림 (2종 오류)
    True Positive : Positive로 예측하고 맞음

    from sklearn.metrics import confusion_matrix

    confusion_matrix(y_test,fakepred)
    위의 경우와 마찬가지로 Positive 데이터가 Negative에 비해 상당히 적으로 경우 FN의 수가 적어지고 FP 또한 Postive로 예측하는 경우도 감소하기 때문에 같이 적어지는 편향이 발생할 수 있다
```
* 정밀도 : Postivie로 예측한 것 중 맞은 것  (유의 수준이 높음, 얼마만큼 안틀리는가(FP낮을수록 훨씬 유리))
TP/(FP+TP)
* 재현율 : 실제 값이 Positive인 것 중 예측에 성공한 것  (검증력과 유사, 얼마만큼 잘 잡는가-FN 낮을수록 유리)
TP/(FN+TP)

```
    이진 분류 모델의 업무 특성에 따라 정밀도와 재현율 지표 중 특정 평가 지표가 더 중요한 지표로 간주될 수 있음

    예) 실제 Positive 양성 데이터를 Negative로 잘못 판단하게 되면 업무상 큰 영향이 발생한는 경우 (검증력이 중요한 경우, 2종 오류에 민감한 경우)
        스팸 메일 분류| Negative로 잘못 예측하면 불편한 정도로 그치지만 Positive로 잘못 예측하면 중요한 메일을 못 볼 수 있믕 (1종 오류에 민감한 경우)
```
```
    from sklearn.metrics import accuracy_score, precision_score,recall_score,confusion_matrix

    def get_clf_eval(y_test,pred):
        confusion= confusion_matrix(y_test,pred)
        accuracy = accuracy_score(y_test,pred)
        precision=precision_score(y_test,pred)
        recall=recall_score(y_test,pred)
    import pandas as pd
    from sklear.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    titanic_df=pd.read_csv("train.csv")
    X_train_df=titanic_df.drop("Survived",axis=1)
    y_train_df=titanic_df["Survived"]
    X_train_df=transform_features(X_train_df) # 앞서 작성한 preprocessingg 함수

    X_train,y_train,X_test,y_test=train_test_split(X_train_df,y_train_df,test_size=0.2,)

    lr_clf=LogisticRegression()

    lr_clf.fit(X_train,y_train)
    pred=lr_clf.predict(X_test)
    get_clf_eval(y_test,pred)
```