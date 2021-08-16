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
* 정밀도/재현율 트레이드오프

> 분류의 결정 임계값 (Threshold)을 조정해 정밀도/ 재현율 수치 조절 가능 (하나를 강제로 높이면 다른 하나의 수치가 떨어질 수 있음)

```
    * predict_proba() : 개별 데이터 별로 예측 확률을 반환

    pred_proba = Ir_clof.predict_proba(X_test)
    pred = Ir_clf.predict(X_test)
    
    pred_proba[:3]
->  Output: [[class0prob, class1prob]   
             [class0prob, class1prob]
             [class0prob, class1prob]] 
             # predict() 메서드는 두 prob 중 큰 쪽으로 예측 (임계값 : 0.5)
             # predict() 메서드는 predict_proba() 메서드를 기반해 생성된 API
    pred_proba_result = np.concatenate([pred_proba,pred.reshape(-1,1),axis=1])
```
*  Binarizer 클라스
```
    from sklearn.preprocessing import Binarizer

    X=[[1, -1, 2], [2, 0, 0],[0,1.1 ,1.2]]
    binarizer = Binarizer(threshold=1.1)
    # 임계값 1.1로 설정 -> 넘으면 1 못 넘으면 0

    binarizer.fit_transform(X)
```
* precision_recall_curve() : 임계값별 평가 지표
```
    from sklearn.metrics import precision_recall_curve

    # 레이블 값이 1일 때의 예측 확률 추출
    pred_proba_class1 = Ir_clf.predict_proba(X_test)[:,1]

    precisions,recalls,thresolds = precision_recall_curve(y_test, pred_proba_class1)

    # 반환된 임계값 배열 로우가 147건이므로 샘플로 10건만 추출하되, 임곗값을 15 Step으로 추출
    thr_index = np.arange(0,thresold.shape[0],15)

    # 샘플용 10개의 임곗값
    np.round(thresolds[thr_index],3)

    np.round(precisions[thr_index],3)
    np.round(recalls[thr_index],3)
```
* 정밀도와 재현율 곡선 시각화
```
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    %matplotlib inline

    def precision_recall_curve_plot(y_test,pred_prob_c1):
        precisions,recalls,thresolds=precision_recall_curve(y_test,pred_proba_c1)

        plt.figure(figsize=(8,6))
        thresold_boundary=thresolds.shape[0]
        plt.plot(thresholds,precisions[0:threshold_boundary],linestyle='--',label='precision')
        plt.plot(threshold,recalls[0:threshold_boundary],label='recall')

        # threshold 값 X축의 Scalse을 0.1 단위로 변경
        start,end=plt.xlim()
        plt.xticks(np.round(np.arange(start,end,0.1),2))

        plt.xlabel('Threshold value'); plt.ylabel('Precision and REcall value')
        plt.legend();plt.grid()
        plt.show()
```
* F1 score : 정밀도와 재현율을 결합한 지표
```
    F1 = 2/(1/recall)+(1/precision)
    2 x (precision x recall)/(precision+recall)
```
```
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test,pred)
```
* ROC(Receiver Operation Characteristic Curve) 곡선과 AUC
ROC : FPR(False Postitive Rate)이 변할 때 TPR(True Positive Rate - 재현율/민감도)이 어떻게 변하는지를 나타내는 곡선

TNR(True Negative Rate) : 실제값 Negative가 정확히 얘측돼야 하는 수준

roc_curve()
```
    from sklearn.metrics import roc_curve

    # 레이블 값이 1일때의 예측 확률을 추출
    pred_proba_class1=Ir_clf.predict_proba(X_test)[:1]

    fprs , tprs, thresholds = roc_curve(y_test, pred_proba_class1)
    
    # 반환된 임계값 배열에서 샘플로 데이터를 추출하되, 임곗값을 5 step으로 추출
    thresholds[0]은 max(예측확률)+1로 임의 설정됨. 이를 제외하기 위해 np.arrange는 1부터 시작
    thr_index = np.arange(1,thresholds.shpae[0],5)
    np.round(fprs[thr_index],3)
    np.round(tprs[thr_index],3) # 각 임계값별 반환값 추출
```

AUC(Area Under Curve) : ROC와 밑 면적을 구하는 것으로 1에 가까워 질수록 좋음, FPR이 작은 상태에서 얼마나 큰 TPR을 얻을 수 있을지가 관건