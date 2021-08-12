# Numpy Pandas Basics

## Numpy

* ndarray: 다차원 배열 (Multi-dimension) 배열을 쉽게 생성및 연산 가능

### ndarray CRUD
**Create**

``` 
import numpy as np
array1 = np.array([1,2,3]) -> array1.shape:(3,) &nbsp; # 1차원 데이터    
 array3 = np.array([[1,2,3]]) -> array3.shape(1,3)&nbsp; # 2차원 데이터 (1행 3열)
```
* arange, zeros, ones (편리하게 생성하기)   
```
    1. arange(): 파이썬 range()와 유사
        sequence_array = np.arange(10)
        output: [0 1 2 3 4 5 6 7 8 9]
    2. zeros(): shqpe 값 입력하면 모든 값을 0으로 할당
        zeros=np.zeros((3,2),dtype='int32) # 튜플형태로 입력
        output : [[0 0][0 0][0 0]]
    3. ones() : zeros와 동일하게 활용
``` 

**Update**

* reshape() : 차원크기 변경
```
    array1=np.arange(10)    #shape(10,)
    array2=array1.reshape(5,2)
        output=[[0 1][2 3][4 5][6 7][8 9]]
    array3=narray1.reshape(4,3)  # 사이즈 변경 불가
```
    reshape(-1,1)의 활용
```
    array4=array1.reshape(-1,5) # 다섯개의 컬럼에 맞는 로우 자동 생성, 이것도 호환이 되어야 함

    array = np.arange(8)
    array3d = array.reshape((2,2,2))
    output : [[[0 1][2 3]][4 5][6 7]] #[[행렬][헹렬]], 3차원 행렬

    array3d.reshape(-1,1)
    output : [[0] [1] [2] [3] [4] [5] [6] [7]]
```
* 정렬
```
    array.sort() # 원 행렬을 바꾸고 반환값 None
    np.sort(array) # 원행렬 그대로 반환값 sort

    np.argsort() 3 정렬된 행렬의 원봉 행렬 인덱스 반환
    
    name=np.array(['june','july','agust','may'])
    score=np.array([23,56,34,09])
    score_index=np.argsort(score)[::-1]
    name[score_index] # fancy indexing 통해 검색됨

```

**Read**
* Indexing
```
    array[axis0=1,axis1=2] # 2행 3열 인덱싱

    array[0:5,:] # row index값 0~4까지 인덱싱

    array[[0,1],3] # 1행 4열, 2행 4열 인덱싱

    array[False, True, Ture, False, False] # True 값만 인덱싱하는 Boolean indexing
    -> 응용 : 
        array[array[,1]>6,:]
```
Boolean indexing 작동 단계
> 1: 조건에 따라 True 값인 index 값만 저장  
  2: 해당 index 값으로 ndarray 조회

## Pandas
* numpy 기반 편리한 데이터 핸들링
### Panda CRUD
**Create**
```
    import pandas as pd
    titanic_df=pd.read_csv(r'파일 경로') # 동일 디렉토리는 파일 이름.확장자만 써도 가능  # DataFrame 생성
```
* 특징 : 첫 줄에 index 자동 생성  

**Read**
* DataFrame 살피기
```
    titanic_df.shape # 크기 튜플 형태로 반환
    titanic_df.info() # 각 컬럼별 데이터 타입, 크기, 컬럼 수
    titanic_df.describe() # 각 컬럼별 count (not null 데이터 건수), mean, std, min...

    titancic_df['Pclass'].value_counts() # 각 유형별 건수 # Series 객체에서만 활용

    titanic_df.isna().sum() # 결손 데이터 개수 구하기
``` 
* [ ] 연산자
```
    칼럼 명 문자, 인덱스로 변환 가능한 표현식만 가능
    titanic_df[['Pclass','Age']]
    titanic_df[0:2] # 0,1 행 추출, 인덱스로 변환 가능하기 때문에 가능, 되도록 지양
    titanic_df[titanic_df['Pclass']==3] # 인덱스로 변환 가능, 조건에 맞는 행 추출
```
*iloc[ ]
```
    위치 기반 인덱싱만 허용하기 때문에 integer/integer형의 슬라이싱, 팬시 리스트 값만 허용

```
*loc[ ]
```
    명칭 기반 데이터 추출, 행:DataFrame index값, 열: 칼럼 명

    
    titanic_df[[0:2],'Pclass'] # 0,1,2 까지 포함, 명칭기반이기 때문
```
**Update**
* 컬럼 생성/수정
```
    titanic_df['Age3']=3 # 해당 컬럼 네임 없으므로 새로 생성

    titanic_df['Age30']=titanic_df['Age3']*10

    titanic_df['Richpoor']=titanic_df['Pclass'].apply(lambda x: 'Rich' if x==1 else 'Poor') # 너무 길면 함수 적용 가능
```
* 정렬
```
    titanic_sorted = titanic_df.sort_values(by=['Name'],ascending=False)
```
* aggregation 함수
```
    titanic_df.count() # 모든 칼럼에 적용

    titanic_groupby=titanic_df.groupby('Pclass')[['PassengerID','Survived']].count()

    titanic_df.groupby('Pclass').agg([max,min]) # 여러 함수 적용할 필요 있을 때 활용, 딕셔내리 형태로 인수 전달 가능 agg_fortmat={'Age':'max','SibSp':'Sum'}
```
* 결손 데이터 처리
```
    titanic_df['Cabin'] = titanic_df['Cabin'].fillna(titanic_df['Cabin'].mean()) # inplace defualt False, True해야 원본 변경
```
**Delete**
```
    titanic_df.drop('Age3',axis1=1) # 해당 열 전체 삭제 반화, 원봉 행렬 그대로
    
    titanic_df.drop([0,1,2],axis=0,inplace=Ture) # 해당 행 삭제, 원본 행렬 변경, 반환값 None
```