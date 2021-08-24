## 비교 조건문으로 DataFrame 불리언 인덱싱
```
# 조건에 해당하는 인덱스 추출
idx= df[(df['Age']<4) | (df['Age']>60)].index

# 조건마다 () 표시: 없으면 TypeError: cannot compare a dtyped [object] array with a scalar of type [bool] 발생

# and or -> & | 로 표시: 아니면 ValueError: The truth value of a series is ambiguous. Use a empty, a.bool(), a.item(), a.any() or a.all() 발생
```

## 넘파이 random 모듈 이해하기
- np.random.seed(n): seed를 통한 난수 생성, seed(0) 지정하면 같은 seed 안에서는 같은 난수 발생, seed(1)로 새로운 seed 지정하면 생성되는 난수 달라짐
- np.random.randint: **균일 분포**의 정수 난수 1개 생성
- np.random.rand(m,n): 0과 1사이의 **균일 분포**에서 난수 matrix array 생성 # m행 n열 matrix
- np.random.randn: **가우시안 표준 정규 분포**에서 난수 matrix array 생성
- np.random.shuffle: 기존의 데이터의 순서 바꾸기
- np.random.choice: 기존의 데이터에서 샘플링
