## 파일 경로 참조
- vscode 활용시 설정되어 있는 현재 작업 디렉토리 확인 필요
```
import os
current_dir=os.getcwd()
print(current_dir)
```
- /: 최상위 경로
- ./: 현재 작업중인 디렉토리
- ../: 현재 작업중인 디렉토리 기준 상위 디렉토리

# 파이썬 strip 함수
- 인자로 문자 1개를 전달하면 그 문자와 동일한 것을 모두 제거한다

```
word = '000:000;000'

word.lstrip('0')
output: ':000;000'

word.rstrip('0')
output: '000:000;'

word.strip('0')
output:':000;'
```

# 파이썬 pass, continue, break 비교

- pass: 아무 영향을 안줌
    - 조건문에서 딱히 넣을 조건이 없을 경우
    - class 선언시 초기에 넣을 값이 없을 경우
- continue: 다음 루프로 넘어감
- break: 반복문을 멈추고 밖으로 나감