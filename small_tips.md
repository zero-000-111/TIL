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
