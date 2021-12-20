# LinearRegression 을 이용한 수식 계산

## 모듈 import
```
import numpy as np
from sklearn.linear_model import LinearRegression
```
## 데이타 셑 만들기
```
x = np.arange(-10, 10, 1)  # -10 에서 10까지 1씩 증가하는 1차춴 배열 생성
y = 2 * x + 7
print(x.shape)  # (20, )

idx = np.arange(x.shape[0])
np.random.shuffle(idx)
```
## Train 데이타 생성
머신러닝에서 x 변수의 데이터 값 배열은 가로 방향으로 긴 형태가 아니라 세로 방향으로 긴 형태로 만듭니다. 
따라서, x 배열의 형태를 세로 형태로 변환합니다. 
```
X_train = x[idx]  # 입력값
y_train = y[idx]  # 출력값

# 1 차원의 형태의 x 배열을 2차원 형태로 변환
X_train = X_train.reshape(-1, 1)  # 첫번째 인자는 행의 개수, 두번째 인자는 열의 개수 -1은 정해진 개수를 나타내지 않음
```
## Training
```
lr = LinearRegression()
lr.fit(X_train, y_train)  # 입출력 train 데이타를 입력한다.
print('기울기', lr.coef_)
print('y절편', lr.intercept_)
```
## 테스트 데이타 생성 및 결과 확인
새로운 입력값을 만들어 결과값을 확인합니다.
```
X_test = np.arange(11, 16, 1).reshape(-1, 1)  # X_test 도 위의 X_train과 동일한 차원을 가져야 합니다.
pred = lr.predict(X_test)  # 예상값 생성 (input 에 대한 예상값 설정
print(X_test)
print(pred)

```