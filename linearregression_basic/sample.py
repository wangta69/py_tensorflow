import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 데이타 셑 만들기
x = np.arange(-10, 10, 1)  # -10 에서 10까지 1씩 증가하는 1차춴 배열 생성
y = 2 * x + 7
print(x.shape)  # (20, )

# plt.plot(x, y)
# plt.show()

idx = np.arange(x.shape[0])
np.random.shuffle(idx)

# Train 데이타 생성
X_train = x[idx]  # 입력값
y_train = y[idx]  # 출력값

# 1 차원의 형태의 x 배열을 2차원 형태로 변환
X_train = X_train.reshape(-1, 1)  # 첫번째 인자는 행의 개수, 두번째 인자는 열의 개수 -1은 정해진 개수를 나타내지 않음

# Training
lr = LinearRegression()
lr.fit(X_train, y_train)  # 입출력 train 데이타를 입력한다.
print('기울기', lr.coef_)
print('y절편', lr.intercept_)

# 테스트 데이타 생성
X_test = np.arange(11, 16, 1).reshape(-1, 1)
pred = lr.predict(X_test)  # 예상값 생성 (input 에 대한 예상값 설정
print(X_test)
print(pred)
