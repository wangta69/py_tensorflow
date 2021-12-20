# TensorFlow Keras 을 이용한 수식 계산
데이터 분포가 선형관계일 때 사이킷런의 선형회귀 모형인 LinearRegression 클래스는 매우 정확한 예측력을 갖습니다.
하지만 데이타가 선형이 아닐경우는 어떨까요?
여기서는 선형데이타를 사용하겠지만 그렇지 않은 경우도 keras는 매우 잘 인식합니다.
## 모듈 import
```
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense
```
## 데이타 셑 만들기
```
x = np.arange(-10, 10, 1)  # -10 에서 10까지 1씩 증가하는 1차춴 배열 생성
y = 2 * x

idx = np.arange(x.shape[0])
np.random.shuffle(idx)
```
## Train 데이타 생성
```
X_train = x[idx]  # 입력값
y_train = y[idx]  # 출력값

# 1 차원의 형태의 x 배열을 2차원 형태로 변환
X_train = X_train.reshape(-1, 1)  # 첫번째 인자는 행의 개수, 두번째 인자는 열의 개수 -1은 정해진 개수를 나타내지 않음
```
## Training
* optimizer 종류 : https://keras.io/ko/optimizers/
    * SGD
        * 모멘텀과 네스테로프 모멘텀(Nesterov momentum), 그리고 학습률 감소 기법(learning rate decay)을 지원합니다.
    * RMSprop
        * RMSProp을 사용할 때는 학습률을 제외한 모든 인자의 기본값을 사용하는 것이 권장됩니다.
        * 일반적으로 순환 신경망(Recurrent Neural Networks)의 옵티마이저로 많이 사용됩니다.
    * Adagrad
        * Adagrad는 모델 파라미터별 학습률을 사용하는 옵티마이저로, 파라미터의 값이 업데이트되는 빈도에 의해 학습률이 결정됩니다. 파라미터가 더 자주 업데이트될수록, 더 작은 학습률이 사용됩니다.
    * Adadelta
        * Adadelta는 Adagrad를 확장한 보다 견고한 옵티마이저로 과거의 모든 그래디언트를 축적하는 대신, 그래디언트 업데이트의 이동창(moving window)에 기반하여 학습률을 조절합니다 
    * Adam
* loss : https://keras.io/ko/losses/
* Dense : https://han-py.tistory.com/207
    * 완전연결층 Dense 레이어를 add 명령을 사용하여 추가합니다. units=1 인자는 목표변수(타겟 레이블)의 출력 데이터가 1가지 종류라는 뜻입니다. 
    * 회귀 문제는 하나의 단일 값을 예측하기 때문에 1로 지정합니다. input_shape=(1,) 옵션은 입력 데이터인 설명변수(x 변수)의 데이터 구조를 지정합니다.
    * 여기서는 1개의 열, 즉 입력 변수인 x가 1개라는 뜻입니다. 
```
model = keras.Sequential()
# model.add(Dense(units=1, input_shape=(1,)))
model.add(Dense(units=1, activation='linear', input_shape=(1, )))
model.compile(optimizer='sgd', loss='mse')
# model.summary()

model.fit(X_train, y_train, epochs=10)  # 10번의 반복 훈련을 하겠다
```
## 테스트 데이타 생성 및 결과 확인
새로운 입력값을 만들어 결과값을 확인합니다.
```
X_test = np.arange(11, 16, 1).reshape(-1, 1)
pred = model.predict(X_test)  # 예상값 생성 (input 에 대한 예상값 설정
print(X_test)
print(pred)
```