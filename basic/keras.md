# Keras 
참조 : https://tensorflow.blog/케라스-딥러닝/
## keras 
[파이썬 딥러닝 라이브러리] 이다 \
이외에 대표적 딥러닝 프레임워크로는 텐서플로, 카페, 씨아노, 토치, 파이토치 등이 있다
## 케라스 맛보기
- 입력 텐서와 타깃 텐서로 이루어진 훈련 데이터 정의
- 입력과 타깃을 매핑하는 층으로 이루어진 네트워크(또는 모델)를 정의 (Sequential or 함수형 API)
- 손실함수, 옵티마이저, 모니터링을 위한 측정 지표를 선택하여 학습 과정을 설정
- 훈련 데이터에 대해 모델의 fit() 메서드를 반복적으로 호출
## 모델 정의 방법
- Sequential 클래스 : 층을 순서대로 쌓아 올린 네트워크
- 함수형 API: 임의의 구조를 만들 수 있는 비순환 그래프

## Sequential 클래스 예제
```
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(784,)))
model.add(layers.Dense(10, activation='softmax'))
```
첫 layers.Dense에 입력데이터의 크기가 전달
## 함수형 API 예제
```
input_tensor = layers.Input(shape=(784,))
x = layers.Dense(32, activation='relu')(input_tensor)
output_tensor = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs=input_tensor, outputs=output_tensor)
```
모델 구조가 정의되면 컴파일 단계에서 학습 과정 설정
## 컴파일
옵티마이저(optimizer)와 손실 함수(loss), 훈련하는 동안 모니터링하기 위해 필요한 측정 지표를 지정
```
from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='mse',
              metrics=['accuracy'])
```
## 학습하기(training)
입력 데이터의 넘파이 배열을 (그리고 이에 상응하는 타깃 데이터를) 모델의 fit() 메서드에 전달함으로써 학습 과정이 이루어짐
```
model.fit(input_tensor, target_tensor, batch_size=128, epochs=10)
```


