# 영화 리뷰
참조 : https://tensorflow.blog/케라스-딥러닝/3-4-영화-리뷰-분류-이진-분류-예제/


## IMDB 데이터셋 
MNIST 데이터셋처럼 IMDB 데이터셋도 케라스에 포함되어 있습니다. 이 데이터는 전처리되어 있어 각 리뷰(단어 시퀀스)가 숫자 시퀀스로 변환되어 있습니다. 여기서 각 숫자는 사전에 있는 고유한 단어를 나타냅니다.
```
from tensorflow.keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000)
```

## 데이타 준비
IMDB 데이터셋(리스트)을  텐서로 바꾸는 방법은 두 가지가 있습니다.
- 같은 길이가 되도록 리스트에 padding을 추가하고 (samples, sequence_length) 크기의 정수 텐서로 변환합니다. 그 다음 이 정수 텐서를 다룰 수 있는 층을 신경망의 첫 번째 층으로 사용
- 리스트를 one-hot encoding하여 0과 1의 벡터로 변환합니다. 예를 들어 시퀀스 [3, 5]를 인덱스 3과 5의 위치는 1이고 그 외는 모두 0인 10,000차원의 벡터로 각각 변환합니다.그다음 부동 소수 벡터 데이터를 다룰 수 있는 Dense 층을 신경망의 첫 번째 층으로 사용

여기서는 두 번째 방식을 사용
## 정수 시퀀스를 이진 행렬로 인코딩
```
def vectorize_sequences(sequences, dimension=10000):
    # 크기가 (len(sequences), dimension)이고 모든 원소가 0인 행렬을 만듭니다.
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1. # results[i]에서 특정 인덱스의 위치를 1로 만듭니다.
    return results

x_train = vectorize_sequences(train_data) # 훈련 데이터를 벡터로 변환합니다.
x_test = vectorize_sequences(test_data) # 테스트 데이터를 벡터로 변환합니다
```

## 레이블을 벡터(1D 텐서)로 변경
```
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
```
위와 같이 처리하면 특정 문장(댓글)에서 사용된 단어들(x_train)을 구할 수 있고 그 단어들의 집합이 긍정인지 부정인지(y_train 1 or 0) 인지 알 수가 있다. \
이것을 이용하여 신경망 모델을 만들어 보자

## Training Set 준비
```
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
```


## 신경망 모델 만들기
### 모델 정의
```
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```
### 모델 컴파일하기 1-1
```
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

### 모델 컴파일하기 1-2
옵티마이저의 매개변수 변경하기
```
from tensorflow.keras import optimizers
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

### 모델 컴파일하기 1-3
손실과 측정을 함수 객체로 지정하기
```
from tensorflow.keras import losses
from tensorflow.keras import metrics

model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])
```


## Training
```
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val)
```

![alt train_images[0]](assets/images/movie-review-01-01.png)
![alt train_images[0]](assets/images/movie-review-01-02.png)