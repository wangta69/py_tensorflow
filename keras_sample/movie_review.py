import numpy as np

from tensorflow.keras import models
from tensorflow.keras.datasets import imdb
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics


import matplotlib.pyplot as plt

# IMDB 데이터셋 로드하기
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000)

print('train_data[0] ==================')
print(train_data[0]) # [1, 14, 22, 16, .. . 178, 32]
print('train_labels[0] ==================')
print(train_labels[0]) # 1
print('max([max(sequence) for sequence in train_data]) ==================')
print(max([max(sequence) for sequence in train_data]))  # 가장 자주 등장하는 단어 1만 개로 제한했기 때문에 단어 인덱스는 10,000을 넘지 않습니다.
# max([max(sequence) for sequence in train_data]) ==================
# 9999

# # 재미 삼아 이 리뷰 데이터 하나를 원래 영어 단어로 어떻게 바꾸는지 보겠습니다
# word_index = imdb.get_word_index() # word_index는 단어와 정수 인덱스를 매핑한 딕셔너리입니다.
# reverse_word_index = dict(
#     # 정수 인덱스와 단어를 매핑하도록 뒤집습니다.
#     [(value, key) for (key, value) in word_index.items()])
# decoded_review = ' '.join(
#     # 리뷰를 디코딩합니다. 0, 1, 2는 ‘패딩’, ‘문서 시작’, ‘사전에 없음’을 위한 인덱스이므로 3을 뺍니다.
#     [reverse_word_index.get(i - 3, '?') for i in train_data[0]])

# 데이터 준비 (preprocess)
def vectorize_sequences(sequences, dimension=10000):
    # 크기가 (len(sequences), dimension)이고 모든 원소가 0인 행렬을 만듭니다.
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # results[i]에서 특정 인덱스의 위치를 1로 만듭니다.
    return results

x_train = vectorize_sequences(train_data) # 훈련 데이터를 벡터로 변환합니다.
x_test = vectorize_sequences(test_data)  # 테스트 데이터를 벡터로 변환합니다
# print('x_train[0]) ==================')
# print(x_train[0])  # [0. 1. 1. ... 0. 0. 0.]


# 레이블을 벡터로 변경
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
print('train_labels', train_labels)
print('y_train', y_train)


# 신경망 모델 만들기
## 모델 정의하기
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

## 모델 컴파일하기
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# ## 옵티마이저 설정하기
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# ## 손실과 측정을 함수 객체로 지정
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#               loss=losses.binary_crossentropy,
#               metrics=[metrics.binary_accuracy])

## 훈련 검증
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

## 모델 훈련하기
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['acc'])

model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# model.fit() 메서드는 History 객체를 반환합니다. 이 객체는 훈련하는 동안 발생한 모든 정보를 담고 있는 딕셔너리인 history 속성을 가지고 있습니다. 한번 확인해 보죠.
history_dict = history.history
print(history_dict.keys())


## 훈련과 검증 손실 그리기
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')  # ‘bo’는 파란색 점을 의미합니다.
plt.plot(epochs, val_loss, 'b', label='Validation loss') # ‘b’는 파란색 실선을 의미합니다.
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

## 훈련과 검증 정확도 그리기
plt.clf() # 그래프를 초기화합니다.
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

## 모델을 처음부터 다시 훈련하기
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

print(results)

print(model.predict(x_test))



