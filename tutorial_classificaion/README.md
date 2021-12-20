# Basic image classification
원본: https://www.tensorflow.org/tutorials/keras/classification
```
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```
mnist 로부터 데이타를 불러온다.
```
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```
- train_images.shape: (60000, 28, 28) - 60,000 개의 이미지가 28 X 28 픽셀로 되어 있음
- len(train_labels)): 60000
- train_labels: [9 0 0 ... 3 0 5] - 각각의 라벨은 0에서 9의 숫자로 되어 있음
- test_images.shape: (10000, 28, 28) - 10,000 개의 TEST 이미지가 28 X 28 픽셀로 되어 있음
- len(test_labels): 10000

클래스 명을 정해준다.
```
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```
train image중 첫번째 이미지를 보여준다.
```
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
```
![alt train_images[0]](images/1.png)
데이타는 training 전에 전처리과정을 거쳐야 한다. 위의 그림에서 보여지는 것처럼 이미지의 pixel은 0~255 이다. 
따라서 아래와 같이 이 값들을 0에서 1로 변경하여야 한다.
255로 나누어 보겠다.
```
train_images = train_images / 255.0
test_images = test_images / 255.0
```
이미지가 제대로 되었는지 25개의 이미지를 보자
```
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
```
![alt train_images[0]](images/2.png)

## 모델구성
### 층설정
```
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
```
이 네트워크의 첫 번째 층인 tf.keras.layers.Flatten은 2차원 배열(28 x 28 픽셀)의 이미지 포맷을 28 * 28 = 784 픽셀의 1차원 배열로 변환합니다. 이 층은 이미지에 있는 픽셀의 행을 펼쳐서 일렬로 늘립니다. 이 층에는 학습되는 가중치가 없고 데이터를 변환하기만 합니다.
픽셀을 펼친 후에는 두 개의 tf.keras.layers.Dense 층이 연속되어 연결됩니다. 이 층을 밀집 연결(densely-connected) 또는 완전 연결(fully-connected) 층이라고 부릅니다. 첫 번째 Dense 층은 128개의 노드(또는 뉴런)를 가집니다. 두 번째 (마지막) 층은 10개의 노드의 소프트맥스(softmax) 층입니다. 이 층은 10개의 확률을 반환하고 반환된 값의 전체 합은 1입니다. 각 노드는 현재 이미지가 10개 클래스 중 하나에 속할 확률을 출력합니다.

### 모델 컴파일
모델을 훈련할 준비가 되기 전에 몇 가지 설정이 더 필요합니다. 다음은 모델의 컴파일 단계에서 추가됩니다.
- 손실 함수 - 훈련 중 모델이 얼마나 정확한지 측정합니다. 모델을 올바른 방향으로 "조정"하려면 이 함수를 최소화해야 합니다.
- 옵티마이저 - 모델이 인식하는 데이터와 해당 손실 함수를 기반으로 모델이 업데이트되는 방식입니다.
- 메트릭 — 훈련 및 테스트 단계를 모니터링하는 데 사용됩니다. 다음 예에서는 올바르게 분류된 이미지의 비율인 정확도를 사용합니다.

```
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```
### 모델 훈련
신경망 모델을 훈련하려면 다음 단계가 필요합니다.

- 훈련 데이터를 모델에 주입합니다-이 예에서는 train_images와 train_labels 배열입니다.
- 모델이 이미지와 레이블을 매핑하는 방법을 배웁니다.
- 테스트 세트에 대한 모델의 예측을 만듭니다-이 예에서는 test_images 배열입니다. 이 예측이 test_labels 배열의 레이블과 맞는지 확인합니다.
- 예측이 test_labels 배열의 레이블과 일치하는지 확인합니다.
#### 모델 피드
```
model.fit(train_images, train_labels, epochs=10)
```

```
Epoch 1/10
1875/1875 [==============================] - 2s 827us/step - loss: 0.4936 - accuracy: 0.8266
Epoch 2/10
1875/1875 [==============================] - 2s 816us/step - loss: 0.3723 - accuracy: 0.8673
Epoch 3/10
1875/1875 [==============================] - 2s 893us/step - loss: 0.3344 - accuracy: 0.8770
.......................
```
모델이 훈련되면서 손실과 정확도 지표가 출력됩니다. 이 모델은 훈련 세트에서 약 0.88(88%) 정도의 정확도를 달성합니다.
###  정확도평가
다음으로, 모델이 테스트 데이터세트에서 작동하는 방식을 비교합니다.
```
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
```

```
313/313 - 0s - loss: 0.3232 - accuracy: 0.8864 - 274ms/epoch - 875us/step
Test accuracy: 0.8863999843597412
```
테스트 세트의 정확도가 훈련 세트의 정확도보다 조금 낮습니다. 훈련 세트의 정확도와 테스트 세트의 정확도 사이의 차이는 과대적합(overfitting) 때문입니다. 과대적합은 머신러닝 모델이 훈련 데이터보다 새로운 데이터에서 성능이 낮아지는 현상을 말합니다.

### 예측하기
훈련된 모델을 사용하여 일부 이미지에 대한 예측을 수행할 수 있습니다. 모델의 선형 출력, 로짓. 소프트맥스 레이어를 연결하여 로짓을 해석하기 쉬운 확률로 변환합니다.
```
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
```

```
predictions = probability_model.predict(test_images)
```
여기서는 테스트 세트에 있는 각 이미지의 레이블을 예측했습니다. 첫 번째 예측을 확인해 보죠:
```
print(predictions[0])
```

```
[1.0952453e-05 8.7558591e-09 1.1588849e-06 1.0692807e-08 6.1164324e-06
 1.1773867e-03 3.2505698e-06 1.8590044e-02 1.3603465e-06 9.8020971e-01]
```
이 예측은 10개의 숫자 배열로 나타납니다. 이 값은 10개의 옷 품목에 상응하는 모델의 신뢰도(confidence)를 나타냅니다. 가장 높은 신뢰도를 가진 레이블을 찾아보죠:
```
print(np.argmax(predictions[0]))
```
```
9
```
모델은 이 이미지가 앵클 부츠(class_name[9])라고 가장 확신하고 있습니다. 이 값이 맞는지 테스트 레이블을 확인해 보죠:
```
print(test_labels[0])
```
```
9
```
10개 클래스에 대한 예측을 모두 그래프로 표현해 보겠습니다.
```
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
```
### 예측 확인
훈련된 모델을 사용하여 일부 이미지에 대한 예측을 수행할 수 있습니다.

0번째 원소의 이미지, 예측, 신뢰도 점수 배열을 확인해 보겠습니다.
```
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
```
![alt train_images[0]](images/3.png)
```
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
```
![alt train_images[0]](images/4.png)
몇 개의 이미지의 예측을 출력해 보죠. 올바르게 예측된 레이블은 파란색이고 잘못 예측된 레이블은 빨강색입니다. 숫자는 예측 레이블의 신뢰도 퍼센트(100점 만점)입니다. 신뢰도 점수가 높을 때도 잘못 예측할 수 있습니다.

```
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
```
![alt train_images[0]](images/5.png)
### 훈련된 모델 사용하기

```
# Grab an image from the test dataset
img = test_images[1]
print(img.shape)
```
```
(28, 28)
```
tf.keras 모델은 한 번에 샘플의 묶음 또는 배치(batch)로 예측을 만드는데 최적화되어 있습니다. 하나의 이미지를 사용할 때에도 2차원 배열로 만들어야 합니다:
```
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))
print(img.shape)
```
```
(1, 28, 28)
```
이제 이 이미지의 예측을 만듭니다
```
predictions_single = probability_model.predict(img)
print(predictions_single)

```
```
[[2.6491098e-06 4.3829015e-14 9.9893981e-01 4.4089024e-10 3.0652853e-04
  1.8657725e-13 7.5100036e-04 1.6970431e-17 8.0927216e-11 3.9574041e-13]]
```
```
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
```
![alt train_images[0]](images/6.png)

