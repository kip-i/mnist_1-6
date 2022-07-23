import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
"""
mnist: 手書き数字画像データセット
Sequential: Kerasを用いてモデルを生成するためのモジュール
Dense: 全結合層のレイヤモジュール
Dropout: ドロップアウトモジュール
Conv2D: 2次元畳み込み層のモジュール
MaxPool2D: 2次元最大プーリング層のモジュール
Flatten: 入力を平滑化するモジュール
"""

batch_size = 128
num_classes = 7
epochs = 80

# 入力画像の大きさ(行と列）
img_rows, img_cols = 28, 28

# 学習データとテストデータに分割したデータ
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 自作1~6以外のデータ削除
for i in reversed(range(len(y_train))):
    y_int=int(y_train[i])
    if (1>y_int)or(y_int>6):
      plt.imshow(x_train[i])
      x_train=np.delete(x_train,i,0)
      y_train=np.delete(y_train,i,0)

for i in reversed(range(len(y_test))):
    y_int=int(y_test[i])
    if (1>y_int)or(y_int>6):
      plt.imshow(x_test[i])
      x_test=np.delete(x_test,i,0)
      y_test=np.delete(y_test,i,0)

# backendがTensorFlowとTheanoで配列のshapeが異なるために2パターン記述
if K.image_data_format() == 'channels_first':
    # 1次元配列に変換
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    # 1次元配列に変換
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 入力データの各画素値を0-1の範囲で正規化(学習コストを下げるため)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# ラベルをバイナリベクトルとして扱う
# Kerasはラベルを数値ではなく、0or1を要素に持つベクトルで扱うため
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

# CNNネットワークの構築
# 畳み込みフィルターのサイズ(kernel_size)は3×3。整数か単一の整数からなるタプル/リストで指定
# https://keras.io/ja/layers/convolutional/
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

# 損失関数,最適化関数,評価指標を指定してモデルをコンパイル
#ワンホットの場合 loss=keras.losses.categorical_crossentropy
#整数値の場合 loss=keras.losses.sparse_categorical_crossentropy,
model.compile(loss=keras.losses.categorical_crossentropy,
              # optimizer=keras.optimizers.Adadelta(),
              optimizer=tf.optimizers.Adadelta(),
              metrics=['accuracy'])

# モデルの学習
hist = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

#モデルの評価
score=model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('test acc:', score[1])

#学習のグラフ化
epochs = range(1, len(hist.history['accuracy']) + 1)

plt.plot(epochs, hist.history['loss'], label='Training loss', ls='-') #損失値
plt.plot(epochs, hist.history['val_loss'], label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, hist.history['accuracy'],  label='Training acc') #正解率
plt.plot(epochs, hist.history['val_accuracy'], label="Validation acc")
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# モデルを変換
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
#モデルの保存
open("ver2.tflite", "wb").write(tflite_model)