import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
print(x_train.shape)
print(y_train.shape)
x_test = x_test.astype("float32") / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test,10)

inputs = keras.Input(shape=(32, 32, 3), name="img")
x = layers.Conv2D(64, 3, activation="relu")(inputs)
x = layers.Conv2D(64, 3, activation="relu")(x)
x = layers.Dropout(0.2)(x)
block_1_output = layers.MaxPooling2D(3)(x)

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.Dropout(0.1)(x)
block_2_output = layers.add([x,block_1_output])

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.Dropout(0.1)(x)
block_3_output = layers.add([x,block_2_output])

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_3_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs, outputs, name="toy_ResNet")
model.summary()
model.compile(
    optimizer = keras.optimizers.RMSprop(1e-3),
    loss = keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=128, epochs=20, validation_split=0.2)
test_loss,test_acc = model.evaluate(x_test, y_test, verbose=2)
model.save_weights('/home/mdg/PycharmProjects/untitled/Mini_Resnet_saveweights')
print('\nTest accuracy',test_acc)
# ResNet模型
# # 随机批处理数据
# BATCH_SIZE = 32
# IMG_SIZE = 32
#
# # 使用MobileNet_V2模型，从预训练的卷积开始创建基本模型
#
# # 加载预训练模型
# IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
# #创建基本模型,不包含mobilet的分类层
# model = tf.keras.applications.ResNet50(
#     input_shape=IMG_SHAPE,
#     include_top=False,
#     weights='imagenet'
# )
#
# model.trainable = False
# model.summary()
# keras.utils.plot_model(model, "mini_resnet.png", show_shapes=True)
# feature_map = model(x_train)
# #设置损失函数优化器，编译模型
# model.compile(
#     optimizer = keras.optimizers.RMSprop(1e-3),
#     loss = keras.losses.CategoricalCrossentropy(from_logits=True),
#     metrics=["accuracy"],
# )
# Pooling_layer = tf.keras.layers.GlobalAveragePooling2D()
# ########################feature_extracter =
#将数据集限制在钱1000个样本中，限制时间

#查看冻结特征提取器时的训练和验证的准确性和学习率曲线
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validition Accuracy')
plt.legend(loc='Lower right')
plt.ylabel("Accuracy")
plt.ylim([min(plt.ylim()), 1])
plt.title('Traing and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entrppy')
plt.title([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()