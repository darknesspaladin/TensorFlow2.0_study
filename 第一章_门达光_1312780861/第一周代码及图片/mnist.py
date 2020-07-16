import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import  layers
import numpy as np
import matplotlib.pyplot as plt

# laoding mnist dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.dtype)
print(train_images.shape)
print(train_labels.shape)
# print(len(test_labels))
# show the dataset picture
class_names = ['0','1','2','3','4',
               '5','6','7','8','9']

plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid()
    plt.imshow(train_images[i], cmap = plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

train_images = train_images.reshape(60000, 784).astype("float32") / 255.0
# print(train_labels[1])
test_images = test_images.reshape(10000,784).astype("float32") / 255.0

# buliding network structure
mnist_inputs = keras.Input(shape=(784,), name="img")
x = layers.Dense(256, activation="relu")(mnist_inputs)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dense(20, activation="relu")(x)
mnist_outputs = layers.Dense(10)(x)

model = keras.Model(mnist_inputs, mnist_outputs, name="my_mnist")

model.compile(
              optimizer="adam",
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['acc'],
)
# Start train
model.fit(train_images, train_labels, epochs=10, validation_split=0.1)

test_loss,test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy',test_acc)

probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

train_images = train_images.reshape(60000, 28, 28).astype("float32") / 255.0
# print(train_labels[1])
test_images = test_images.reshape(10000,28, 28).astype("float32") / 255.0

def plot_image(i,predictions_arry,true_label,img):
    predictions_arry, true_label, img = predictions_arry, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_arry)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_arry),
                                         class_names[true_label]),
                                         color = color)
def plot_value_array(i,predictions_array,true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    thisplot = plt.bar(range(10), predictions_array, color = "#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

#yan zheng yu ce
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()


#shi yong xun lian hao de mo xing cong  ce shi ji zhong zhua qu tu xiang
img = test_images[2]
img = img.reshape(1,784).astype("float32")

predictions_test = probability_model.predict(img)
print(predictions_test)

plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(1, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()


