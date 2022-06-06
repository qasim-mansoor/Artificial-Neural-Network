import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps 
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

ALPHABETS = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
# def assign_label(flower_type):
#     return flower_type

X=[]
Z=[]

def make_train_data(letter,DIR):
    for img in os.listdir(DIR):
        label=letter
        path = os.path.join(DIR,img)
        img = Image.open(path)
        img_gray = ImageOps.grayscale(img)
        img_gray = img_gray.resize((100,100))
        X.append(np.array(img_gray))
        Z.append(str(label))


for letter in ALPHABETS:
    make_train_data(letter, 'Alphabets\{}'.format(letter))

# print(X[0])

labels = LabelEncoder()
Y = labels.fit_transform(Z)
# print(Y)
X = np.array(X)
# print(X.shape)
X = X/255

# print(Y.shape)
# print(X.shape)

train_images,test_images , train_labels, test_labels = train_test_split(X,Y, test_size=0.13, random_state=42)

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
# plt.show()


model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(100, 100)),
        tf.keras.layers.Dense(1000,activation='relu'),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dense(250, activation='relu'),
        tf.keras.layers.Dense(26)])

model.compile(optimizer='SGD',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images,train_labels, epochs=100)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
# print(predictions)

actual_output=[]

for i in range(len(test_labels)):
    actual_output.append(np.argmax(predictions[i]))

confusion = confusion_matrix(test_labels,actual_output)
print('Confusion Matrix: \n', confusion)
