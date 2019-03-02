from __future__ import absolute_import, division, print_function
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import PIL.Image as Image
import os
import cv2
from mtcnn.mtcnn import MTCNN
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, \
    MaxPooling2D

from keras.models import Model, Sequential

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)

current_patch = os.getcwd()
image_data = image_generator.flow_from_directory(str(current_patch + "/FaceDB"))

print("Mot vai thong so ve hinh anh: ")
for image_batch, label_batch in image_data:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break

nb_classes = image_data.num_classes

model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2nd Convolution layer
model.add(Conv2D(128, (5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 3rd Convolution layer
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 4th Convolution layer
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flattening
model.add(Flatten())

# Fully connected layer 1st layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Fully connected layer 2nd layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(nb_classes, activation='softmax'))

model.compile(
    optimizer=tf.train.AdamOptimizer(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

print("Ket qua thong so cua model")
model.summary()

sess = K.get_session()
init = tf.global_variables_initializer()
sess.run(init)

label_names = sorted(image_data.class_indices.items(), key=lambda pair: pair[1])
label_names = np.array([key.title() for key, value in label_names])
print(label_names)

print("N data: ", image_data.n)
print("batch size: ", image_data.batch_size)

model.fit_generator(generator=image_data,
                    steps_per_epoch=5,
                    epochs=10
                    )

# test hình thật
detector = MTCNN()

image_test = cv2.imread("test.png")
image_test_for_paint = cv2.imread("test.png")

result_dec = detector.detect_faces(image_test)

for result in result_dec:
    bounding_box = result['box']
    img_crop = image_test[bounding_box[1]:bounding_box[1] + bounding_box[3],
               bounding_box[0]:bounding_box[0] + bounding_box[2]]
    # print(img_crop)
    img_crop = cv2.resize(img_crop, (256, 256))
    # plt.imshow(img_crop)

    predict_face = model.predict(img_crop[np.newaxis, ...])
    label_predict = label_names[np.argmax(predict_face, axis=-1)]
    print(predict_face, label_predict)

    cv2.rectangle(image_test_for_paint, (bounding_box[0], bounding_box[1]),
                  (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (255, 0, 0),
                  17)
    cv2.putText(image_test_for_paint,
                label_predict[0],
                (bounding_box[0], bounding_box[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (255, 0, 0),
                10
                )

plt.imshow(image_test_for_paint)
cv2.imwrite("test_draw_conv.jpg", image_test_for_paint)
plt.show()
