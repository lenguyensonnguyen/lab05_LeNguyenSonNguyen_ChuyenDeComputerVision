from __future__ import absolute_import, division, print_function
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras.backend as K
import numpy as np
import PIL.Image as Image
import os
import cv2
from mtcnn.mtcnn import MTCNN
from tensorflow.keras import layers

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2"


def feature_extractor(x):
    feature_extractor_module = hub.Module(feature_extractor_url)
    return feature_extractor_module(x)


IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))
current_patch = os.getcwd()
image_data = image_generator.flow_from_directory(str(current_patch + "/FaceDB"),
                                                 target_size=IMAGE_SIZE)

print("Mot vai thong so ve hinh anh: ")
for image_batch, label_batch in image_data:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break

features_extractor_layer = layers.Lambda(feature_extractor, input_shape=IMAGE_SIZE + [3])
features_extractor_layer.trainable = False

model = tf.keras.Sequential([
    features_extractor_layer,
    layers.Dense(image_data.num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

print("Ket qua thong so cua model")
model.summary()

sess = K.get_session()
init = tf.global_variables_initializer()
sess.run(init)


class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])


steps_per_epoch = image_data.samples // image_data.batch_size
print("step per epoch: ", steps_per_epoch)
batch_stats = CollectBatchStats()
model.fit((item for item in image_data), epochs=10,
          steps_per_epoch=steps_per_epoch,
          callbacks=[batch_stats])

label_names = sorted(image_data.class_indices.items(), key=lambda pair: pair[1])
label_names = np.array([key.title() for key, value in label_names])
print(label_names)

# test hình thật
detector = MTCNN()

image_test = cv2.imread("test.png")
image_test_for_paint = cv2.imread("test.png")

result_dec = detector.detect_faces(image_test)

for result in result_dec:
    bounding_box = result['box']
    img_crop = image_test[bounding_box[1]:bounding_box[1] + bounding_box[3],
               bounding_box[0]:bounding_box[0] + bounding_box[2]]

    img_crop = cv2.resize(img_crop, (224, 224))
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
cv2.imwrite("test_draw.jpg", image_test_for_paint)
plt.show()
