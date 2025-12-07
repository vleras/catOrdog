import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf

model = tf.keras.models.load_model("./saved_model/cnn_cat_dog.h5")

img_path = "./dataset/single_prediction/cat_or_dog_1.jpg"

test_img = image.load_img(img_path, target_size=(64, 64))
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img, axis=0)

prediction = model.predict(test_img)[0][0]

if prediction >= 0.5:
    print("Prediction: Dog")
else:
    print("Prediction: Cat")
