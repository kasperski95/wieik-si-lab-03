import keras
import keras.preprocessing as keras_utils
import keras.applications.vgg16 as vgg16
import numpy as np

image_file_base_names = ["dog", "doggo", "muffin"]

for image_file_base_name in image_file_base_names:
    img = keras_utils.image.load_img(f"resources/{image_file_base_name}.jpg", target_size=(224, 224))
    x = keras_utils.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    vgg16_input = vgg16.preprocess_input(x)
    modelVGG16 = vgg16.VGG16(weights="imagenet", include_top=True)
    vgg16_features = modelVGG16.predict(vgg16_input)
    results = vgg16.decode_predictions(vgg16_features, top=3)[0]

    print(image_file_base_name)
    for result in results:
        print(f"{(result[2]*100):.2f}% - {result[1]}")
    print()
