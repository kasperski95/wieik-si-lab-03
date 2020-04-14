import keras
import keras.preprocessing as keras_utils
import keras.applications.resnet50 as resnet50
import numpy as np

image_file_base_names = ["dog", "doggo", "muffin"]

for image_file_base_name in image_file_base_names:
    img = keras_utils.image.load_img(f"resources/{image_file_base_name}.jpg", target_size=(224, 224))
    x = keras_utils.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    modelresnet50 = resnet50.ResNet50(weights="imagenet")
    resnet_input = resnet50.preprocess_input(x)
    resnet_features = modelresnet50.predict(resnet_input)
    results = resnet50.decode_predictions(resnet_features, top=3)[0]

    print(image_file_base_name)
    for result in results:
        print(f"{(result[2]*100):.2f}% - {result[1]}")
    print()
