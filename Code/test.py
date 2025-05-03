from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import os

# Load model
model = load_model('model.h5')
print("Model Loaded Successfully")

# Classify function
def classify(img_file):
    try:
        img = image.load_img(img_file, target_size=(256, 256), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # normalize
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        arr = np.array(prediction[0])
        print("Prediction Array:", arr)

        max_prob = arr.argmax()
        classes = ["NONE", "ONE", "TWO", "THREE", "FOUR", "FIVE"]
        result = classes[max_prob]

        print(f"{img_file} → {result}")
    except Exception as e:
        print(f"Error processing {img_file}: {e}")

# Classify all .png images from given path
path = 'D:/MasterClass/Artificial_Intelligence/Day13/Dataset/val/TWO'
files = [os.path.join(r, file)
         for r, _, f in os.walk(path)
         for file in f if file.endswith('.png')]

for f in files:
    classify(f)
    print('\n')
