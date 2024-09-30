import tensorflow as tf
import numpy as np
import os


model = tf.keras.models.load_model('vegetable_scanner_model.h5')


test_data_dir = 'testRed'  
img_width, img_height = 224, 224


vegetable_classes = ['bean', 'bitter_gourd', 'bottle_gourd', 'brinjal', 'broccoli', 
                     'cabbage', 'capsicum', 'carrot', 'cauliflower', 'cucumber', 
                     'papaya', 'potato', 'pumpkin', 'radish', 'tomato']


test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=1,
    class_mode='categorical',
    classes=vegetable_classes,
    shuffle=False
)


test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.2f}")


def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_width, img_height))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.resnet_v2.preprocess_input(img_array[np.newaxis, ...])
    
    predictions = model.predict(img_array)
    predicted_class = vegetable_classes[np.argmax(predictions)]
    confidence = np.max(predictions)
    
    return predicted_class, confidence


test_images = [
    os.path.join(test_data_dir, vegetable, os.listdir(os.path.join(test_data_dir, vegetable))[0])
    for vegetable in vegetable_classes
]

for image_path in test_images:
    predicted_class, confidence = predict_image(image_path)
    print(f"Image: {image_path}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")
    print()
