import cv2
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model('vegetable_scanner_model.h5')


vegetable_classes = ['bean', 'bitter_gourd', 'bottle_gourd', 'brinjal', 'broccoli', 
                     'cabbage', 'capsicum', 'carrot', 'cauliflower', 'cucumber', 
                     'papaya', 'potato', 'pumpkin', 'radish', 'tomato']


cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fejl ved indlæsning af billede fra kameraet.")
            break

    
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0  
        img = np.expand_dims(img, axis=0) 

       
        predictions = model.predict(img)
        predicted_class = vegetable_classes[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

       
        cv2.putText(frame, f"{predicted_class}: {confidence:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Vegetable Classifier', frame)

  
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"En fejl opstod: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
