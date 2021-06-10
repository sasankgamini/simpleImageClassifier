import numpy as np
import cv2
from keras.models import load_model

model = load_model('SmthnOrNothinImgClassifier.h5')

capture = cv2.VideoCapture(0)
while True:
    _,frame = capture.read()
    cv2.imshow('live',frame)
    frame = cv2.resize(frame,(100,100))
    frameList=[frame]
    frameArray=np.array(frameList)
    frameArray=frameArray/255
    predictions=model.predict(frameArray)
    if np.argmax(predictions[0]) == 0:
        print('something')
    else:
        print('nothing')
    if cv2.waitKey(3) == ord('s'):
        break
capture.release()
cv2.destroyAllWindows()

#something(0), nothing(1)
