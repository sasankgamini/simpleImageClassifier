import cv2
import os
import numpy as np
import sklearn
import sklearn.model_selection
import keras

datafolder='SomethingOrNothing'
data=[]
labels=[]
folders=['Something','Nothing']
for symbol in folders:
    path = os.path.join(datafolder,symbol)
    images = os.listdir(path)
    for eachImage in images:
        imgarray = cv2.imread(os.path.join(path,eachImage))
        data.append(imgarray)
        if symbol == "Something":
            labels.append(0)
        if symbol == "Nothing":
            labels.append(1)


data = np.array(data)
labels = np.array(labels)

##print(len(data))
##print(len(labels))

train_images,test_images,train_labels,test_labels=sklearn.model_selection.train_test_split(data,labels,test_size=0.1)
train_images=train_images/255
test_images=test_images/255

print(train_images.shape)
print(test_images.shape)
'''
#building the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(100,100,3)),
    keras.layers.Dense(128,activation = 'relu'), #activation if it passes certian threshold(relu: rectified linear unit)
    keras.layers.Dense(2,activation='softmax')
    ])

#compile the model/properties of model
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Train the model
model.fit(train_images, train_labels, epochs=3) #accuracy can't be 100 because then its overfitting(memoriaing images), should be from 80 to 90

#test the model
test_loss, test_acc = model.evaluate(test_images, test_labels) #prediction on the test images
print(test_acc)

model.save('SmthnOrNothinImgClassifier.h5')



'''
