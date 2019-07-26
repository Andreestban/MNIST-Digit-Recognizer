#importing librarairs

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#importing dataset

dataset_train = pd.read_csv("train.csv")

#anaylzing data

sns.countplot(x = dataset_train["label"])

##seperating data

#training data
dataset_train_X = dataset_train.iloc[:, 1:].values
dataset_train_y = dataset_train.iloc[:, 0].values


#reshaping data


#training datra
dataset_train_X = dataset_train_X.reshape(dataset_train_X.shape[0],28,28,1)

#converting images from int to float format
#train set
dataset_train_X = dataset_train_X.astype('float32')

#scaling immages
#training images
dataset_train_X = dataset_train_X/255


#converting varaibels of label

from keras.utils import to_categorical
dataset_train_y = to_categorical(dataset_train_y)

#splitting dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(dataset_train_X,
                                                    dataset_train_y,
                                                    test_size = 0.15,
                                                    random_state = 42)

#appling Convolutional nueral network

#importing librarires

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

#biuldind API

def Build_Model():
    
    #initializing model
    Model = Sequential()
    
    #adding 1st conv,maxpooled, dropout layer
    Model.add(Conv2D(64,
                     (3,3),
                     activation = 'relu',
                     input_shape = (28,28,1)))
    
    Model.add(MaxPooling2D(pool_size = (2,2)))
    Model.add(Dropout(0.20))
    
    #adding 2nd conv 2d layer
    Model.add(Conv2D(64,
                     (3,3),
                     activation = 'relu'))
    
    Model.add(MaxPooling2D(pool_size = (2,2)))
    Model.add(Dropout(0.20))
        
        
        #adding 3rd conv 2d layer
    Model.add(Conv2D(64,
                     (3,3),
                     activation = 'relu'))
        
    Model.add(MaxPooling2D(pool_size = (2,2)))
    Model.add(Dropout(0.20))
        
    #adding flatten 
    Model.add(Flatten())
    
    #adding dense layers
    Model.add(Dense(units = 128,
                    activation = 'relu'))
    #adding output layers
    Model.add(Dense(units = 10,
                    activation = 'softmax'))
    
    #compilong Model
    Model.compile(optimizer = 'adam',
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    
    return Model

#calling API
new_model = Build_Model()

#diagramitic summary
new_model.summary()

#fitting data
history = new_model.fit(x = X_train,
              y = y_train,
              batch_size = 100,
              epochs = 100,
              validation_data = (X_val, y_val))

#ploting graph

#accuracy

plt.title("accuracy plot")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.legend(["X_tain","X_val"], loc = "upper right")
plt.show()

#loss plot

plt.title("loss plot")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["X_tain","X_val"], loc = "upper right")
plt.show()

#retrunung architecture of model
from keras.utils import plot_model
plot_model(new_model,
           to_file = "model_architecture.png",
           show_shapes = True,
           show_layer_names = True)

#stroring history data

train_acc = pd.Series(history.history["acc"],
                      name = "Train_Accuracy")
train_loss = pd.Series(history.history["loss"],
                       name = "Train_loss")

val_acc = pd.Series(history.history["acc"],
                    name = "Validation_Accuracy")

val_loss = pd.Series(history.history["loss"],
                     name = "Validation_Loss")

#joining data
train_results = pd.concat((train_acc,train_loss),
                          axis = 1,
                          join = "outer")

val_results = pd.concat((val_acc,val_loss),
                        axis =1,
                        join = "outer")

#converting to csv

train_results.to_csv("train_results_model.csv")
val_results.to_csv("val_reults_model.csv")

#save model
new_model.save("digit_recognizer_model.h5")