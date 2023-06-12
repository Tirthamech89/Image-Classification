from keras.models import Sequential
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Dense
import pandas as pd

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Activation
import argparse 
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

import os
import tarfile

import random
SEED = 99
def random_seed(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    #torch.manual_seed(SEED)
    #torch.cuda.manual_seed(SEED)
    #torch.cuda.manual_seed_all(SEED)
    #torch.backends.cudnn.deterministic = True
    tf.random.set_seed(SEED)
random_seed(SEED)



import optuna
from optuna import Trial
from optuna.samplers import TPESampler


train=pd.read_csv('/root/MH_TLDCC/train.csv')
test=pd.read_csv('/root/MH_TLDCC/test.csv')
image_path='/root/MH_TLDCC/train/'
image_path_test='/root/MH_TLDCC/test/'

train_img=[]
for i in range(len(train)):
    temp_img=image.load_img(image_path+train['Image'][i],target_size=(224,224))
    temp_img=image.img_to_array(temp_img)
    train_img.append(temp_img)

train_img=np.array(train_img)
train_img=preprocess_input(train_img)

test_img=[]
for i in range(len(test)):
    temp_img=image.load_img(image_path_test+test['Image'][i],target_size=(224,224))
    temp_img=image.img_to_array(temp_img)
    test_img.append(temp_img)

test_img=np.array(test_img)
test_img=preprocess_input(test_img)

model = ResNet50(weights='imagenet', include_top=False)

features_train=model.predict(train_img)
features_test=model.predict(test_img)

dim1 = features_train.shape[1]*features_train.shape[2]*features_train.shape[3]

train_x=features_train.reshape(train.shape[0],dim1)
train_y=np.asarray(train['Label'])

test_x=features_test.reshape(test.shape[0],dim1)

train_y=pd.get_dummies(train_y)
train_y=np.array(train_y)

def objective(trial):
    keras.backend.clear_session()
    X_train, X_valid, Y_train, Y_valid = train_test_split(train_x,train_y,test_size=0.3, random_state=42)
    val_ds = (X_valid,Y_valid)
    #optimum number of hidden layers
    n_layers = trial.suggest_int('n_layers', 1, 3)
    model = keras.Sequential()
    
    for i in range(n_layers):
        #optimum number of hidden nodes
        num_hidden = trial.suggest_int(f'n_units_l{i}', 100, 1800, log=True)
        #optimum activation function
        model.add(keras.layers.Dense(num_hidden, input_shape=(dim1,),
                               activation=trial.suggest_categorical(f'activation{i}', ['relu', 'linear','swish','sigmoid'])))
        #optimum dropout value
        model.add(keras.layers.Dropout(rate = trial.suggest_float(f'dropout{i}', 0.1, 0.6))) 
        
    model.add(keras.layers.Dense(10,activation=tf.keras.activations.sigmoid)) #output Layer

    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=1,min_lr=1e-05,verbose=0)
    early_stoping = EarlyStopping(monitor="val_loss",min_delta=0.001,patience=5,verbose=0,mode="auto", baseline=None,restore_best_weights=True)
    model.compile(loss='binary_crossentropy',metrics='categorical_crossentropy', optimizer='Adam')
    #optimum batch size
    histroy = model.fit(X_train,Y_train, validation_data=val_ds,epochs=200,callbacks=[reduce_lr,early_stoping],verbose=0,
                       batch_size=trial.suggest_int('size', 8, 128))
    return min(histroy.history['val_loss'])

if __name__ == "__main__":
    sampler = TPESampler(seed=11062023)
    study = optuna.create_study(direction="minimize",sampler=sampler)
    study.optimize(objective, n_trials=50, timeout=1200)
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))

print(" Params: ")
for key, value in trial.params.items():
    print("{}: {}".format(key, value))   
    
def wider_model():
    model = keras.Sequential()
    model.add(Dense(1050, input_dim=dim1, activation='swish',kernel_initializer='uniform'))
    keras.layers.core.Dropout(0.5454650439678453, noise_shape=None, seed=None)

    model.add(Dense(123,input_dim=1050,activation='swish'))
    keras.layers.core.Dropout(0.1379020569045911, noise_shape=None, seed=None)

    # model.add(Dense(150,input_dim=500,activation='sigmoid'))
    # keras.layers.core.Dropout(0.2, noise_shape=None, seed=None)

    model.add(Dense(units=10))
    model.add(Activation('softmax'))
    
    return model
    

# def wider_model():
    
#     model = keras.Sequential()
#     model.add(Dense(1000, input_dim=dim1, activation='relu',kernel_initializer='uniform'))
#     keras.layers.core.Dropout(0.3, noise_shape=None, seed=None)

#     model.add(Dense(500,input_dim=1000,activation='sigmoid'))
#     keras.layers.core.Dropout(0.4, noise_shape=None, seed=None)

#     model.add(Dense(150,input_dim=500,activation='sigmoid'))
#     keras.layers.core.Dropout(0.2, noise_shape=None, seed=None)

#     model.add(Dense(units=10))
#     model.add(Activation('softmax'))

#     #model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

#     return model


skf = KFold(n_splits=5, shuffle=True, random_state=1234)
Final_Subbmission = []
val_loss_print = []
i=1
for train_index, test_index in skf.split(train_x,train_y):
    keras.backend.clear_session()
    print('#################')
    print(i)
    print('#################')
    X_train, X_test = train_x[train_index], train_x[test_index]
    y_train, y_test = train_y[train_index], train_y[test_index]
    model = wider_model()
    val_ds = (X_test,y_test)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=1,min_lr=1e-05,verbose=1)
    
    early_stoping = EarlyStopping(monitor="val_loss",min_delta=0.001,patience=5,verbose=1,mode="auto", baseline=None,restore_best_weights=True)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy',metrics='categorical_crossentropy', optimizer='Adam')
    
    #histroy = model.fit(X_train,y_train, validation_data=val_ds,epochs=30,callbacks=early_stoping,verbose=1,batch_size=30)
    histroy = model.fit(X_train,y_train, validation_data=val_ds,epochs=30,callbacks=[reduce_lr,early_stoping],verbose=1,batch_size=25)
    
    
    print(min(histroy.history['val_loss']))
    val_loss_print.append(min(histroy.history['val_loss']))
    Test_seq_pred = model.predict(test_x)
    Final_Subbmission.append(Test_seq_pred)
    i=i+1

out = np.mean(Final_Subbmission, axis=0)
class_labels = np.argmax(out, axis=1)
class_labels_dt=pd.DataFrame(class_labels)
class_labels_dt.columns=['Label']

class_labels_dt.to_csv('class_labels_dt_adv_v2.csv',index=False)


























