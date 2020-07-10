import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation, Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

#Colcar la direccion de memoria donde se encuentra el dataset de entrenamiento y de validacion correspondientemente.
K.clear_session()
data_entrenamiento = '/content/drive/My Drive/I^2GPF/DataSetBarbijos/Entrenamiento'
data_validacion = '/content/drive/My Drive/I^2GPF/DataSetBarbijos/Validacion'


#parametros
epocas = 25
batch_size = 75
pasos = 60
pasos_validacion = 15
clases = 2
lr = 0.0005


#Preprocesamiento de imagenes
entrenamiento_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.3,
    width_shift_range=.15,
    height_shift_range=.15,
    zoom_range = 0.3,
    horizontal_flip = True,
)

validacion_datagen = ImageDataGenerator(
    rescale = 1./255,
)

imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    shuffle = True,
    target_size = (100,100),
    batch_size = batch_size,
    class_mode = 'categorical'
)

imagen_validacion = validacion_datagen.flow_from_directory(
    data_validacion,
    target_size = (100,100),
    batch_size = batch_size,
    class_mode = 'categorical'
)



# Red neuronal 
cnn = Sequential()

cnn.add(Convolution2D(16,(3,3), padding='same',input_shape=(100,100,3),activation='relu'))

cnn.add(MaxPooling2D(pool_size= (2,2)))

cnn.add(Dropout(0.2))

cnn.add(Convolution2D(32,(3,3), padding='same',activation='relu'))

cnn.add(MaxPooling2D(pool_size= (2,2)))

cnn.add(Convolution2D(64,(3,3), padding='same',activation='relu'))

cnn.add(MaxPooling2D(pool_size= (2,2)))

cnn.add(Dropout(0.2))

cnn.add(Flatten())

cnn.add(Dense(512, activation='relu', ))

cnn.add(Dropout(0.5))

cnn.summary()

cnn.add(Dense(clases, activation= 'softmax'))

cnn.compile(loss='categorical_crossentropy', optimizer = optimizers.Adam(lr = lr), metrics=['accuracy'])



#Entrenar y guardar
#En dir colocar la direccion de memoria donde se creara el directorio
#en save y save weigths colocar la direccion de memoria donde se guardara el modelo y sus pesos.
cnn.fit(imagen_entrenamiento, steps_per_epoch= pasos, epochs= epocas, validation_data= imagen_validacion, validation_steps=pasos_validacion)

dir = ''

if not os.path.exists(dir):
  os.mkdir(dir)

cnn.save('')
cnn.save_weights('')