import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# colocar la direccion de memoria donde se encuentra el modelo y sus pesos
longitud, altura = 100, 100
modelo = ""
pesos = ""

#se carga el modelo
cnn = load_model(modelo)
cnn.load_weights(pesos)

#funciona encargada de predecir
def predict(file):
  x = load_img(file, target_size=(longitud,altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  arreglo = cnn.predict(x) ##[[1,0]]
  resultado = arreglo[0]
  respuesta = np.argmax(resultado)  

  if (respuesta == 0):
    print ("Con barbijo")
  elif (respuesta == 1):
    print ("Sin barbijo")
  


#prediccion 
predict("")