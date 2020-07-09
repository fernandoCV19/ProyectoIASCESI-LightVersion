#librerias previas

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

#preparamos el modelo que distingue barbijos
longitud, altura = 100, 100
modelo = "ModeloReconocedorBarbijos/modeloV2.h5"
pesos = "ModeloReconocedorBarbijos/pesosV2.h5"

#cargamos el modelo y sus pesos
cnn = load_model(modelo)
cnn.load_weights(pesos)

#preparamos el modelo de haarcascade para reconocimiento facial
cap = cv2.VideoCapture(0)
faceClassif = cv2.CascadeClassifier('Haarcascade/haarcascade_eye_tree_eyeglasses.xml')
#faceClassif = cv2.CascadeClassifier('Haarcascade/haarcascade_rigtheye_2splits.xml')


#funcion que predice si alguien esta usando un barbijo o no
def predict(x):
  arreglo = cnn.predict(x) ##[[1,0]]
  resultado = arreglo[0]
  respuesta = np.argmax(resultado)  

  if (respuesta == 0):
    #con barbijo
    return True
  elif (respuesta == 1):
    #sin barbijo
    return False


#analizis en tiempo real de un rostro
while True:
  ret,frame = cap.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  faces = faceClassif.detectMultiScale(gray, 1.3, 5)

  for (x,y,w,h) in faces:
    # recortar imagen acorde al rostro
    rostro = frame[y:y+h,x:x+w]
    # redimensionar el rostro
    rostro = cv2.resize(rostro,(100,100))
    rostro = img_to_array(rostro)
    rostro = np.expand_dims(rostro, axis=0)
    rostro = rostro/255.0
    
    # procesar imagen en la otra red neuronal
    # si la persona lleva barbijo mostrar el rectangulo que rodea la cara verde
    # si la persona no lleva barbijo mostrar el rectangilo que rodea el rostro en rojo
    # el modelo sirve con multiples rostros al mismo tiempo
    if (predict(rostro)):
        #eye tree eyeglasse
        cv2.rectangle(frame, (x-w,y-(3*h)),(x+w+(4*w),y+(5*h)),(0,255,0),2)
        #rigth eye
        #cv2.rectangle(frame, (x-w,y-(2*h)),(x+w+(3*w),y+(4*h)),(0,255,0),2)
    else:
        #eye tree eyeglasse
        cv2.rectangle(frame, (x-w,y-(3*h)),(x+w+(4*w),y+(5*h)),(0,0,255),2)
        #rigth eye
        #cv2.rectangle(frame, (x-w,y-(2*h)),(x+w+(3*w),y+(4*h)),(0,0,255),2)
    
  cv2.imshow('frame',frame)
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    cap.release()
    cv2.destroyAllWindows()
    break