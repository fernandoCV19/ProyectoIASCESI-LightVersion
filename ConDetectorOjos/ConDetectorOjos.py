#librerias previas

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

#preparamos el modelo que distingue barbijos
longitud, altura = 100, 100
modelo = "modeloV2.h5"
pesos = "pesosV2.h5"

#cargamos el modelo y sus pesos
cnn = load_model(modelo)
cnn.load_weights(pesos)

#preparamos el modelo de haarcascade para reconocimiento facial
cap = cv2.VideoCapture(0)

#eye tree eyeglasse
faceClassif = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
#rigth eye
#faceClassif = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')



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
   
    #eye tree eyeglasse
    rostro = frame[y-(3*h):y+(5*h),x-w:x+w+(4*w)]
    #rigth eye
    #rostro = frame[y-(2*h):y+(4*h),x-w:x+w+(3*w)]
    
    #verificar posicion correcta del rostro
    
    #eye tree eyeglasse
    rostro2 = gray[y-(3*h):y+(5*h),x-w:x+w+(4*w)]
    #rigth eye
    #rostro2 = gray[y-(2*h):y+(4*h),x-w:x+w+(3*w)]
    
    aux = faceClassif.detectMultiScale(rostro2, 1.3, 5)
    if (len(aux)>1):
        # redimensionar el rostro
        rostro = cv2.resize(rostro,(100,100))
        rostro = img_to_array(rostro)
        rostro = np.expand_dims(rostro, axis=0)
        

        # procesar imagen en la otra red neuronal
        # el modelo sirve con multiples rostros al mismo tiempo
        if (predict(rostro)):
            # si la persona lleva barbijo mostrar el rectangulo que rodea la cara verde
            
            #eye tree eyeglasse
            cv2.rectangle(frame, (x-w,y-(3*h)),(x+w+(4*w),y+(5*h)),(0,255,0),2)
            #rigth eye
            #cv2.rectangle(frame, (x-w,y-(2*h)),(x+w+(3*w),y+(4*h)),(0,255,0),2)
        else:
            # si la persona no lleva barbijo mostrar el rectangilo que rodea el rostro en rojo
            
            #eye tree eyeglasse
            cv2.rectangle(frame, (x-w,y-(3*h)),(x+w+(4*w),y+(5*h)),(0,0,255),2)
            #rigth eye
            #cv2.rectangle(frame, (x-w,y-(2*h)),(x+w+(3*w),y+(4*h)),(0,0,255),2)
    
  cv2.imshow('frame',frame)
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    cap.release()
    cv2.destroyAllWindows()
    break