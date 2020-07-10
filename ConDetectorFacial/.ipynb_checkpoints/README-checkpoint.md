# Con detector facial 

Este codigo usa el detector facial de haarcascade para obtener las coordenadas de un rostro en cada frame de la grabacion. Una vez que obtiene las coordenadas de un rostro, se recorta el rostro de la imagen, y se lo procesa en el modelo que entrenamos, Si el rostro esta usando un barbijo, el recuadro alrededor del rostro sera de color verde; si la persona no esta usando barbijo, el cuadro alrededor del rostro sera de color rojo.

## Librerias necesarias para ejecutar
Este codigo no puede ser ejecutado en google colab, ya que presenta fallos a la hora de usar openCV. Una alternativa para poder ejecutar el ".ipynb" seria usando anaconda, el cual ya tiene todas las librerias necesarias. Usando jupyter lab.

Para poder ejecutar el ".py" se requiere de las siguientes librerias instaladas previamente:
* openCV
* numpy
* tensorflow2.x

(*) Tambien es necesario una camara conectado al dispositivo.
## Notas
El problema de utilizar el reconocedor facial, es que si bien nos da las coordenadas exactas del rostro, cuando una persona se pone un barbijo, presenta problemas al detectar rostros. Como una solucion a esto fue utilizar el detector de ojos. Una explicacion a mayor detalle  de como funciona esta dentro de su propio directorio.