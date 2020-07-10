# Con detector de ojos

Como mencione anteriormente el problema de usar el detector facial, es que no siempre es capaz de encontrar rostros con barbijo.
Para solucionar dicho problema, en cambio utilice el detector de ojos.

## Logica del codigo
El detector de ojos, como su nombre lo indica detecta la posicion de los ojos en el rostro. Y guarda las coordenadas en un arreglo. 
La solucion fue con el tamaño del recuadro que detecta el ojo, proporcionalmente incremento el tamaño para que el recuadro abarque todo el ojo. El problema era que al tener cada ojo un recuadro, si bien el primer recuadro enmarcaba muy bien el rostro, el segundo no lo hacia, ya que el recuadro cubria la mitad del rostro y la otra mitad fuera del rostro. 

Analizando un poco el funcionamiento del detector de ojos, el primer ojo que siempre analizaba era el derecho, por lo que la primera solucion fue, no usar los numeros impares del arreglo de coordenadas, ya que los numeros impares del arreglo pertenecian a los ojos izquierdos. El problema de esta solucion era que si no se detectaba un ojo derecho, los numeros pares pasarian a ser los izquierdos, por lo que nuevamente los recuadros serian fueron del rostro.

La ultima solucion fue, una vez encontrado un ojo, recortar la posicion donde posiblemente este el rostro, y esta imagen recortada, nuevamente pasarla por el detector de ojos, ya que si esta imagen contenia dos ojos, significa que esta enmarcando correctamente el rostro, por lo que podemos pasar dicha imagen al red clasificadora de barbijos. 

## Como usar el codigo
El codigo utiliza dos detectores de ojos:
* eye_tree_glases
* rigth_eye

Ambos cumplen la misma funcion, la unica diferencia es que el tamaño del recuadro que enmarca al ojo es un poco diferente.
El codigo contiene ambos implementados, para usar uno u el otro, solo hay que desmarcar las partes donde se menciona el clasificador que se esta usando. No los dos al mismo tiempo ya que esto generaria fallas, ni mezclar uno con el otro, ya que el tamaño del recuadro no seria el correcto.

## Librerias necesarias para ejecutar
Este codigo no puede ser ejecutado en google colab, ya que presenta fallos a la hora de usar openCV. Una alternativa para poder ejecutar el ".ipynb" seria usando anaconda, el cual ya tiene todas las librerias necesarias. Usando jupyter lab.

Para poder ejecutar el ".py" se requiere de las siguientes librerias instaladas previamente:
* openCV
* numpy
* tensorflow2.x

(*) Tambien es necesario una camara conectado al dispositivo.