# Prueba en vivo
Hay dos directorios para hacer pruebas con una camara. Si bien ambos sirven para lo mismo, cada uno tiene una diferente implementacion. El primero utiliza el detector facial de haarcascade para detectar los rostros, mientras que el segundo usa el detector de ojos de haarcascade, para de esta forma poder detectar un rostro. Dentro de cada directorio a una explicacion a mayor detalle de como funciona cada uno.
Cada uno tiene una version ejecutable ".py" y ".ipynb".

## Haarcascade
Este directorio contiene los archivos necesarios de haarcascade para poder detectar tanto rostros como ojos.

## Modelo reconocedor de barbijos
Este directorio contiene el modelo "modeloV2" para clasificar personas que usan barbijo y personas que no. Este es el modelo que entrene previamente. El modeloV2 fue el que mejor resultados mostro por eso es el utilizado en las pruebas en vivo. Ademas dentro de este directorio tambien estan los pesos del modelo antes mencionado.