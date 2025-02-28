#Importamos la parte de la libreria de matplotlib que necesitamos.
from matplotlib import pyplot as plt
#Declaramos dos listas de valores para las x e y
x=[1,3,5]
y=[2,6,3]
#Invocamos el metodo plot de la clase plt
plt.show(x,y)
plt.show()
#Podemos pasar mas metodos y argumentos para personalizar nuestro grafico
plt.plot(x,y,color='n',label='nuestra',linewidth=5)
plt.title('MI primer grafico en matplotlib')
plt.xlabel('Eje x')
plt.ylabel('Eje y')
plt.legend()
plt.show()
#Creamos una matriz de 2x2 subgraficos
fig,axes=plt.subplots(2,2)
#Damos contenido a cada subgrafico
axes[]