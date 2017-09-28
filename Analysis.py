# Esta oracion trae la funcion print de versiones futura y con imcompatibilidad
from __future__ import print_function


# Emplado para el trabajo con matrices
import numpy
# Empleado en el trabajo de redes neuronales
import theano
import theano.tensor as T # Abreviacion
# Del archivo importa un metodo(def)
from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
#from theano.misc.pkl_utils import dump
import six.moves.cPickle as pickle1
import matplotlib.pyplot as plt
from utils import tile_raster_images
try:
    import PIL.Image as Image
except ImportError:
    import Image
# correr esta al principipo
from CNN_MLP import LeNetConvPoolLayer

index1 = 0 # # de batch
batch_size = 10 # Tamano del conjunto de imagenes de test
nkerns = [20, 50] # Cantidad de filtros en las capas de convolucion
n_out = 500 # Neuronas en la capa oculta
dataset = 'mnist.pkl.gz'
# load_data es una subrutina que carga y trata los datos
datasets = load_data(dataset)
# Junto con los datos de test se encuentran los datos de entrenamiento y validacion
test_set_x, test_set_y = datasets[2]

# 10000 Imagenes de (28x28)784 pixeles
Imagenes=test_set_x.get_value(borrow=True)
# Tomar un Index1 de batch_size imagenes
# Ejemplo: las primeras 500 imagenes

layer_input_0=Imagenes[index1 * batch_size: (index1 + 1) * batch_size]
# Reformar las imagenes a (#imagenes, canales, ancho, altura)
# De esta forma debven entrar a las capas de tipo convolucion, pooling + activacion.
layer_input_0=layer_input_0.reshape((batch_size, 1, 28, 28))
# PRUEBA PARA MOSTRAR UN MISMO NUMERO EN MUCHAS NEURONAS
e=1
i=0
layer_input_tmp=[]
layer_input_0tmp=[]
while (i<batch_size) and (e == 1):
    layer_input_tmp=layer_input_0[4]
    layer_input_0tmp.append(layer_input_tmp) 
    i=i+1
if (e==1):
    layer_input_0=layer_input_0tmp
# Cargar modelos guardado en la epoca n para cada capa
Visual_Layer_0 = pickle1.load(open('datos_CNN_MLP/layer0_200.pkl'))
Visual_Layer_1 = pickle1.load(open('datos_CNN_MLP/layer1_200.pkl'))
Visual_Layer_2 = pickle1.load(open('datos_CNN_MLP/layer2_200.pkl'))
Visual_Layer_3 = pickle1.load(open('datos_CNN_MLP/layer3_200.pkl'))
# Crear funcion de convolucion + pooling + activacion de cada capa
# Capa 0
Visualizar_Layer_0 = theano.function(
inputs=[Visual_Layer_0.input],
outputs=Visual_Layer_0.output)
# Capa_1
Visualizar_Layer_1 = theano.function(
inputs=[Visual_Layer_1.input],
outputs=Visual_Layer_1.output)
# Capa_2
Visualizar_Layer_2 = theano.function(
inputs=[Visual_Layer_2.input],
outputs=Visual_Layer_2.output)
# Capa_3
Visualizar_Layer_3 = theano.function(
inputs=[Visual_Layer_3.input],
outputs=Visual_Layer_3.predicted_y())

# Computar la capa de convolucion al conjunto de imagenes
Imagenes_conv_0 = Visualizar_Layer_0(layer_input_0)
# Computar la segunda capa de convolucion
Imagenes_conv_1 = Visualizar_Layer_1(Imagenes_conv_0)
# Visualizar alguna imagen luego de atravesar la capa
plt.figure(1)
# Ejemplo de resultado despues de aplicar capa 0 de convolucion + pooling + activacion
plt.title('Capa de convolucion cero')
plt.imshow(Imagenes_conv_0[0][0], cmap=plt.get_cmap('gray'))
# Visualizar una imagen a la salida de la capa 2
plt.savefig('datos_CNN_MLP/layer0_200.png') 
print(Imagenes_conv_0.shape)
plt.figure(2)
plt.title('Capa de convolucion uno')
plt.imshow(Imagenes_conv_1[0][0], cmap=plt.get_cmap('gray'))
plt.savefig('datos_CNN_MLP/layer1_200.png') 

# Para visualizar un significado de lo que hace
# cada neurona de la capa oculta se grafican
# los pesos de neuronas sobre imagenes filtradas
# por las capas anteriores.
# Computar la tercera capa (Fully connected)
Imagenes_conv_1_Dim = Imagenes_conv_1.shape
layer_2_input = Imagenes_conv_1.reshape((Imagenes_conv_1_Dim[0]),(Imagenes_conv_1_Dim[1]*Imagenes_conv_1_Dim[2]*Imagenes_conv_1_Dim[3]))
Hiddenlayer = Visualizar_Layer_2(layer_2_input)


# Crear un conjunto de imagenes visualizables en mosaico
# Crear variable de entrada de tipo TensorType de 4D
Mosaico = theano.tensor.TensorType('float64', (False,) * 4)()
# Se crea una operacion de conversion de dimension
Mosaico_reshaped = Mosaico.reshape((Mosaico.shape[0]*Mosaico.shape[1], Mosaico.shape[2]*Mosaico.shape[3]))
# Se crea la funcion con la variable de entrada y operacion para volver a tener filas de pixeles de las imagenes, esta ves despues de la convolucion
convert_for_mosaic = theano.function(inputs=[Mosaico], outputs=Mosaico_reshaped)
# Se ejecuta la funcion donde la entrada son las imagenes despues de la capa convolucion+pooling + activacion
Mosaico_conv_0 = convert_for_mosaic(Imagenes_conv_0)
Mosaico_conv_1 = convert_for_mosaic(Imagenes_conv_1)

# Crea la matriz del Mosaico de imagenes despues de convolucion 0
Mosaico_conv_0 = tile_raster_images(
X=Mosaico_conv_0,
img_shape=(12, 12), tile_shape=(10, 10),
tile_spacing=(1, 1))

# Crea la matriz del Mosaico de imagenes despues de convolucion 1

Mosaico_conv_1 = tile_raster_images(
X=Mosaico_conv_1,
img_shape=(4, 4), tile_shape=(10, 10),
tile_spacing=(1, 1))

# Crear datos tipo arreglo para visualizar
Mosaico_conv_0 = Image.fromarray(Mosaico_conv_0)
Mosaico_conv_1 = Image.fromarray(Mosaico_conv_1)
# Mostrart imagen de mosaico con 100 imagenes aleatorias despues de la convolucion
# Capa 0
plt.figure(3)
plt.title('Capa de convolucion cero')
plt.imshow(Mosaico_conv_0, cmap=plt.get_cmap('gray'))
plt.savefig('datos_CNN_MLP/Mosaico_conv_0.png')
# Capa 1
plt.figure(4)
plt.title('Capa de convolucion uno')
plt.imshow(Mosaico_conv_1, cmap=plt.get_cmap('gray'))
plt.savefig('datos_CNN_MLP/Mosaico_conv_1.png')
# Obtener la matriz de pesos W de la capa oculta
# para este caso 800 x 500
W_2 = Visual_Layer_2.W.get_value()
# Representacion de Neuronas en la capa oculta
W_2 = numpy.array(W_2)
W_2l = W_2.reshape(n_out,nkerns[1],4,4)
# Capa 2
plt.figure(5)
plt.title('Neurona de capa oculta')
plt.imshow(W_2l[0][0], cmap=plt.get_cmap('gray'))

W_2lM = W_2l[0]
W_2lM.reshape(nkerns[1],4*4)

Mosaico_Hiddenlayer = tile_raster_images(
X=W_2lM,
img_shape=(4, 4), tile_shape = (5, 5),
tile_spacing=(1, 1))

Mosaico_Hiddenlayer = Image.fromarray(Mosaico_Hiddenlayer)

plt.figure(6)
plt.title('Neurona de capa oculta')
plt.imshow(Mosaico_Hiddenlayer, cmap=plt.get_cmap('gray'))

W_2 = W_2.reshape(n_out*nkerns[1],4*4)

Mosaico_Hiddenlayer = tile_raster_images(
X=W_2,
img_shape=(4, 4), tile_shape=(10, 10),
tile_spacing=(1, 1))

Mosaico_Hiddenlayer = Image.fromarray(Mosaico_Hiddenlayer)

# Capa 2
plt.figure(7)
plt.title('Neuronas aleatorias de capa oculta')
plt.imshow(Mosaico_Hiddenlayer, cmap=plt.get_cmap('gray'))
plt.savefig('datos_CNN_MLP/Mosaico_Hiddenlayer.png')
# Computar la cuarta capa (Prediccion empleando regresion logistica y la funcion Softmax)
Prediccion = Visualizar_Layer_3(Hiddenlayer)
print(Prediccion)

# Imprimir las curvas de errores de entrenamiento, validacion y test
Errores = pickle1.load(open('datos_CNN_MLP/Error.pkl'))
Errores = numpy.array(Errores)
Costo = Errores[:,0]
Error_entrenamiento = Errores[:,1]
Error_test = Errores[:,2]
Error_validacion = Errores[:,3]
n_muestras = 221
t = numpy.arange(0.0,n_muestras, 1)
print(t.shape)
print(Error_entrenamiento.shape)
plt.figure(8)
plt.plot(t, Error_entrenamiento*100,'o', label = 'Error de entrenamiento')
plt.plot(t, Error_validacion*100,'x', label = 'Error de validacion')
plt.plot(t, Error_test*100, 's', label = 'Error de test')
plt.legend()
plt.xlabel('epoca')
plt.ylabel('Error')
plt.title('Errores (Entrenamiento, validacion y test)')
plt.axis([0,n_muestras, 0, Errores[:,1:3].max()*100])
plt.grid(True)
plt.savefig('datos_CNN_MLP/Errores.png')
plt.show()

plt.figure(9)
plt.plot(t, Costo,'ko')
plt.xlabel('Epoca')
plt.ylabel('Costo')
plt.title('Funcion de costo')
plt.axis([0, n_muestras, 0, Costo.max()])
plt.grid(True)
plt.savefig('datos_CNN_MLP/Costo.png')
plt.show()

# Graficar gradientes por capa
c=0
W_3=[]
W_2=[]
W_1=[]
W_0=[]
b_3=[]
b_2=[]
b_1=[]
b_0=[]
while (c < 200):
    c = c + 10
    Parametros='datos_CNN_MLP/Parametros%i.zip' % c
    W3=numpy.load(Parametros)['W']
    b3=numpy.load(Parametros)['b']
    W2=numpy.load(Parametros)['W_2']
    b2=numpy.load(Parametros)['b_2']
    W1=numpy.load(Parametros)['W_3']
    b1=numpy.load(Parametros)['b_3']
    W0=numpy.load(Parametros)['W_4']
    b0=numpy.load(Parametros)['b_4']
    W_3.append(W3)
    W_2.append(W2)
    W_1.append(W1)
    W_0.append(W0)
    b_3.append(b3)
    b_2.append(b2)
    b_1.append(b1)
    b_0.append(b0)

c=0
W3grad = 0
W_3grad = []
W2grad = 0
W_2grad = []
W1grad = 0
W_1grad = []
W0grad = 0
W_0grad = []
b3grad = 0
b_3grad = []
b2grad = 0
b_2grad = []
b1grad = 0
b_1grad = []
b0grad = 0
b_0grad = []
while (c < 18):
    W3grad = numpy.sum(W_3[c]-W_3[c+1])
    W_3grad.append(W3grad) 
    W2grad = numpy.sum(W_2[c]-W_2[c+1])
    W_2grad.append(W2grad) 
    W1grad = numpy.sum(W_1[c]-W_1[c+1])
    W_1grad.append(W1grad) 
    W0grad = numpy.sum(W_0[c]-W_0[c+1])
    W_0grad.append(W0grad)
    b3grad = numpy.sum(b_3[c]-b_3[c+1])
    b_3grad.append(b3grad) 
    b2grad = numpy.sum(b_2[c]-b_2[c+1])
    b_2grad.append(b2grad) 
    b1grad = numpy.sum(b_1[c]-b_1[c+1])
    b_1grad.append(b1grad) 
    b0grad = numpy.sum(b_0[c]-b_0[c+1])
    b_0grad.append(b0grad)
    c = c + 1 

t = numpy.arange(0.0,18, 1)
print(t.shape)
print(Error_entrenamiento.shape)
plt.figure(10)
plt.plot(t, W_3grad,'o', label = 'Gradiente capa 3')
plt.plot(t, W_2grad,'x', label = 'Gradiente Capa 2')
plt.plot(t, W_1grad, 's', label = 'Gradiente capa 1')
plt.plot(t, W_0grad, 'v', label = 'Gradiente capa 0')
plt.legend()
plt.xlabel('epoca')
plt.ylabel('Valor del gradiente')
plt.title('Gradiente por capa')
plt.axis([0,18,-0.5,2.5])
plt.grid(True)
plt.savefig('datos_CNN_MLP/Gradientes.png')
plt.show()

plt.figure(11)
plt.plot(t, b_3grad,'o', label = 'Bias capa 3')
plt.plot(t, b_2grad,'x', label = 'Bias Capa 2')
plt.plot(t, b_1grad, 's', label = 'Bias capa 1')
plt.plot(t, b_0grad, 'v', label = 'Bias capa 0')
plt.legend()
plt.xlabel('epoca')
plt.ylabel('Valor del Gradiente')
plt.title('Gradientes de bias por capa')
plt.axis([0,18,-0.1,0.4])
plt.grid(True)
plt.savefig('datos_CNN_MLP/Gradiente_bias.png')
plt.show()

rng = numpy.random.RandomState(23455)
datasets = load_data(dataset)

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

# compute number of minibatches for training, validation and testing
n_train_batches = train_set_x.get_value(borrow=True).shape[0]
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
n_test_batches = test_set_x.get_value(borrow=True).shape[0]
n_train_batches //= batch_size
n_valid_batches //= batch_size
n_test_batches //= batch_size

# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch

# start-snippet-1
x = T.matrix('x')   # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of
                    # [int] labels

######################
# BUILD ACTUAL MODEL #
######################
print('... building the model')

# Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
# to a 4D tensor, compatible with our LeNetConvPoolLayer
# (28, 28) is the size of MNIST images.
layer0_input = x.reshape((batch_size, 1, 28, 28))

# Construct the first convolutional pooling layer:
# filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
# maxpooling reduces this further to (24/2, 24/2) = (12, 12)
# 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
layer0 = LeNetConvPoolLayer(
    rng,
    input=layer0_input,
    image_shape=(batch_size, 1, 28, 28),
    filter_shape=(nkerns[0], 1, 5, 5),
    poolsize=(2, 2)
)

# Construct the second convolutional pooling layer
# filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
# maxpooling reduces this further to (8/2, 8/2) = (4, 4)
# 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
layer1 = LeNetConvPoolLayer(
    rng,
    input=layer0.output,
    image_shape=(batch_size, nkerns[0], 12, 12),
    filter_shape=(nkerns[1], nkerns[0], 5, 5),
    poolsize=(2, 2)
)

# the HiddenLayer being fully-connected, it operates on 2D matrices of
# shape (batch_size, num_pixels) (i.e matrix of rasterized images).
# This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
# or (500, 50 * 4 * 4) = (500, 800) with the default values.
layer2_input = layer1.output.flatten(2)

# construct a fully-connected sigmoidal layer
layer2 = HiddenLayer(
    rng,
    input=layer2_input,
    n_in=nkerns[1] * 4 * 4,
    n_out=500,
    activation=T.tanh
)
# classify the values of the fully-connected sigmoidal layer
layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

train_model = theano.function(
    [index],
    layer3.errors(y),
    givens={
        x: train_set_x[index * batch_size: (index + 1) * batch_size],
        y: train_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

test_model = theano.function(
    [index],
    layer3.errors(y),
    givens={
        x: test_set_x[index * batch_size: (index + 1) * batch_size],
        y: test_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

validate_model = theano.function(
    [index],
    layer3.errors(y),
    givens={
        x: valid_set_x[index * batch_size: (index + 1) * batch_size],
        y: valid_set_y[index * batch_size: (index + 1) * batch_size]
    }
)


predicted_value = theano.function(
    [index],
    layer3.predicted_y(),
    givens={
        x: test_set_x[index * batch_size: (index + 1) * batch_size],
    }
)

epoch_ant=200
#    if os.path.isfile('datos_CNN_MLP/epoch_ant.pkl'):
#        epoch_ant = pickle1.load(open('datos_CNN_MLP/epoch_ant.pkl'))
#
#
#    while (epoch <= epoch_ant):
#        epoch = epoch + 1
Parametros='datos_CNN_MLP/Parametros%i.zip' % epoch_ant
#        if not os.path.isfile(Parametros):
#           Parametros='Parametros1.zip'

#        if os.path.isfile(Parametros):
W3=numpy.load(Parametros)['W']
b3=numpy.load(Parametros)['b']
W2=numpy.load(Parametros)['W_2']
b2=numpy.load(Parametros)['b_2']
W1=numpy.load(Parametros)['W_3']
b1=numpy.load(Parametros)['b_3']
W0=numpy.load(Parametros)['W_4']
b0=numpy.load(Parametros)['b_4']# create a list of all model parameters to be fit by gradient descent
#       MODIFICA LOS PARAMETROS DE LAS CAPAS DE ACUERDO AA PARAMETROS ANTERIORES GUARDADOS
layer3.W.set_value(W3)
layer3.b.set_value(b3)
layer2.W.set_value(W2)
layer2.b.set_value(b2)
layer1.W.set_value(W1)
layer1.b.set_value(b1)
layer0.W.set_value(W0)
layer0.b.set_value(b0)

#            params = layer3.params + layer2.params + layer1.params + layer0.params
#            validation_losses = [validate_model(i) for i
#                in range(n_valid_batches)]
#            this_validation_loss = numpy.mean(validation_losses)
#            test_losses = [test_model(i)
#                for i in range(n_test_batches)]
#            test_score = numpy.mean(test_losses)
#
#            training_error.append(train_model(epoch)*100)
#            validation_error.append(this_validation_loss*100)
#            test_error.append(test_score*100)
#



img0=test_set_x.get_value(borrow=True)
index1=0
img0=img0[index1 * batch_size: (index1 + 1) * batch_size]
img0=img0.reshape((batch_size, 1, 28, 28))
Mosaico_img0=convert_for_mosaic(img0)
Mosaico_prediccion = tile_raster_images(
X=Mosaico_img0,
img_shape=(28, 28), tile_shape=(10, 1),
tile_spacing=(1, 1))

# Crear datos tipo arreglo para visualizar
Mosaico = Image.fromarray(Mosaico_prediccion)
plt.figure(10)
plt.imshow(Mosaico, cmap=plt.get_cmap('gray'))

predicted_values = predicted_value(0)
print("Predicted values for the first 10 examples in test set:")
print(predicted_values)


