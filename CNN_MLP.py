"""
El algoritmo implementa la red neuronal de convolución para tareas de clasificación.
Este software fue hecho para enseñar a personas que inician el aprendizaje con redes neuronales 
profundas, empleando una red neuronal de convolución profunda ya estudiada como Lenet5.
"""
"""Referencias:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
"""
# LLAMADO A LIBRERIAS PYTHON
# Esta oracion trae la funcion print de versiones futura y con imcompatibilidad
from __future__ import print_function 
# Emplear funciones del sistema operativo
import os
# Provee acceso a algunas variables usadas o mantenidas por el interprete
import sys
# Empleado para medir tiempos de ejecucion
import timeit
# Emplado para el trabajo con matrices
import numpy
# Empleado en el trabajo de redes neuronales
import theano
import theano.tensor as T # Abreviacion
from theano.tensor.signal import downsample # Implementa muestreo
from theano.tensor.nnet import conv2d # Implementar convoluciones2d con ciertas caracteristicas y entradas
# Del archivo importa un método(def)
from logistic_sgd import LogisticRegression, load_data, sgd_optimization_mnist
from mlp import HiddenLayer
#from theano.misc.pkl_utils import dump
import six.moves.cPickle as pickle1
from theano.misc.pkl_utils import dump
import matplotlib.pyplot as plt
from utils import tile_raster_images
from theano import function
try:
    import PIL.Image as Image
except ImportError:
    import Image

# CREA LA CLASE QUE CONSTRUYE LA RED NEURONAL DE CONVOLUCION 
class LeNetConvPoolLayer(object):
    """Capa de pooling de la red neuromal """
# Constructor __init__ empleado para inicializar los parámetros.
# self: Empleado para compartir variables  
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Permite que la capa de convolución LeNetConvPoolLayer tenga variables compartidas.

        :tipo rng: numpy.random.RandomState
        :parámetro rng: Generador de números aleatorios para inicializar los pesos.
          
        :tipo input: theano.tensor.dtensor4
        :parámetro input: tensor simólico de imágen.

        :tipo filter_shape: lista de longitud 4
        :parámetro filter_shape: (número de filtros, número de entradas  como mapas de carácteristicas,
                              alto y ancho del filtro)

        :tipo image_shape: lista de longitud 4
        :parámetro image_shape: (Dimnsion del batch, número de mapas de entrada carácteristicos,
                            altura de imágen, ancho de imágen)

        :tipo poolsize: lista de longitud 2
        :parámetro poolsize: El submuestro (pooling) factor (#filas, #columnas)
        """
# SE COMPRUEBA EN TIEMPO DE DEPURACION COMO UNA CONDICION DE NO ERROR
        assert image_shape[1] == filter_shape[1]
        self.input = input # Variable compartida

        # Son  "num input feature maps * filter height * filter width"
        # entradas a cada unidad oculta
        fan_in = numpy.prod(filter_shape[1:]) # Producto de los elementos de a al total por filas
	# Cada unidad en la capa inferior recibe un gradiente de:
        # "num output feature maps * filter height * filter width" /
        #  Dimensión del pooling
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
	# Inicializar pesos con valores aleatorios
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),name='W',
            borrow=True
        ) # borrow=True permite que w sea una copia que se pueda cambiar  

        # bias es un tensor de una dimensión -- uno por mapa carácteristico
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values,name='b', borrow=True)

        # convolución de las entradas carácteristicas con filtros

        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

	# Submuestreo de cada mapa de carácteristicas usando el maxpooling.
	    pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # Adicionar el bias. un tensor de forma (1, n_filters, 1, 1). 
        # Ancho y altura
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # Almacenar parámetros de esta capa
        self.params = [self.W, self.b]

        self.input = input

def evaluate_lenet5(learning_rate=0.1, n_epochs=250,
                    dataset='mnist.pkl.gz',
                    nkerns=[20, 50], batch_size=500):
    """ Demostración de aprendizaje profundo empleando lenet5

    :tipo learning_rate: float
    :parámetro learning_rate: learning rate (para el gradiente estocástico)

    :tipo n_epochs: int
    :parámetro n_epochs: número de veces que se ejecuta el optimizador

    :typo dataset: string
    :parámetro dataset: ruta al dataset para entrenar y testear (MNIST)

    :typo nkerns: lista de ints
    :param nkerns: número d kernels e cada capa
    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

	#Compute el número de minibatchpara validar, ntrenar y testear
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    index = T.lscalar()  # indice a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # imágenes rasterizadas
    y = T.ivector('y')  # Vector de etiquetas

    ######################
    # Construir el modelo #
    ######################
    print('Construyendo modelo')

    # Reformar imágenes rasterizadas
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # Construcción de primera capa de comvolución:
    # filtrado reduce a (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduce a (24/2, 24/2) = (12, 12)
    # tensor de salida 4D (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    # Segunda capa de convolción 
    # fiiltrado reduce a (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduce a (8/2, 8/2) = (4, 4)
    # Tensor de salida (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    # Capa oculta completamenta conectadas shape (batch_size, num_pixels).
    # Genera matrices de (batch_size, nkerns[1] * 4 * 4),
    # o (500, 50 * 4 * 4) = (500, 800) con los valores por defecto.
    layer2_input = layer1.output.flatten(2)

    # Capa completamene conectada con activación sigmoidea
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=500,
        activation=T.tanh
    )

    # Clasificador de las salidas de la capa anterior
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

    # Costo que se debe minimizar durante el entrenamiento
    cost = layer3.negative_log_likelihood(y)

    # Función que genea el error producido por el modelo
    train_error_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )    
# Coomputa el error con los datos de validación
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
    
#Permite entrenar el modelo por epocas y retomar el modelo en pasadas simulaciones
    epoch_ant=0
    if os.path.isfile('datos_CNN_MLP/epoch_ant.pkl'):
        epoch_ant = pickle1.load(open('datos_CNN_MLP/epoch_ant.pkl')) 
    
    Parametros='datos_CNN_MLP/Parametros%i.zip' % epoch_ant
    
	# Carga de parámetros iniciales
    if os.path.isfile(Parametros):
        W3=numpy.load(Parametros)['W']
        b3=numpy.load(Parametros)['b']    
        W2=numpy.load(Parametros)['W_2']
        b2=numpy.load(Parametros)['b_2']
        W1=numpy.load(Parametros)['W_3']
        b1=numpy.load(Parametros)['b_3']    
        W0=numpy.load(Parametros)['W_4']
        b0=numpy.load(Parametros)['b_4']# create a list of all model parameters to be fit by gradient descent
#       MODIFICA LOS PARAMETROS DE LAS CAPAS DE ACUERDO A PARAMETROS ANTERIORES GUARDADOS 
        layer3.W.set_value(W3)    
        layer3.b.set_value(b3)
        layer2.W.set_value(W2)    
        layer2.b.set_value(b2)
        layer1.W.set_value(W1)    
        layer1.b.set_value(b1)
        layer0.W.set_value(W0)    
        layer0.b.set_value(b0)     
        
    params = layer3.params + layer2.params + layer1.params + layer0.params    
#    img=train_set_x.get_value(borrow=True)
#    img=img.reshape(50000, 28, 28)
#    print(img.shape)    
#    imgplot = plt.imshow(img[1000], cmap=plt.get_cmap('gray'))   
# Crea  lista de gradientes para todos los parámetros del modelo     
    grads = T.grad(cost, params)

# Actualiza los parámetros basado en el modelo de entrenamiento y mediante SGD (Stochastic Gradient Descent)
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    #print(train_set_x[index| * batch_size: (index + 1) * batch_size])
    ###############
    # Modelo de entrenamiento #
    ###############
    print('... Entrenando el modelo')
    # Parámetros de parada
    patience = 10000  
    patience_increase = 2 
                          
    improvement_threshold = 0.995  
    validation_frequency = min(n_train_batches, patience // 2)
                                  
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0
    start_time = timeit.default_timer()

    epoch = 0
    epoch_ant=0
    if os.path.isfile('datos_CNN_MLP/epoch_ant.pkl'):
        epoch_ant = pickle1.load(open('datos_CNN_MLP/epoch_ant.pkl')) 
    iter = 0
    done_looping = False
    Error=[]
    while (epoch < n_epochs) and (not done_looping):
        epoch=epoch_ant
        epoch = epoch + 1 
        Modelo='datos_CNN_MLP/epoch_ant.pkl'            
        with open(Modelo, 'wb') as P:
               pickle1.dump(epoch_ant, P) 
               P.close()
        
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)
                       
            # Guarda los parametros del modelo en cada epoca
            if epoch % 10 == 0:
                Modelo='datos_CNN_MLP/Parametros%i.zip' % epoch            
                with open(Modelo, 'wb') as P:
                    dump(params, P) 
                    P.close()
                Modelo='datos_CNN_MLP/layer0_%i.pkl' % epoch            
                with open(Modelo, 'wb') as L:
                    pickle1.dump(layer0, L)
                    L.close()
   
                Modelo='datos_CNN_MLP/layer1_%i.pkl' % epoch            
                with open(Modelo, 'wb') as L:
                    pickle1.dump(layer1, L)
                    L.close()   
                
                Modelo='datos_CNN_MLP/layer2_%i.pkl' % epoch            
                with open(Modelo, 'wb') as L:
                    pickle1.dump(layer2, L)
                    L.close()            
                
                Modelo='datos_CNN_MLP/layer3_%i.pkl' % epoch            
                with open(Modelo, 'wb') as L:
                    pickle1.dump(layer3, L)
                    L.close()

            epoch_ant=epoch
            if (iter + 1) % validation_frequency == 0:
                train_losses = [train_error_model(i) for i in range(n_train_batches)]
                train_error = numpy.mean(train_losses)
 
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                
                test_losses = [test_model(i) for i in range(n_test_batches)]
                test_score = numpy.mean(test_losses)
                Error0=[cost_ij, train_error, test_score, this_validation_loss]                    
                Error.append(Error0)
                print('epoch %i, minibatch %i/%i, training error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       train_error * 100.))
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_valid_batches,
                       this_validation_loss * 100.))
                print('epoch %i, minibatch %i/%i, Test error %f %%' %
                      (epoch, minibatch_index + 1, n_test_batches,
                       test_score * 100.))
                
                if this_validation_loss < best_validation_loss:

                    
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    
                    Modelo='datos_CNN_MLP/Error.pkl'            
                    with open(Modelo, 'wb') as L:
                        pickle1.dump(Error, L)
                        L.close()                     
                    
                    print(('epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            
            
            if patience <= iter:
                done_looping = True
                break
         
    end_time = timeit.default_timer()
    print('Optimizacion completa.')
    print('Mejor validación %f %% en iteración %i, '
          'con desempeño de %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('El código del archivo ' +
           os.path.split(__file__)[1] +
           ' se ejecuto en %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)


if __name__ == '__main__':
    evaluate_lenet5()

def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
