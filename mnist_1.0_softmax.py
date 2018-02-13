import tensorflow as tf
import tensorflowvisu
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

# Descargar imagenes y labels a mnist.test (10K imagenes + labels) y mnist.train (60K imagenes+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# X: 28x28 imagenes a escala de grises, la primer dimension (None) indexará las imagenes
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# respuestas correctas
Y_ = tf.placeholder(tf.float32, [None, 10])
# weights W[784, 10]   784=28*28
W = tf.Variable(tf.zeros([784, 10]))
# biases b[10]
b = tf.Variable(tf.zeros([10]))

# convierte imagenes a una sola linea de pixeles
# -1 significa "la unica dimension posible que mantendra el numero de elementos"
XX = tf.reshape(X, [-1, 784])