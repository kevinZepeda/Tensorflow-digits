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

# El modelo
Y = tf.nn.softmax(tf.matmul(XX, W) + b)

# cross-entropy
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0  
                                                          

# presicion del modelo entrenado, entre 0 (peor) y 1 (mejor)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# entrenamiento, ratio de aprendizaje = 0.005
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)


# matplotlib visualisation
allweights = tf.reshape(W, [-1])
allbiases = tf.reshape(b, [-1])
I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)
It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)
datavis = tensorflowvisu.MnistDataVis()


# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)