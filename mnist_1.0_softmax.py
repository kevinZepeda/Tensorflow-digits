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

# Funcion para entrenar el modelo 100 imagenes a la vez
def training_step(i, update_test_data, update_train_data):

    # entrenar en batches de 100 imagenes con 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    # computar valores de entrenamiento para visualizacion
    if update_train_data:
        a, c, im, w, b = sess.run([accuracy, cross_entropy, I, allweights, allbiases], feed_dict={X: batch_X, Y_: batch_Y})
        datavis.append_training_curves_data(i, a, c)
        datavis.append_data_histograms(i, w, b)
        datavis.update_image1(im)
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))

    # computar valores de prueba para visualizacion
    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    # propagacion del entrenamiento
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})

# animacion
datavis.animate(training_step, iterations=2000+1, train_data_update_freq=10, test_data_update_freq=50, more_tests_at_start=True)