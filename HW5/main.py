import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import LeNet
from sklearn.utils import shuffle


def evaluate(x, y, x_data, y_data, Batch_size, acc):
    num_examples = len(x_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for begin in range(0, num_examples, Batch_size):
        batch_x, batch_y = x_data[begin: begin+Batch_size], y_data[begin: begin+Batch_size]
        accuracy = sess.run(acc, feed_dict={x: batch_x, y: batch_y})
        total_accuracy = total_accuracy + (accuracy*len(batch_x))
    return total_accuracy/num_examples

def MNIST(is_train=True, Epochs = 10, Batch_size = 128, learning_rate = 0.001, 
          data_path = "./MNIST_data", model_path = './model/', log_path='./log/'):
    
    mnist = input_data.read_data_sets(data_path, reshape=False)

    x_train, y_train = mnist.train.images, mnist.train.labels
    x_val, y_val = mnist.validation.images, mnist.validation.labels
    x_test, y_test = mnist.test.images, mnist.test.labels
    
    tf.summary.image('input', x_test, 5)

    x_train = np.pad(x_train, [(0,0),(2,2),(2,2),(0,0)], "constant")
    x_val = np.pad(x_val, [(0,0),(2,2),(2,2),(0,0)], "constant")
    x_test = np.pad(x_test, [(0,0),(2,2),(2,2),(0,0)], "constant")

    x_train, y_train = shuffle(x_train, y_train)

    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))
    y = tf.placeholder(tf.int32, shape=(None))
    y_onehot = tf.one_hot(y, 10)

    logits = LeNet(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=logits)

    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_onehot, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('loss',loss_operation)
    tf.summary.scalar('accuracy', acc)
    summ = tf.summary.merge_all()

    saver = tf.train.Saver()

    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    with tf.Session() as sess:
        total_batch = int(mnist.train.num_examples/Batch_size)
        writer = tf.summary.FileWriter(log_path)
        writer.add_graph(sess.graph)
        if is_train:
            sess.run(tf.global_variables_initializer())
            num_examples = len(x_train)
            print("-------Start Training-------")
            for i in range(Epochs):
                x_train, y_train = shuffle(x_train, y_train)
                for j in range(total_batch):
                    begin = j * Batch_size
                    end = begin + Batch_size
                    # batch_x, batch_y = mnist.train.next_batch(Batch_size)
                    batch_x, batch_y = x_train[begin:end], y_train[begin:end]
                    _, summ1 = sess.run([training_operation,summ], feed_dict={x:batch_x, y:batch_y})
                    # writer.add_summary(summ1, i)
                    writer.add_summary(summ1, i * total_batch + j)
                val_acc = evaluate(x, y, x_val, y_val, Batch_size, acc)
                print("Epochs {}".format(i+1))
                print("Validation Accuracy = {:.4f}".format(val_acc))
                saver.save(sess, model_path+"model.ckpt")
            print("model saved")
        else:
            print("-------Start Testing-------")
            model_file = tf.train.latest_checkpoint(model_path)
            saver.restore(sess,model_file)
            test_acc = sess.run(acc, feed_dict={x: x_test, y: y_test})
            print('test_acc: {:.4f}'.format(test_acc))
        writer.close()
        

if __name__ == '__main__':
    N=len(sys.argv)
    if N==3:
        if sys.argv[1]=='True':
            MNIST(is_train=True, data_path = sys.argv[2])
        else:
            MNIST(is_train=False, data_path = sys.argv[2])
    else:
        MNIST()