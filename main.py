
import tensorflow as tf


def run_iris_model():
    saver = tf.train.import_meta_graph('./models/trained/iris_model/iris_model.ckpt.meta')
    g = tf.get_default_graph()
    brain = g.get_tensor_by_name("iris_brain:0")
    x = g.get_tensor_by_name("iris_x:0")
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./models/trained/iris_model/'))
        data = [
            [6.0, 2.2, 4.0, 1.0],
            [6.7, 3.3, 5.7, 2.5],
            [4.9, 3.0, 1.4, 0.2],
            [6.4, 2.7, 5.3, 1.9],
            [4.8, 3.4, 1.6, 0.2],
            [6.9, 3.1, 4.9, 1.5],
        ]
        print(sess.run(tf.argmax(brain, 1), feed_dict={x: data}))


def run_zoo_model():
    saver = tf.train.import_meta_graph('./models/trained/zoo_model/zoo_model.ckpt.meta')
    g = tf.get_default_graph()
    brain = g.get_tensor_by_name("zoo_brain:0")
    x = g.get_tensor_by_name("zoo_x:0")
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./models/trained/zoo_model/'))
        data = [
            [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 4, 0, 0, 1],
            [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
            [0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 2, 1, 1, 0],
            [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 6, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 6, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 4, 1, 0, 0],
        ]
        print(sess.run(tf.argmax(brain, 1), feed_dict={x: data}))

if __name__ == '__main__':
    run_zoo_model()
    run_iris_model()