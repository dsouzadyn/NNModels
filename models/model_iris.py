from .datasets.iris.data_iris import *
import tensorflow as tf

n_inputs = 4
n_hidden_1 = 256
n_hidden_2 = 256
n_classes = 3

data = get_iris_data()
features = data['f']
labels = data['l']

x = tf.placeholder("float", [None, n_inputs], name='iris_x')
y = tf.placeholder("float", [None, n_classes], name='iris_y')


def iris_net(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.tanh(layer_1, name="iris_layer_1")
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.tanh(layer_2, name="iris_layer_2")
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['o'], name='iris_brain')
    return out_layer


def train_iris_net(training_epochs=10000, learning_rate=0.0001, batch_size=50):
    display_step = 1000
    total_batch = int(len(data['f']) / batch_size)
    bf = create_batches(features, batch_size)
    bl = create_batches(labels, batch_size)
    weights = {
        'h1': tf.Variable(tf.random_normal([n_inputs, n_hidden_1]), name="iris_hidden_1"),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="iris_hidden_2"),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name="iris_output")
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1]), name="iris_bias_1"),
        'b2': tf.Variable(tf.random_normal([n_hidden_2]), name="iris_bias_2"),
        'o': tf.Variable(tf.random_normal([n_classes]), name="iris_output_bias")
    }
    brain = iris_net(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=brain, labels=y), name="iris_cost")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name="zoo_optimizer").minimize(cost)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        test_data = get_test_data()
        for epoch in range(training_epochs):
            avg_cost = 0
            for b in range(total_batch - 1):
                batch_x, batch_y = bf[b], bl[b]
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c/batch_size
            if epoch % display_step == 0:
                pass
                print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
        print("Optimization Finished")
        correct_prediction = tf.equal(tf.argmax(brain, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        acc = accuracy.eval({x: test_data['f'][:], y: test_data['l'][:]})
        print("Accuracy:", accuracy.eval({x: test_data['f'], y: test_data['l'][:30]}))
        if acc > 0.98:
            dir_path = os.path.dirname(__file__)
            dir_path += '/trained/'
            try:
                os.makedirs(dir_path+'iris_model')
            except OSError as e:
                print("[ERROR]:", e)
            file_path = os.path.join(dir_path+'iris_model/', 'iris_model.ckpt')
            saver.save(sess, file_path)
            print("Model trained and saved.")
        else:
            print("Model trained but not saved.")
