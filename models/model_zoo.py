from .datasets.zoo.data_zoo import *
import tensorflow as tf

n_inputs = 16
n_hidden_1 = 512
n_hidden_2 = 256
n_classes = 7

data = get_zoo_data()
features = data['f']
labels = data['l']

x = tf.placeholder("float", [None, n_inputs], name='zoo_x')
y = tf.placeholder("float", [None, n_classes], name='zoo_y')


def zoo_net(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1, name="zoo_layer_1")
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2, name="zoo_layer_2")
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['o'], name='zoo_brain')
    return out_layer


def train_zoo_net(training_epochs=10000, learning_rate=0.0001, batch_size=50):
    display_step = 1000
    total_batch = int(len(data['f']) / batch_size)
    bf = create_batches(features, batch_size)
    bl = create_batches(labels, batch_size)
    weights = {
        'h1': tf.Variable(tf.random_normal([n_inputs, n_hidden_1]), name="zoo_hidden_1"),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="zoo_hidden_2"),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name="zoo_output")
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1]), name="zoo_bias_1"),
        'b2': tf.Variable(tf.random_normal([n_hidden_2]), name="zoo_bias_2"),
        'o': tf.Variable(tf.random_normal([n_classes]), name="zoo_output_bias")
    }
    brain = zoo_net(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=brain, labels=y), name="zoo_cost")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="zoo_optimizer").minimize(cost)
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
                os.makedirs(dir_path+'zoo_model')
            except OSError as e:
                print("[ERROR]:", e)
            file_path = os.path.join(dir_path+'zoo_model/', 'zoo_model.ckpt')
            saver.save(sess, file_path)
            print("Model trained and saved.")
        else:
            print("Model trained but not saved.")
