import tensorflow as tf
import numpy as np
import time

class RnnCharGenModel:
    """ Generate text on rnn model

    Using char as the input, for Chinese using the embedding, because have more
    than 40K ofen used chars, for English do not using embeddings, bacause have
    less than 128 chars
    """

    def __init__(self, batch_size, n_steps,
            lstm_size, lstm_layers, vocab_size,
            decay_steps, decay_rate, learning_rate, keep_rate,
            using_embedding, embedding_size=100):
        """ create the networks

        Args:
            batch_size: each batch size
            n_steps: the time step
            lstm_size: the state unit size of lstm cell
            lstm_layers: how many lays need to stack
            vocab_size: the vocabulary size
            decay_steps: decay limit steps
            decay_rate: decay rate
            learning_rate: initial learning rate
            keep_rate: the keep rate of every lstm output
            using_embedding: weather to using embedding
            embedding_size: define the embedding size
        """

        self.batch_size = batch_size
        self.n_steps = n_steps
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.vocab_size = vocab_size
        self.using_embedding = using_embedding
        self.embedding_size = embedding_size
        self.initial_state_val = None

        with tf.variable_scope("networks"):
            self.x_holder = tf.placeholder(dtype=tf.int32, shape=[batch_size,
                n_steps], name="input_x")
            self.y_holder = tf.placeholder(dtype=tf.int32, shape=[batch_size,
                n_steps], name="input_y")

            if using_embedding:
                embeddings = tf.get_variable(name="embeddings", shape=[vocab_size,
                    embedding_size], dtype=tf.float32,
                    )
                inputs = tf.nn.embedding_lookup(embeddings, self.x_holder)
            else:
                # using the one hot as the inputs, the last dim is vocab_size
                inputs = tf.one_hot(self.x_holder, vocab_size)

            def get_cell(lstm_size, keep_rate):
                cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
                return tf.nn.rnn_cell.DropoutWrapper(cell,
                        output_keep_prob=keep_rate)

            self.keep_rate = tf.Variable(keep_rate, trainable=False)
            stack_cell = tf.nn.rnn_cell.MultiRNNCell([get_cell(self.lstm_size,
                self.keep_rate) for _ in range(lstm_layers)])

            self.lstm_init_state = stack_cell.zero_state(batch_size, tf.float32)

            lstm_outputs, self.final_state = tf.nn.dynamic_rnn(stack_cell, inputs,
                    initial_state=self.lstm_init_state)

            # define the mapping
            seq_outputs = tf.concat(lstm_outputs, 1)
            reshape_lstm_outputs = tf.reshape(seq_outputs, shape=[-1, lstm_size])

            w = tf.Variable(tf.truncated_normal(shape=[lstm_size, vocab_size],
                stddev=0.1))
            b = tf.Variable(tf.zeros(vocab_size))

            logits = tf.matmul(reshape_lstm_outputs, w) + b

            self.probs = tf.nn.softmax(logits, name="probs")

        with tf.variable_scope("accuracy"):
            self.prediction = tf.squeeze(tf.argmax(logits, 1), name="class")
            self.prediction = tf.reshape(self.prediction, shape=[batch_size,
                n_steps])
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction,
                tf.cast(self.y_holder,
                tf.int64)), tf.float32), name="accuracy")

        with tf.variable_scope("loss"):
            logits = tf.reshape(logits, shape=[batch_size, n_steps, vocab_size])
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_holder,
                    logits=logits, name="loss"))

            self.global_step = tf.Variable(0, trainable=False)
            tf_learning_rate = tf.Variable(learning_rate, trainable=False)
            decay_learning_rate = tf.train.exponential_decay(tf_learning_rate,
                self.global_step, decay_steps, decay_rate)

            self.train_op = tf.contrib.layers.optimize_loss(self.loss,
                global_step=self.global_step, learning_rate=decay_learning_rate,
                optimizer="Adam")

        self.saver = tf.train.Saver()

    def train(self, sess, data_generator, log_fn):
        """add train op to the graph

        Args:
            sess: tf session
            data_generator: yield x, y inputs
            log_fn: the checkpoint prefix
        """
        for x, y in data_generator:
            start = time.time()
            accuracy_val, loss_val, global_step_val, self.initial_state_val, _ = sess.run(
                    [self.accuracy, self.loss, self.global_step, self.final_state, self.train_op],
                    {self.x_holder: x, self.y_holder: y,
                        self.lstm_init_state:self.initial_state_val})

            end = time.time()
            if global_step_val % 10 == 0:
                print("loss_val {} at step {} time {:.4f} s/batch accuracy {:.4f}".format(
                    loss_val, global_step_val, (end - start), accuracy_val))

        self.saver.save(sess, log_fn, global_step=global_step_val)

    def get_topn(self, probs, topn):
        """ random choice word id from the topn prob

        Args:
            probs: the prob of the networks output
            topn: the top int to return

        Returns:
            word id
        """
        probs = np.squeeze(probs)
        probs[np.argsort(probs)[:-topn]] = 0.0

        probs = probs / np.sum(probs)

        return np.random.choice(self.vocab_size, 1, p=probs)[0]


    def generate(self, sess, seeds, max_length):
        """ run the rnn with seed, and generate max_length text

        the network used for generation is batch_size=1, n_steps=1

        Args:
            sess: tf session
            seeds: vector for pre train the networks, to get the network real
                input
            max_length: the max generate length

        Returns:
            vector of word id
        """
        gen_result = [s for s in seeds]
        probs = np.ones(self.vocab_size, dtype="float32")

        initial_state = sess.run(self.lstm_init_state)

        for seed in gen_result:
            x = np.array([[1]], dtype="int32")
            x[0,0] = seed

            probs, initial_state = sess.run([self.probs, self.final_state],
                    {self.x_holder: x, self.lstm_init_state: initial_state,
                        self.keep_rate: 1.0})

        seed = self.get_topn(probs, 5)

        gen_result.append(seed)

        for _ in range(max_length):
            x = np.array([[1]], dtype="int32")
            x[0,0] = seed
            probs, initial_state = sess.run([self.probs, self.final_state],
                    {self.x_holder: x, self.lstm_init_state: initial_state})

            seed = self.get_topn(probs, 5)
            gen_result.append(seed)

        return gen_result

    def restore(self, sess, model_dir, fn):
        """ restore the latest model from model_fn

        Args:
            sess: tf session
            model_fn: the model prefix
        """
        model_dir
        model_prefix = tf.train.latest_checkpoint(model_dir)
        print(model_prefix)

        self.saver.restore(sess, model_prefix)
