import tensorflow as tf
import numpy as np

class RnnTextClassifyModel:
    """Using bidirect lstm model to classify short text, layers is 1
    """

    def __init__(self, class_number, learning_rate, gradients_norm,
            keep_rate, vocab_size, embedding_size, hidden_units_size):
        """ create graph
        """

        self.class_number = class_number
        self.learning_rate = learning_rate
        self.gradients_norm = gradients_norm
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_units_size = hidden_units_size

        with tf.variable_scope("inputs"):
            self.x_holder = tf.placeholder(tf.int32, shape=[None, None],
                    name="x")
            self.y_holder = tf.placeholder(tf.int32, shape=[None],
                    name="y")

            self.sequence_length = tf.placeholder(tf.int32, shape=[None],
                    name="sequence_length")

            self.embeddings = tf.get_variable(name="embeddings",
                    shape=[vocab_size, embedding_size], dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=0.1))

            embedding_inputs = tf.nn.embedding_lookup(self.embeddings,
                    self.x_holder)

        with tf.variable_scope("rnn"):
            self.keep_rate = tf.Variable(keep_rate, trainable=False)

            def _get_lstm_cell(hidden_units_size):
                lstm_forward_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units_size)

                drop_forward_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_forward_cell,
                    output_keep_prob=self.keep_rate)
                return drop_forward_cell

            multi_forward_cell = tf.nn.rnn_cell.MultiRNNCell(
                    [_get_lstm_cell(hidden_units_size) for _ in range(2)])
            multi_backward_cell= tf.nn.rnn_cell.MultiRNNCell(
                    [_get_lstm_cell(hidden_units_size) for _ in range(2)])

            # diff from the lstm language model, the lstm state zeroed for batch
            lstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(multi_forward_cell,
                    multi_backward_cell, embedding_inputs,
                    sequence_length=self.sequence_length, dtype=tf.float32)
            outputs = tf.concat(lstm_outputs, axis=2)
            # the outputs of padding word is zero
            sentence_outputs = tf.reduce_mean(outputs, axis=1)

        with tf.variable_scope("dense"):
            w = tf.get_variable(name="w", shape=[hidden_units_size * 2,
                class_number], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable(name="b", shape=[class_number],
                    dtype=tf.float32, initializer=tf.constant_initializer(0.0))

            logits = tf.matmul(sentence_outputs, w) + b

        with tf.variable_scope("prediction"):
            self.prediction = tf.squeeze(tf.argmax(logits, 1), name="class")
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction,
                tf.cast(self.y_holder,
                tf.int64)), tf.float32), name="accuracy")

        with tf.variable_scope("loss"):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                    labels=self.y_holder)
            self.loss = tf.reduce_mean(loss)

            # using the gradient clip to train the model
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                    gradients_norm)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))





