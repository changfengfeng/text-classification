'''
using word embedding and cnn to do short text classification
'''
import tensorflow as tf
import numpy as np

class TextClassifyCnnModel:
    """ Define the text classification cnn model

    Define the training graph structure
    Define the inference graph sructure

    """
    def __init__(self, filter_list, filter_size, max_sequence_length,
            vocabulary_size, embedding_size, class_label_size):
        """ Init some variables

        Args:
            filter_list: define the different height filter, [3,4,5], like ngrams
            filter_size: define the different filter size for fiter, like channel
                       size
            max_sequence_length: define the max length of text
            vocabulary_size: give the vocabulary size
            embedding_size: give the word embedding size
            class_label_size: the target class label size
        """

        self.filter_list = filter_list
        self.filter_size = filter_size
        self.max_sequence_length = max_sequence_length
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.class_label_size = class_label_size

    def create_train_graph(self, using_bn, using_chunk):
        """ Create the training graph

        layer 1: input x is [None, max_sequence_length], input y is [None]
        layer 2: embedding shape is  [vocabulary_size, embedding_size], using id
        0 for <unk> and padding word
        layer 3: convlution, input is [None, max_sequence_length,
        embedding_size, 1], output is [None, height, 1, filter_size], there are
        len(filter_list) outputs, height_i = max_sequence_length -
        filter_list[i] + 1, here we will have len(filter_list) convlution.
        layer 4: max_pooling, input is [None, height, 1, filter_size], output is
        [None, 1, 1, filter_size]
        layer 5: dense. input is [None, len(filter_list) * filter_size], output
        is [None, class_label_sizes]
        layer 6: softmax

        Args:
            using_bn: True to use batch normalization
            using_chunk: True to use chunk pooling
        """
        print("embedding in graph ", self.vocabulary_size, self.embedding_size)

        with tf.variable_scope("text_classify_cnn_graph"):
            self.x_holder = tf.placeholder(tf.int32, shape=[None,
                self.max_sequence_length], name="input_x")
            self.y_holder = tf.placeholder(tf.int32, shape=[None], name="input_y")
            self.embeddings = tf.get_variable(name="word_embedding",
                    shape=[self.vocabulary_size, self.embedding_size],
                    dtype=tf.float32, trainable=True,
                    initializer=tf.random_normal_initializer(stddev=0.01))
            self.is_training = tf.get_variable(name="training", shape=[],
                    dtype=tf.bool,
                    initializer=tf.constant_initializer(True),
                    trainable=False)


            x_embedding = tf.nn.embedding_lookup(self.embeddings, self.x_holder)
            x_embedding = tf.expand_dims(x_embedding, axis=-1)

            conv_outputs = []
            for i, ngram in enumerate(self.filter_list):
                conv = tf.layers.conv2d(x_embedding, filters=self.filter_size,
                        kernel_size=[ngram, self.embedding_size],
                        padding='valid', activation=tf.nn.relu,
                        name="conv%d" % i)

                if using_bn:
                    conv_bn = tf.layers.batch_normalization(conv,
                    name="bn%d" % i, training=self.is_training)
                else:
                    conv_bn = conv

                if using_chunk:
                    # using chunk=2, split conv_bn to two tensor
                    sub_conv_height_1 = (self.max_sequence_length - ngram + 1) // 2
                    sub_conv_height_2 = self.max_sequence_length - ngram + 1 - sub_conv_height_1
                    conv_bn_1, conv_bn_2 = tf.split(conv_bn, [sub_conv_height_1,
                        sub_conv_height_2], axis=1)

                    pooling_1 = tf.layers.max_pooling2d(conv_bn_1,
                            pool_size=[sub_conv_height_1, 1], strides=[1, 1],
                            padding="valid", name="chunk_pooling1_%d" % i)
                    pooling_1 = tf.reshape(pooling_1, shape=[-1, self.filter_size],
                            name="chunk_pooling_reshape1_%d" % i)

                    pooling_2 = tf.layers.max_pooling2d(conv_bn_2,
                            pool_size=[sub_conv_height_2, 1], strides=[1, 1],
                            padding="valid", name="chunk_pooling2_%d" % i)
                    pooling_2 = tf.reshape(pooling_2, shape=[-1, self.filter_size],
                            name="chunk_pooling_reshape2_%d" % i)

                    pooling = tf.concat([pooling_1, pooling_2], axis=1)
                else:
                    pooling = tf.layers.max_pooling2d(conv_bn,
                            pool_size=[self.max_sequence_length - ngram + 1, 1], strides=[1,1],
                            padding="valid", name="pooling%d" % i)
                    pooling = tf.reshape(pooling, shape=[-1, self.filter_size],
                            name="pooling_reshape%d" % i)

                conv_outputs.append(pooling)

            convs = tf.concat(conv_outputs, axis=-1)
            convs_dropout = tf.layers.dropout(convs, rate=0.5, training=self.is_training)

            if using_chunk:
                w = tf.get_variable(name="dense_w", shape=[2 * len(self.filter_list) *
                    self.filter_size, self.class_label_size], dtype=tf.float32,
                    initializer=tf.random_normal_initializer(stddev=0.01))
            else:
                w = tf.get_variable(name="dense_w", shape=[len(self.filter_list) *
                    self.filter_size, self.class_label_size], dtype=tf.float32,
                    initializer=tf.random_normal_initializer(stddev=0.01))

            b = tf.get_variable(name="dense_b", shape=[self.class_label_size],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=0.01))

            logits = tf.matmul(convs_dropout, w) + b

        with tf.variable_scope("prediction"):
            softmax = tf.nn.softmax(logits)
            self.prediction = tf.squeeze(tf.argmax(logits, 1), name="class")
            self.prediction_prob = tf.gather(softmax, self.prediction, axis=-1,
                name = "class_prob")
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction,
                tf.cast(self.y_holder,
                tf.int64)), tf.float32), name="accuracy")

        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_holder,
                    logits=logits, name="loss"))
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.global_step = tf.Variable(0, trainable=False)
                self.learning_rate = tf.Variable(0.001, trainable=False)
                learning_rate = tf.train.exponential_decay(self.learning_rate,
                        self.global_step, 400, 0.8)
                self.train_op = tf.contrib.layers.optimize_loss(self.loss,
                        global_step = self.global_step,
                        learning_rate = learning_rate, optimizer="Adam")
                #optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
                #self.train_op = optimizer.minimize(self.loss, name="train_op")

    def create_inference_graph(self):
        pass

    def freeze_model(self):
        pass
