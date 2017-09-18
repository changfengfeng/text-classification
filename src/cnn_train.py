import tensorflow as tf
import numpy as np
import word2vec as w2v
from cnn_model import TextClassifyCnnModel

class TextClassifyTrainer:
    """ Train the cnn model

    using the pretrained word embedding model
    """

    def __init__(self, train_data_path, validate_data_path, embedding_path):
        """
        Args:
            train_data_path: the training data file path. the format of training
                            data is "class_name  text"
            validate_data_path: the validate data file path. format is same as
                                train data
            embedding_path: the word2vect embedding path

        """
        self.load_word2vect(embedding_path)
        self.train_data = self.load_data(train_data_path)
        self.validate_data = self.load_data(validate_data_path)

        np.random.shuffle(self.validate_data)
        np.random.shuffle(self.train_data)
        self.validate_data = self.validate_data[:1000]

        self.labels = set([data[0] for data in self.train_data])
        self.labels_to_idx = {labels:i for i, labels in enumerate(self.labels)}

        self.max_seq_len = np.max([len(data[1]) for data in self.train_data])

        print("max sequence length: ", self.max_seq_len)

    def train(self, epoches, batch_size):
        """ create training data with class id, word id, do padding with word id
            0

        Args:
            epoches: integer
            batch_size: integer
        """
        def _word_to_id(word):
            if word in self.word_to_idx:
                return self.word_to_idx[word]
            else:
                self.word_to_idx[word] = len(self.vocab)
                self.vocab.append(word)
                self.embeddings.append(np.random.uniform(-0.25, 0.25,
                    self.embedding_size))
                return self.word_to_idx[word]

        x_inputs = []
        y_inputs = []

        for i, data in enumerate(self.train_data):

            y_inputs.append(self.labels_to_idx[self.train_data[i][0]])

            words = self.train_data[i][1].split(" ")
            ids = list(map(_word_to_id, words))

            padding_size = self.max_seq_len - len(ids)
            if padding_size > 1:
                before = padding_size // 2
                end = padding_size - before
                padding_array = np.reshape(np.concatenate(
                        (np.zeros([before, 1], "int32"),
                         np.reshape(ids, [-1, 1]),
                         np.zeros([end, 1], "int32"))), newshape=-1)
            elif padding_size == 1:
                padding_array = np.array(ids.append(0))

            assert len(padding_array) == self.max_seq_len
            x_inputs.append(padding_array)

        x_inputs = np.array(x_inputs)
        y_inputs = np.array(y_inputs)
        assert len(y_inputs) == len(x_inputs)
        assert len(self.vocab) == len(self.embeddings)

        # do training
        batches = len(y_inputs) // batch_size

        cnn_model = TextClassifyCnnModel(
                [2,3,4], 100, self.max_seq_len,
                len(self.vocab), self.embedding_size, len(self.labels))

        cnn_model.create_train_graph(True, True)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            global_step = 0
            for epoch in range(epoches):
                print("training @ epoch ", epoch)
                for i in range(batches):
                    x_inputs_batch = x_inputs[i * batch_size:(i+1) *
                            batch_size, :]
                    y_inputs_batch = y_inputs[i * batch_size:(i+1) *
                            batch_size]

                    # must assigned
                    embedding_input = tf.constant(np.array(self.embeddings), dtype=tf.float32)
                    assign_embedding_op = tf.assign(cnn_model.embeddings, embedding_input)
                    embedding_in_graph = sess.run(assign_embedding_op);

                    loss_val, _ = sess.run(
                            [cnn_model.loss, cnn_model.train_op],
                            {cnn_model.x_holder:x_inputs_batch,
                             cnn_model.y_holder:y_inputs_batch})

                    print("loss {} @ step {}".format(loss_val, global_step))
                    global_step += 1


    def evaluate(self, sess, x_inputs, y_inputs):
        pass

    def load_data(self, data_path):
        """ load the training or validate data

        Args:
            data_path: file path

        Returns:
            data list
        """
        data = []
        with open(data_path, "r") as f:
            data = [line.split("\t") for line in f if len(line.strip()) > 0 and
                    line.strip()[0] != '#']
        return data


    def load_word2vect(self, file_path):
        """ load the pretrained word2vect model, create the word2idx, idex2word,
        embedding dict, word_id 0 is for padding token <pad>

        Args:
            file_path: the absoluate path of the word2vect model
        """
        self.embeddings = []
        self.word_to_idx = {'<pad>' : 0}
        self.vocab = ['<pad>']

        model = w2v.load(file_path)
        self.embedding_size = model.vectors.shape[1]
        pad_embedding = np.zeros(self.embedding_size, "float32")
        self.embeddings.append(pad_embedding)

        for w in model.vocab:
            self.word_to_idx[w] = len(self.vocab)
            self.vocab.append(w)
            self.embeddings.append(model[w])

        del model

if __name__ == "__main__":
    trainer = TextClassifyTrainer("data/domain324.train.seg",
        "data/domain324.test.seg", "model/vectors.bin")
    trainer.train(3, 128)
