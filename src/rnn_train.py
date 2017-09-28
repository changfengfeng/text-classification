import tensorflow as tf
import numpy as np
import pickle
import word2vec as w2v
from rnn_model import RnnTextClassifyModel

class TextClassifyTrainer:
    """ Train the cnn model

    using the pretrained word embedding model
    """

    def __init__(self, train_data_path, test_data_path, embedding_path):
        """
        Args:
            train_data_path: the training data file path. the format of training
                            data is "class_name  text"
            test_data_path: the validate data file path. format is same as
                                train data
            embedding_path: the word2vect embedding path

        """

        self.train_data = self.load_data(train_data_path)
        self.test_data = self.load_data(test_data_path)
        self.load_word2vect(embedding_path)

        self.labels = sorted(set([data[0] for data in self.train_data]))
        self.labels_to_idx = {labels:i for i, labels in enumerate(self.labels)}

        np.random.shuffle(self.test_data)
        np.random.shuffle(self.train_data)

    def train(self, epoches, batch_size):
        """ create training data with class id, word id, do padding with word id
            0

        Args:
            epoches: integer
            batch_size: integer
        """
        def _padding_batch(x_inputs, y_inputs):
             # x_inputs is 2-d array
             max_length = max([len(x) for x in x_inputs])
             real_length = [len(x) for x in x_inputs]

             x_outputs = []
             for x in x_inputs:
                 padding_size = max_length - len(x)
                 x_outputs.append(
                         np.concatenate([np.array(x), np.zeros(padding_size,
                         dtype="int32")]))
             return np.array(x_outputs), np.array(y_inputs), real_length


        x_inputs, y_inputs = self.convert_data_to_model_input(self.train_data)
        test_x_inputs, test_y_inputs = self.convert_data_to_model_input(self.test_data, add_unknow_words=False)
        test_x_inputs, test_y_inputs, test_real_length = _padding_batch(test_x_inputs,
                test_y_inputs)

        self.save_lexicon("log/lexicon")

        train_x_inputs = x_inputs[0:11000]
        train_y_inputs = y_inputs[0:11000]

        validate_x_inputs = x_inputs[11000:]
        validate_y_inputs = y_inputs[11000:]
        validate_x_inputs, validate_y_inputs, validate_real_length = _padding_batch(
                validate_x_inputs, validate_y_inputs)

        assert len(train_y_inputs) == len(train_x_inputs)
        assert len(validate_y_inputs) == len(validate_x_inputs)
        print("train {} validate {} test {}".format(len(train_y_inputs),
            len(validate_y_inputs), len(test_y_inputs)))
        assert len(self.vocab) == len(self.embeddings)

        # do training
        batches = len(train_y_inputs) // batch_size

        rnn_model = RnnTextClassifyModel(
                class_number=len(self.labels), learning_rate=0.01,
                gradients_norm=5, keep_rate=0.5, vocab_size=len(self.vocab),
                embedding_size=self.embedding_size, hidden_units_size=128)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # must assigned
            embedding_input = tf.constant(np.array(self.embeddings), dtype=tf.float32)
            assign_embedding_op = tf.assign(rnn_model.embeddings, embedding_input)
            embedding_in_graph = sess.run(assign_embedding_op);

            global_step = 0
            for epoch in range(epoches):
                print("training @ epoch ", epoch)
                for i in range(batches):

                    x_inputs_batch = train_x_inputs[i * batch_size:(i+1) *
                            batch_size]
                    y_inputs_batch = train_y_inputs[i * batch_size:(i+1) *
                            batch_size]
                    x_inputs_batch, y_inputs_batch, real_length = _padding_batch(
                        x_inputs_batch, y_inputs_batch)

                    loss_val, _ = sess.run(
                            [rnn_model.loss, rnn_model.train_op],
                            {rnn_model.x_holder:x_inputs_batch,
                             rnn_model.y_holder:y_inputs_batch,
                             rnn_model.sequence_length: real_length})

                    print("loss {} @ step {}".format(loss_val, global_step))

                    #saver.save(sess, "log/cnn_model", global_step=global_step)
                    global_step += 1

                    if global_step % 100 == 0:
                        print("______validating")
                        accuracy_val = sess.run(rnn_model.accuracy,
                            {rnn_model.x_holder: validate_x_inputs,
                             rnn_model.y_holder: validate_y_inputs,
                             rnn_model.sequence_length: validate_real_length,
                             rnn_model.keep_rate : 1.0})
                        print("______valiation_accuracy {} at step {}".format(accuracy_val,
                            global_step))

                        accuracy_val = sess.run(rnn_model.accuracy,
                            {rnn_model.x_holder: x_inputs_batch,
                             rnn_model.y_holder: y_inputs_batch,
                             rnn_model.sequence_length: real_length,
                             rnn_model.keep_rate : 1.0})
                        print("______train_accuracy {} at step {}".format(accuracy_val,
                            global_step))


                if batches * batch_size < len(train_y_inputs):
                    x_inputs_batch = train_x_inputs[batches * batch_size:]
                    y_inputs_batch = train_y_inputs[batches * batch_size:]
                    x_inputs_batch, y_inputs_batch, real_length = _padding_batch(
                            x_inputs_batch, y_inputs_batch)

                    loss_val, _ = sess.run(
                            [rnn_model.loss, rnn_model.train_op],
                            {rnn_model.x_holder:x_inputs_batch,
                             rnn_model.y_holder:y_inputs_batch,
                             rnn_model.sequence_length: real_length})

                    print("loss {} @ step {}".format(loss_val, global_step))
                    global_step += 1

                # do evaluate on test data
                print("______test accuracy on epoch ", epoch)
                accuracy_val = sess.run(rnn_model.accuracy,
                            {rnn_model.x_holder: test_x_inputs,
                             rnn_model.y_holder: test_y_inputs,
                             rnn_model.sequence_length: test_real_length,
                             rnn_model.keep_rate : 1.0})

                print("______test_accuracy {} at epoch {}".format(accuracy_val,
                            epoch))

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

    def save_lexicon(self, lexicon_path):
        """ Saving the label_to_idx and word_to_idx lexicon
        """
        with open(lexicon_path, "wb") as f:
            pickle.dump(self.labels_to_idx, f)
            pickle.dump(self.word_to_idx, f)

    def load_lexicon(self, lexicon_path):
        """ load the leixcon
        """
        with open(lexicon_path, "rb") as f:
            self.labels_to_idx = pickle.load(f)
            self.word_to_idx = pickle.load(f)

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

        train_words_set = set([word for text in self.train_data for word in
            text[1].split(" ")])

        for w in model.vocab:
            if w in train_words_set:
              self.word_to_idx[w] = len(self.vocab)
              self.vocab.append(w)
              self.embeddings.append(model[w])

        del model

    def convert_data_to_model_input(self, origin_datas, add_unknow_words=True):
        """ convert data to x_input, y_input

        convert class label to class id
        convert words to word id

        Args:
            origin_datas: the origin data
            add_unknow_words: True add unknow words to embeddings and   vocab
        Returns:
            (x_inputs, y_inputs)
        """
        def _word_to_id(word):
            if word in self.word_to_idx:
                return self.word_to_idx[word]
            elif add_unknow_words:
                self.word_to_idx[word] = len(self.vocab)
                self.vocab.append(word)
                self.embeddings.append(np.random.uniform(-0.25, 0.25,
                    self.embedding_size))
                return self.word_to_idx[word]
            else:
                # padding id
                return 0

        x_inputs = []
        y_inputs = []

        for data in origin_datas:
            y_inputs.append(self.labels_to_idx[data[0]])

            words = data[1].split(" ")
            ids = list(map(_word_to_id, words))

            x_inputs.append(ids)

        return x_inputs, y_inputs

if __name__ == "__main__":
    trainer = TextClassifyTrainer("data/domain324.train.seg2",
        "data/domain324.test.seg2", "model/vectors.bin")
    trainer.train(45, 128)

