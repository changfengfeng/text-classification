import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import pickle
import word2vec as w2v
from cnn_model import TextClassifyCnnModel

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

        self.max_seq_len = np.max([len(data[1]) for data in self.train_data])

        np.random.shuffle(self.test_data)
        np.random.shuffle(self.train_data)

        print("max sequence length: ", self.max_seq_len)

    def train(self, epoches, batch_size):
        """ create training data with class id, word id, do padding with word id
            0

        Args:
            epoches: integer
            batch_size: integer
        """
        x_inputs, y_inputs = self.convert_data_to_model_input(self.train_data)
        test_x_inputs, test_y_inputs = self.convert_data_to_model_input(self.test_data, add_unknow_words=False)

        self.save_lexicon("log/lexicon")

        train_x_inputs = np.array(x_inputs[0:11000])
        train_y_inputs = np.array(y_inputs[0:11000])

        validate_x_inputs = np.array(x_inputs[11000:])
        validate_y_inputs = np.array(y_inputs[11000:])

        assert len(train_y_inputs) == len(train_x_inputs)
        assert len(validate_y_inputs) == len(validate_x_inputs)
        print("train {} validate {} test {}".format(len(train_y_inputs),
            len(validate_y_inputs), len(test_y_inputs)))
        assert len(self.vocab) == len(self.embeddings)

        # do training
        batches = len(train_y_inputs) // batch_size

        cnn_model = TextClassifyCnnModel(
                [2,3,4], 300, self.max_seq_len,
                len(self.vocab), self.embedding_size, len(self.labels))

        cnn_model.create_train_graph(True, True)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # must assigned
            embedding_input = tf.constant(np.array(self.embeddings), dtype=tf.float32)
            assign_embedding_op = tf.assign(cnn_model.embeddings, embedding_input)
            embedding_in_graph = sess.run(assign_embedding_op);

            global_step = 0
            for epoch in range(epoches):
                print("training @ epoch ", epoch)
                for i in range(batches):
                    x_inputs_batch = train_x_inputs[i * batch_size:(i+1) *
                            batch_size, :]
                    y_inputs_batch = train_y_inputs[i * batch_size:(i+1) *
                            batch_size]

                    loss_val, _ = sess.run(
                            [cnn_model.loss, cnn_model.train_op],
                            {cnn_model.x_holder:x_inputs_batch,
                             cnn_model.y_holder:y_inputs_batch})

                    print("loss {} @ step {}".format(loss_val, global_step))

                    saver.save(sess, "log/cnn_model", global_step=global_step)
                    global_step += 1

                    if global_step % 100 == 0:
                        print("______validating")
                        accuracy_val = sess.run(cnn_model.accuracy,
                            {cnn_model.x_holder: validate_x_inputs,
                             cnn_model.y_holder: validate_y_inputs,
                             cnn_model.is_training : False})
                        print("______valiation_accuracy {} at step {}".format(accuracy_val,
                            global_step))

                        accuracy_val = sess.run(cnn_model.accuracy,
                            {cnn_model.x_holder: x_inputs_batch,
                             cnn_model.y_holder: y_inputs_batch,
                             cnn_model.is_training : False})
                        print("______train_accuracy {} at step {}".format(accuracy_val,
                            global_step))


                if batches * batch_size < len(train_y_inputs):
                    x_inputs_batch = train_x_inputs[batches * batch_size:, :]
                    y_inputs_batch = train_y_inputs[batches * batch_size:]

                    loss_val, _ = sess.run(
                            [cnn_model.loss, cnn_model.train_op],
                            {cnn_model.x_holder:x_inputs_batch,
                             cnn_model.y_holder:y_inputs_batch})

                    saver.save(sess, "log/cnn_model", global_step=global_step)
                    print("loss {} @ step {}".format(loss_val, global_step))
                    global_step += 1

                # do evaluate on test data
                print("______test accuracy on epoch ", epoch)
                accuracy_val = sess.run(cnn_model.accuracy,
                            {cnn_model.x_holder: test_x_inputs,
                             cnn_model.y_holder: test_y_inputs,
                             cnn_model.is_training : False})

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

        return x_inputs, y_inputs

def test_with_freeze_model(x_inputs_val, y_inputs_val, final_model_path):
    """ Test with freezed model
    """
    with tf.gfile.GFile(final_model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    tf.import_graph_def(graph_def, input_map=None, return_elements=None,
            name="cnn_model", op_dict=None, producer_op_list=None)

    graph = tf.get_default_graph()
    x_inputs = graph.get_tensor_by_name("cnn_model/text_classify_cnn_graph/input_x:0")
    y_inputs = graph.get_tensor_by_name("cnn_model/text_classify_cnn_graph/input_y:0")
    predict_class = graph.get_tensor_by_name("cnn_model/prediction/class:0")
    predict_prob = graph.get_tensor_by_name("cnn_model/prediction/class_prob:0")
    accuracy = graph.get_tensor_by_name("cnn_model/prediction/accuracy:0")
    training = graph.get_tensor_by_name("cnn_model/text_classify_cnn_graph/training:0")

    with tf.Session() as sess:
        class_val, prob_val, accuracy_val = sess.run([predict_class,
            predict_prob, accuracy],
            {x_inputs: x_inputs_val, y_inputs: y_inputs_val})
        print(accuracy_val)

def test_with_latest_chk(trainer, trained_model_path):
    """ Using the training checkpointed latest model to test
    """
    model_prefix = tf.train.latest_checkpoint(trained_model_path)
    print(model_prefix)

    trainer.convert_data_to_model_input(trainer.train_data)
    trainer.load_lexicon(trained_model_path + "/lexicon")
    x_inputs_val, y_inputs_val = trainer.convert_data_to_model_input(trainer.test_data, add_unknow_words=False)

    cnn_model = TextClassifyCnnModel(
                [2,3,4], 300, trainer.max_seq_len,
                len(trainer.vocab), trainer.embedding_size, len(trainer.labels))

    cnn_model.create_train_graph(True, True)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, model_prefix)

        graph = tf.get_default_graph()

        accuracy_val = sess.run(
                [cnn_model.accuracy],
                {cnn_model.x_holder: x_inputs_val, cnn_model.y_holder:
                    y_inputs_val, cnn_model.is_training: False})
        print(accuracy_val)

def freeze_model(trained_model_path, final_model_path):
    """ Freese the trained model

    Args:
        trained_model_path: the trained model path
        final_model_path: the final model file path
    """
    model_prefix = tf.train.latest_checkpoint(trained_model_path)
    output_operateions = "prediction/class,prediction/class_prob,prediction/accuracy"
    saver = tf.train.import_meta_graph(model_prefix + ".meta", clear_devices=True)

    graph = tf.get_default_graph()
    graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, model_prefix)
        training = graph.get_tensor_by_name("text_classify_cnn_graph/training:0")
        training_op = tf.assign(training, tf.constant(False))
        sess.run(training_op)
        output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def,
                output_operateions.split(","))
        with tf.gfile.GFile(final_model_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())

if __name__ == "__main__":
    trainer = TextClassifyTrainer("data/domain324.train.seg2",
        "data/domain324.test.seg2", "model/vectors.bin")
    #trainer.train(45, 128)
    #freeze_model("log", "model/cnn_model.pb")

    trainer.load_lexicon("log/lexicon")
    x_inputs, y_inputs = trainer.convert_data_to_model_input(trainer.test_data,
            add_unknow_words=False)
    test_with_freeze_model(x_inputs, y_inputs, "model/cnn_model.pb")
    #test_with_latest_chk(trainer, "log")
