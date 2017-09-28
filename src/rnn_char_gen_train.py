import tensorflow as tf
import numpy as np
import codecs
import pickle
import copy
import os
from rnn_char_gen_model import RnnCharGenModel

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer("batch_size", 100, "")
tf.flags.DEFINE_integer("n_steps", 100, "")
tf.flags.DEFINE_integer("lstm_size", 128, "")
tf.flags.DEFINE_integer("lstm_layers", 2, "")
tf.flags.DEFINE_integer("decay_steps", 400, "")
tf.flags.DEFINE_float("decay_rate", 0.8, "")
tf.flags.DEFINE_float("learning_rate", 0.001, "")
tf.flags.DEFINE_float("keep_rate", 0.5, "")
tf.flags.DEFINE_bool("using_embedding", True, "")
tf.flags.DEFINE_integer("embedding_size", 128, "")
tf.flags.DEFINE_string("log_dir", "log", "")
tf.flags.DEFINE_string("seeds", "", "")
tf.flags.DEFINE_integer("gen_length", 100, "")
tf.flags.DEFINE_string("raw_text_fn", "", "")
tf.flags.DEFINE_string("lexicon", "", "")
tf.flags.DEFINE_integer("max_vocab_size", 3500, "")
tf.flags.DEFINE_bool("train", True, "")

class Lexicon:
    def __init__(self, text=None, lexicon_path=None, max_vocab_size=0):

        if lexicon_path:
            self.load(lexicon_path)
            return

        char_count = {}
        for char in text:
            if char in char_count:
                char_count[char] += 1
            else:
                char_count[char] = 1

        char_list = []
        for char in char_count:
            char_list.append([char, char_count[char]])

        char_list.sort(key=lambda x: x[1], reverse=True)
        if len(char_list) > max_vocab_size:
            char_list = char_list[:max_vocab_size]

        print(self.caculate_mean(char_list))

        self.vocab = ["<UNK>"]
        self.vocab += [s[0] for s in char_list]

        self.word_to_idx = {w:i for i, w in enumerate(self.vocab)}
        self.idx_to_word = dict(enumerate(self.vocab))

    def caculate_mean(self, char_list):
        count_sum = np.sum([x[1] for x in char_list])
        avg_id = np.mean([(i * x[1]) / count_sum for i, x in
            enumerate(char_list)])
        return avg_id

    def vocab_size(self):
        return len(self.vocab)

    def convert_word_to_id(self, word):
        if word in self.word_to_idx:
            return self.word_to_idx[word]
        else:
            return 0

    def convert_id_to_word(self, idx):
        return self.idx_to_word[idx]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.vocab, f)
            pickle.dump(self.word_to_idx, f)
            pickle.dump(self.idx_to_word, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.vocab = pickle.load(f)
            self.word_to_idx = pickle.load(f)
            self.idx_to_word = pickle.load(f)

    def text_to_ids(self, text):
        ids = []
        for c in text:
            ids.append(self.convert_word_to_id(c))
        return np.array(ids, dtype="int32")

    def ids_to_text(self, ids):
        text = []
        for i in ids:
            text.append(self.convert_id_to_word(i))
        return "".join(text)

def batch_generator(ids, batch_size, n_steps):
    ids = copy.copy(ids)
    batches = len(ids) // (batch_size * n_steps)
    ids = ids[:batches * batch_size * n_steps]

    ids = np.reshape(ids, newshape=[batch_size, batches, n_steps])

    np.random.shuffle(ids)

    for i in range(batches):
        x = ids[:,i,:]
        x = np.squeeze(x)
        y = np.zeros_like(x)
        y[:,:-1], y[:,-1] = x[:,1:], x[:,0]
        yield x, y

def main(_):

    if FLAGS.train:
        with codecs.open(FLAGS.raw_text_fn, encoding='utf-8') as f:
            text = f.read()

        lexicon = Lexicon(text=text, max_vocab_size=FLAGS.max_vocab_size)
        lexicon.save(FLAGS.lexicon)

        vocab_size = lexicon.vocab_size()

        print("vocab size: ", vocab_size)

        model = RnnCharGenModel(FLAGS.batch_size,
                FLAGS.n_steps,
                FLAGS.lstm_size,
                FLAGS.lstm_layers,
                vocab_size,
                FLAGS.decay_steps,
                FLAGS.decay_rate,
                FLAGS.learning_rate,
                FLAGS.keep_rate,
                FLAGS.using_embedding,
                FLAGS.embedding_size)

        ids = lexicon.text_to_ids(text)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        model.initial_state_val = sess.run(model.lstm_init_state)

        for epoch in range(50):
            print("training @ epoch {}".format(epoch))

            data_generator = batch_generator(ids, FLAGS.batch_size,
                FLAGS.n_steps)

            model.train(sess, data_generator, FLAGS.log_dir +
                    "/rnn_char_gen_model")

    else:
        lexicon = Lexicon(lexicon_path=FLAGS.lexicon)
        vocab_size = lexicon.vocab_size()

        print("vocab size: ", vocab_size)

        model = RnnCharGenModel(1,
                1,
                FLAGS.lstm_size,
                FLAGS.lstm_layers,
                vocab_size,
                FLAGS.decay_steps,
                FLAGS.decay_rate,
                FLAGS.learning_rate,
                1.0,
                FLAGS.using_embedding,
                FLAGS.embedding_size)
        sess = tf.Session()
        model.restore(sess, FLAGS.log_dir, "rnn_char_gen_model")
        seeds_ids = lexicon.text_to_ids(FLAGS.seeds)
        ids = model.generate(sess, seeds_ids, FLAGS.gen_length)

        text = lexicon.ids_to_text(ids)
        print(text)

if __name__ == "__main__":
    tf.app.run()
