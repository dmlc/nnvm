import nnvm
import tvm
import numpy as np
import os

# Tensorflow imports
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from google.protobuf import text_format
from tensorflow.python.framework import graph_util
from tensorflow.python.ops import array_ops
dir(tf.contrib)

#np.set_printoptions(threshold=np.nan)
#np.set_printoptions(suppress=True)


DATA_DIR = './ptb/'
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
sample_repo = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/'
sample_data_file = 'simple-examples.tgz'
sample_url = os.path.join(sample_repo, sample_data_file)

ptb_repo = 'https://github.com/joyalbin/dmlc_store/tree/master/trained-models/tf/'
ptb_checkpoint_file = DATA_DIR+'checkpoint'
ptb_checkpoint_url = os.path.join(ptb_repo, ptb_checkpoint_file)
ptb_data_file = DATA_DIR+'model.ckpt.data-00000-of-00001'
ptb_data_url = os.path.join(ptb_repo, ptb_data_file)
ptb_index_file = DATA_DIR+'model.ckpt.index'
ptb_index_url = os.path.join(ptb_repo, ptb_index_file)
ptb_meta_file = DATA_DIR+'model.ckpt.meta'
ptb_meta_url = os.path.join(ptb_repo, ptb_meta_file)
ptb_graph_file = DATA_DIR+'run_graph.pbtxt'
ptb_graph_url = os.path.join(ptb_repo, ptb_graph_file)
'''
from mxnet.gluon.utils import download
download(sample_url, DATA_DIR+sample_data_file)
download(ptb_checkpoint_url, ptb_checkpoint_file)
download(ptb_index_url, ptb_index_file)
download(ptb_meta_url, ptb_meta_file)
download(ptb_data_url, ptb_data_file)
download(ptb_graph_url, ptb_graph_file)

import tarfile
t = tarfile.open(DATA_DIR+sample_data_file, 'r')
t.extractall(DATA_DIR)
'''

###############################################################################
# Reader
# ---------------------------------------------
# Read the PTB sample data input to create vocabulary
import collections
import os

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().encode("utf-8").decode("utf-8").replace("\n", "<eos>").split()

def _build_vocab(filename):
  data = _read_words(filename)
  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  #for python 3.x
  id_to_word = dict((v, k) for k, v in word_to_id.items())
  return word_to_id, id_to_word

def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]

def ptb_raw_data(data_path=None, prefix="ptb"):
  """Load PTB raw data from data directory "data_path".
  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.
  The PTB dataset comes from Tomas Mikolov's webpage:
  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.
  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, prefix + ".train.txt")
  valid_path = os.path.join(data_path, prefix + ".valid.txt")
  test_path = os.path.join(data_path, prefix + ".test.txt")

  word_to_id, id_2_word = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary, word_to_id, id_2_word



###############################################################################
# Vocabulary
# ---------------------------------------------
# Create vocabulary from the input sample data provided
raw_data = ptb_raw_data(DATA_DIR+'simple-examples/data/')
train_data, valid_data, test_data, vocabulary, word_to_id, id_to_word = raw_data
vocab_size = len(word_to_id)

###############################################################################
# PTB Configuration
# ---------------------------------------------
# PTB test model configurations for sampling
class SmallConfig(object):
  """Small config."""
  is_char_model = False
  optimizer = 'AdamOptimizer'
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 4
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000

def get_config():
  return SmallConfig()

config = get_config()
config.vocab_size = config.vocab_size if config.vocab_size < vocab_size else vocab_size
eval_config = get_config()
eval_config.vocab_size = eval_config.vocab_size if eval_config.vocab_size < vocab_size else vocab_size
eval_config.batch_size = batch_size = 1
eval_config.num_steps = num_steps = 1


###############################################################################
# Restore Model
# ---------------------------------------------
# Restore the PTB pre-trained model checkpoints
def _parse_input_graph_proto(input_graph, input_binary):
  """Parser input tensorflow graph into GraphDef proto."""
  if not tf.gfile.Exists(input_graph):
    print("Input graph file '" + input_graph + "' does not exist!")
    return -1
  input_graph_def = graph_pb2.GraphDef()
  mode = "rb" if input_binary else "r"
  with tf.gfile.FastGFile(input_graph, mode) as f:
    if input_binary:
      input_graph_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read(), input_graph_def)
  return input_graph_def

#create graph from saved meta files.
meta_path = ptb_meta_file # Your .meta file
input_binary = False

with tf.Session() as sess:
    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)
    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint(DATA_DIR))
    graph_def = _parse_input_graph_proto(ptb_graph_file, input_binary)

    final_graph_def = graph_util.convert_variables_to_constants(
        sess,
        graph_def,
        ['Model/Softmax'],
        variable_names_whitelist=None,
        variable_names_blacklist=None)
sym, params = nnvm.frontend.from_tensorflow(final_graph_def)


######################################################################
# Compile the model on NNVM
# ---------------------------------------------
# We should be familiar with the process right now.
import nnvm.compiler
target = 'llvm'
batch_size = 1
num_steps = 1
num_hidden = 200
num_layers = 2
input_shape = (batch_size, num_steps)
output_shape = (1, 200)
shape_dict = {'Model/Placeholder': input_shape,
              'LSTMBlockCell_param_in_state_c':(num_layers, batch_size, num_hidden),
              'LSTMBlockCell_param_in_state_h':(num_layers, batch_size, num_hidden)}
dtype_dict = {'Model/Placeholder': 'int32',
              'LSTMBlockCell_param_in_state_c':'float32',
              'LSTMBlockCell_param_in_state_h':'float32'}
graph, lib, params = nnvm.compiler.build(sym, target, shape_dict,
                                         dtype=dtype_dict, params=params)

lib.export_library("imagenet_tensorflow.so")
with open("imagenet_tensorflow.json", "w") as fo:
    fo.write(graph.json())
with open("imagenet_tensorflow.params", "wb") as fo:
    fo.write(nnvm.compiler.save_param_dict(params))

######################################################################
# Execute on TVM
# ---------------------------------------------
# The process is no different from other example
from tvm.contrib import graph_runtime
ctx = tvm.cpu(0)
out_dtype = 'float32'
m = graph_runtime.create(graph, lib, ctx)

######################################################################
# Predition
# ---------------------------------------------
# Create sample rediction results
def pick_from_weight(weight, pows=1.0):
    weight = weight**pows
    t = np.cumsum(weight)
    s = np.sum(weight)
    #return int(np.searchsorted(t, np.random.rand(1) * s))
    return int(np.searchsorted(t, 0.5 * s))

out_sample_shape = (1, 10000)
out_state_shape = (num_layers,2, batch_size, num_hidden)

def do_sample(model, data, in_states, num_samples):
    """Sampled from the model"""
    samples = []
    state = in_states
    sample = None
    for x in data:
        word_id = np.full((batch_size, num_steps), x, dtype="int32")
        model.set_input('Model/Placeholder', tvm.nd.array(word_id.astype("int32")))
        in_state_tup = np.split(state, indices_or_sections=2, axis=1)
        in_state_c = np.reshape(in_state_tup[0], (num_layers, batch_size, num_hidden))
        in_state_h = np.reshape(in_state_tup[1], (num_layers, batch_size, num_hidden))
        model.set_input('LSTMBlockCell_param_in_state_c', tvm.nd.array(in_state_c.astype("float32")))
        model.set_input('LSTMBlockCell_param_in_state_h', tvm.nd.array(in_state_h.astype("float32")))
        model.set_input(**params)
        model.run()
        tvm_output = model.get_output(0, tvm.nd.empty(out_sample_shape, out_dtype)).asnumpy()
        state_output = model.get_output(1, tvm.nd.empty(out_state_shape, out_dtype)).asnumpy()
        state = state_output
        sample = pick_from_weight(tvm_output[0])

    if sample is not None:
        samples.append(sample)
    else:
        samples.append(0)

    k = 1
    while k < num_samples:
        word_id = np.full((batch_size, num_steps), samples[-1], dtype="int32")
        model.set_input('Model/Placeholder', tvm.nd.array(word_id.astype("int32")))
        in_state_tup = np.split(state, indices_or_sections=2, axis=1)
        in_state_c = np.reshape(in_state_tup[0], (num_layers, batch_size, num_hidden))
        in_state_h = np.reshape(in_state_tup[1], (num_layers, batch_size, num_hidden))
        model.set_input('LSTMBlockCell_param_in_state_c', tvm.nd.array(in_state_c.astype("float32")))
        model.set_input('LSTMBlockCell_param_in_state_h', tvm.nd.array(in_state_h.astype("float32")))
        model.set_input(**params)
        model.run()
        tvm_output = model.get_output(0, tvm.nd.empty(out_sample_shape, out_dtype)).asnumpy()
        state_output = model.get_output(1, tvm.nd.empty(out_state_shape, out_dtype)).asnumpy()
        state = state_output
        sample = pick_from_weight(tvm_output[0])
        samples.append(sample)
        k += 1

    return samples, state

def pretty_print(items, is_char_model, id2word):
    if not is_char_model:
        return ' '.join([id2word[x] for x in items])
    else:
        return ''.join([id2word[x] for x in items]).replace('_', ' ')


###############################################################################
# Input words
# ---------------------------------------------
# The input data provide the context to predict next word
from sys import version_info
while True:
    if version_info[0] < 3:
        inpt = raw_input("Enter your sample prefix: ")
        cnt = int(raw_input("Sample size: "))
    else:
        #python 3.x
        inpt = input("Enter your sample prefix: ")
        cnt = int(input("Sample size: "))

    in_state = np.full((num_layers, 2, batch_size, num_hidden), 0, dtype="float32")
    seed_for_sample = inpt.split()
    print("Seed: %s" % pretty_print([word_to_id[x] for x in seed_for_sample], False, id_to_word))
    samples, _ = do_sample(m, [word_to_id[word] for word in seed_for_sample], in_state, cnt)
    print("Sample: %s" % pretty_print(samples, False, id_to_word))
