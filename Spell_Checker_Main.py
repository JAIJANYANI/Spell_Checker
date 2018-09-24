# ▒█▀▀▀█ ▒█▀▀█ ▒█▀▀▀ ▒█░░░ ▒█░░░ 　 ▒█▀▀█ ▒█░▒█ ▒█▀▀▀ ▒█▀▀█ ▒█░▄▀ ▒█▀▀▀ ▒█▀▀█ 
# ░▀▀▀▄▄ ▒█▄▄█ ▒█▀▀▀ ▒█░░░ ▒█░░░ 　 ▒█░░░ ▒█▀▀█ ▒█▀▀▀ ▒█░░░ ▒█▀▄░ ▒█▀▀▀ ▒█▄▄▀ 
# ▒█▄▄▄█ ▒█░░░ ▒█▄▄▄ ▒█▄▄█ ▒█▄▄█ 　 ▒█▄▄█ ▒█░▒█ ▒█▄▄▄ ▒█▄▄█ ▒█░▒█ ▒█▄▄▄ ▒█░▒█ 


"""
●   Training is disabled by default, Pretrained Checkpoint is provided.
●   Used a General Corpus which contains 3024075 n-grams, it can also be trained on Google Billion Corpus,require more resources.

Prerequisites

    Tensorflow==1.1
    Numpy==1.11.3
    Pandas
    Sklearn
    Dash


Tested on Ubuntu 16.04 LTS amd64 xenial image built on 2017-09-19 40-core CPU 128 GB Ram


        To Run this code: 
            # python Spell_Checker_Main.py

NOTE: If you are receiving this error : "InvalidArgumentError: Assign requires shapes of both tensors to match."

Try switching checkpoints between V1 and V2 on line 258

"""


import os
import re
import difflib
import time
from collections import namedtuple
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
from collections import Counter
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


Sub_Google_Corpora = pd.read_csv('Sub_Google_Corpora.csv')
Sub_Google_Corpora = list(Sub_Google_Corpora['0'])



def clean_text(text):
    text = re.sub(r'\n', ' ', text) 
    text = re.sub(r'[{}@_*>()\\#%+=\[\]]','', text)
    text = re.sub('a0','', text)
    text = re.sub('\'92t','\'t', text)
    text = re.sub('\'92s','\'s', text)
    text = re.sub('\'92m','\'m', text)
    text = re.sub('\'92ll','\'ll', text)
    text = re.sub('\'91','', text)
    text = re.sub('\'92','', text)
    text = re.sub('\'93','', text)
    text = re.sub('\'94','', text)
    text = re.sub('\.','. ', text)
    text = re.sub('\!','! ', text)
    text = re.sub('\?','? ', text)
    text = re.sub(' +',' ', text)
    return text


def words(text): 
    return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('words.txt').read()))

def P(word, N=sum(WORDS.values())): 
    return WORDS[word] / N


def Most_Probable(word): 
    return max(Possibilities(word), key=P)


def Possibilities(word): 
    return (known([word]) or known(Permutation_1(word)) or known(Permutation_2(word)) or [word])


def known(words): 
    return set(w for w in words if w in WORDS)


def Permutation_1(word):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def Permutation_2(word): 
    return (e2 for e1 in Permutation_1(word) for e2 in Permutation_1(e1))


Cleaned_Corpora = []
for book in Sub_Google_Corpora:
    Cleaned_Corpora.append(clean_text(book))

Cleaned_Corpora[0][:500]

TEXT_2_INT = {}
count = 0
for book in Cleaned_Corpora:
    for character in book:
        if character not in TEXT_2_INT:
            TEXT_2_INT[character] = count
            count += 1

codes = ['<PAD>','<EOS>','<GO>']
for code in codes:
    TEXT_2_INT[code] = count
    count += 1


vocab_size = len(TEXT_2_INT)

INT_2_TEX = {}
for character, value in TEXT_2_INT.items():
    INT_2_TEX[value] = character

threshold = 0.9


def Create_Model():
    
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    with tf.name_scope('targets'):
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    inputs_length = tf.placeholder(tf.int32, (None,), name='inputs_length')
    targets_length = tf.placeholder(tf.int32, (None,), name='targets_length')
    max_target_length = tf.reduce_max(targets_length, name='max_target_len')

    return inputs, targets, keep_prob, inputs_length, targets_length, max_target_length

def process_encoding_input(targets, TEXT_2_INT, batch_size):    
    with tf.name_scope("process_encoding"):
        ending = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([batch_size, 1], TEXT_2_INT['<GO>']), ending], 1)

    return dec_input



def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob, direction):    
    if direction == 1:
        with tf.name_scope("RNN_Encoder_Cell_1D"):
            for layer in range(num_layers):
                with tf.variable_scope('encoder_{}'.format(layer)):
                    lstm = tf.contrib.rnn.LSTMCell(rnn_size)
                    drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
                    enc_output, enc_state = tf.nn.dynamic_rnn(drop, rnn_inputs,sequence_length,dtype=tf.float32)
            return enc_output, enc_state
        
        
    if direction == 2:
        with tf.name_scope("RNN_Encoder_Cell_2D"):
            for layer in range(num_layers):
                with tf.variable_scope('encoder_{}'.format(layer)):
                    cell_fw = tf.contrib.rnn.LSTMCell(rnn_size)
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob = keep_prob)
                    cell_bw = tf.contrib.rnn.LSTMCell(rnn_size)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob = keep_prob)
                    enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, rnn_inputs,sequence_length,dtype=tf.float32)
            enc_output = tf.concat(enc_output,2)
            return enc_output, enc_state[0]



def training_decoding_layer(dec_embed_input, targets_length, dec_cell, initial_state, output_layer, vocab_size, max_target_length):

    with tf.name_scope("Training_Decoder"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,sequence_length=targets_length,time_major=False)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,training_helper,initial_state,output_layer) 

        training_logits, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,output_time_major=False,impute_finished=True,maximum_iterations=max_target_length)
        return training_logits


def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer,max_target_length, batch_size):
    with tf.name_scope("Inference_Decoder"):
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,start_tokens,end_token)
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,inference_helper,initial_state,output_layer)
        inference_logits, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,output_time_major=False,impute_finished=True,maximum_iterations=max_target_length)
        return inference_logits



def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, inputs_length, targets_length, max_target_length, rnn_size, TEXT_2_INT, keep_prob, batch_size, num_layers, direction):    
    with tf.name_scope("RNN_Decoder_Cell"):
        for layer in range(num_layers):
            with tf.variable_scope('decoder_{}'.format(layer)):
                lstm = tf.contrib.rnn.LSTMCell(rnn_size)
                dec_cell = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    output_layer = Dense(vocab_size,kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,enc_output,inputs_length,normalize=False,name='BahdanauAttention')
    with tf.name_scope("Attention_Wrapper"):
        dec_cell = tf.contrib.seq2seq.DynamicAttentionWrapper(dec_cell,attn_mech,rnn_size)
    initial_state = tf.contrib.seq2seq.DynamicAttentionWrapperState(enc_state,_zero_state_tensors(rnn_size, batch_size, tf.float32))
    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(dec_embed_input, targets_length, dec_cell, initial_state,output_layer,vocab_size, max_target_length)
    with tf.variable_scope("decode", reuse=True):
        inference_logits = inference_decoding_layer(embeddings,  TEXT_2_INT['<GO>'], TEXT_2_INT['<EOS>'],dec_cell, initial_state, output_layer,max_target_length,batch_size)
    return training_logits, inference_logits


def seq2seq_model(inputs, targets, keep_prob, inputs_length, targets_length, max_target_length, vocab_size, rnn_size, num_layers, TEXT_2_INT, batch_size, embedding_size, direction):
    
    enc_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    enc_embed_input = tf.nn.embedding_lookup(enc_embeddings, inputs)
    enc_output, enc_state = encoding_layer(rnn_size, inputs_length, num_layers, enc_embed_input, keep_prob, direction)
    dec_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    dec_input = process_encoding_input(targets, TEXT_2_INT, batch_size)
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    training_logits, inference_logits  = decoding_layer(dec_embed_input, dec_embeddings,enc_output,enc_state, vocab_size, inputs_length, targets_length, max_target_length,rnn_size, TEXT_2_INT, keep_prob, batch_size,num_layers,direction)
    
    return training_logits, inference_logits


epochs = 100
batch_size = 128
num_layers = 2
rnn_size = 512
embedding_size = 128
learning_rate = 0.0005
direction = 2
threshold = 0.95
keep_probability = 0.75



def build_graph(keep_prob, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction):
    tf.reset_default_graph()
    inputs, targets, keep_prob, inputs_length, targets_length, max_target_length = Create_Model()
    training_logits, inference_logits = seq2seq_model(tf.reverse(inputs, [-1]),targets, keep_prob,   inputs_length,targets_length,max_target_length,len(TEXT_2_INT)+1,rnn_size, num_layers, 
TEXT_2_INT,batch_size,embedding_size,direction)
    training_logits = tf.identity(training_logits.rnn_output, 'logits')

    with tf.name_scope('predictions'):
        predictions = tf.identity(inference_logits.sample_id, name='predictions')
        tf.summary.histogram('predictions', predictions)

    masks = tf.sequence_mask(targets_length, max_target_length, dtype=tf.float32, name='masks')
    
    with tf.name_scope("cost"):
        cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)
        tf.summary.scalar('cost', cost)

    with tf.name_scope("optimze"):
        optimizer = tf.train.AdamOptimizer(learning_rate)

        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

    merged = tf.summary.merge_all()    

    export_nodes = ['inputs', 'targets', 'keep_prob', 'cost', 'inputs_length', 'targets_length','predictions', 'merged', 'train_op','optimizer']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph


def text_to_ints(text):    
    text = clean_text(text)
    return [TEXT_2_INT[word] for word in text]

## TRAINING IS BY DEFAULT DISABLED, PRE-TRAINED CHECKPOINT IS PROVIDED

for i in range(0,5):
    WordList1 = []

    checkpoint = "./Pre-Trained_Checkpoint_V2.ckpt"
    # checkpoint = "./Pre-Trained_Checkpoint_V1.ckpt"

    text = input("\n\nEnter a word or text : \t")
    wordList = re.sub("[^\w]", " ",  text).split()

    
    for j in wordList:
        org = j
        WORDS = Counter(words(open('words.txt').read()))
        Correction_1 = Most_Probable(str(j))
        WORDS = Counter(words(open('corncob_lowercase.txt').read()))
        Correction_2 = Most_Probable(str(j))
        Correction_3 = str('')
        j = text_to_ints(j)
        model = build_graph(keep_probability, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction) 
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint)
            answer_logits = sess.run(model.predictions, {model.inputs: [j]*batch_size, model.inputs_length: [len(j)]*batch_size,model.targets_length: [len(j)+1], model.keep_prob: [1.0]})[0]
        pad = TEXT_2_INT["<PAD>"] 
        for ll in answer_logits:
            Correction_3 = Correction_3 + str(INT_2_TEX[ll])

        Correction_4 = Most_Probable(Correction_3)
        Correction_5 = Most_Probable(Correction_3)
        seq = difflib.SequenceMatcher(None,org,Correction_1)
        word0_word1 = seq.ratio()*100
        seq = difflib.SequenceMatcher(None,org,Correction_2)
        word0_word2 = seq.ratio()*100
        seq = difflib.SequenceMatcher(None,org,Correction_3)
        word0_word3 = seq.ratio()*100
        seq = difflib.SequenceMatcher(None,org,Correction_4)
        word0_word4 = seq.ratio()*100
        seq = difflib.SequenceMatcher(None,org,Correction_5)
        word0_word5 = seq.ratio()*100
        LIST = [word0_word1 , word0_word2 , word0_word3 , word0_word4 , word0_word5]
        m = max(LIST)
        LIST2 = [ind for ind, jnd in enumerate(LIST) if jnd == m]
        if LIST2[0]  == 0:
            WordList1.append(Correction_1)
        elif LIST2[0]  == 1:
            WordList1.append(Correction_2)
        elif LIST2[0]  == 2:
            WordList1.append(Correction_3)
        elif LIST2[0]  == 3:
            WordList1.append(Correction_4)
        elif LIST2[0]  == 4:
            WordList1.append(Correction_5)

    print("\nCorrected Output : \t" , end=' ')
    for i in WordList1:
        print(i , end=' ')
    print()
    print("###########################")
