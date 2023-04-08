from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.layers import Reshape, Dense, Convolution1D, Dropout, Input, Activation, Flatten, MaxPool1D, add, \
    AveragePooling1D, Bidirectional, GRU, LSTM, Multiply, MaxPooling1D, TimeDistributed, AvgPool1D
from keras.layers.merge import Concatenate, concatenate
from keras import activations
import os
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
from keras.layers.wrappers import Bidirectional
from six.moves import cPickle as pickle
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop, Adamax, Nadam
from keras.models import Model, load_model
#from keras.utils import plot_model
from keras.regularizers import l2, l1
from sklearn.metrics import confusion_matrix
from keras import backend as K
from keras.backend import sigmoid
from DProcess import convertRawToXY
from BertDealEmbedding import circRNABert
#from tcn import TCN
from keras import metrics
from keras.constraints import max_norm
import logging
import os
import sys
import numpy as np
import time
import argparse
import math
import logging
import os
import sys
import numpy as np
import time
import math
from capsule import Capsule,PrimaryCap
import tensorflow as tf
import collections
from itertools import cycle
from numpy import interp
from Deal_Kmer import *
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score,precision_recall_curve
from keras.engine.topology import Layer
from keras.utils.generic_utils import get_custom_objects
from keras.layers.core import Lambda
from keras.layers import dot
import sys
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.doc2vec import Doc2Vec
# from gensim.models.doc2vec import Doc2Vec,LabeledSentence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, GlobalAveragePooling1D
from keras.utils import to_categorical
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import keras
import os
import pandas as pd
import numpy as np
import pickle
import pdb
import logging, multiprocessing
from collections import namedtuple
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import train_test_split
from keras_self_attention import SeqSelfAttention, ScaledDotProductAttention
from numpy import interp
#from keras.utils import plot_model
# import matplotlib.pyplot as plt
import numpy as np
from getSequenceAndStructure import *
from AnalyseFASTA import analyseFixedPredict

import matplotlib.pyplot as plt
# gpu_id = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
# os.system('echo $CUDA_VISIBLE_DEVICES')
# tf_config = tf.ConfigProto()
# tf_config.gpu_options.per_process_gpu_memory_fraction = 0.1
# tf_config.gpu_options.allow_growth = True
# tf.Session(config=tf_config)
np.random.seed(4)
def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.685 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

def seq2ngram(seqs, k, s, wv):
    list22 = []
    print('need to n-gram %d lines' % len(seqs))

    for num, line in enumerate(seqs):
        if num < 3000000:
            line = line.strip()
            l = len(line)
            list2 = []
            for i in range(0, l, s):
                if i + k >= l + 1:
                    break
                list2.append(line[i:i + k])
            list22.append(convert_data_to_index(list2, wv))
    return list22


def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        if word in wv:
            index_data.append(wv.vocab[word].index)
    return index_data


def split_overlap_seq(seq):
    window_size = 101
    overlap_size = 20
    bag_seqs = []
    seq_len = len(seq)
    if seq_len >= window_size:
        num_ins = (seq_len - 101) / (window_size - overlap_size) + 1
        remain_ins = (seq_len - 101) % (window_size - overlap_size)
    else:
        num_ins = 0
    bag = []
    end = 0
    for ind in range(int(num_ins)):
        start = end - overlap_size
        if start < 0:
            start = 0
        end = start + window_size
        subseq = seq[start:end]
        bag_seqs.append(subseq)
    if num_ins == 0:
        bag_seqs.append(seq)
    else:
        if remain_ins > 10:
            new_size = end - overlap_size
            seq1 = seq[-new_size:]
            bag_seqs.append(seq1)
    return bag_seqs


def build_class_file(np, ng, class_file):
    with open(class_file, 'w') as outfile:
        outfile.write('label' + '\n')
        for i in range(np):
            outfile.write('1' + '\n')
        for i in range(ng):
            outfile.write('0' + '\n')


def circRNA2Vec(k, s, vector_dim, model, MAX_LEN, pos_sequences, neg_sequences):
    model1 = gensim.models.Doc2Vec.load(model)
    pos_list = seq2ngram(pos_sequences, k, s, model1.wv)
    neg_list = seq2ngram(neg_sequences, k, s, model1.wv)
    seqs = pos_list + neg_list

    X = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')
    y = np.array([1] * len(pos_list) + [0] * len(neg_list))
    print("y:",y)
    y = to_categorical(y)
    print("y1:",y)

    # build_class_file(len(pos_list), len(neg_list), root_path + 'class')

    indexes = np.random.choice(len(y), len(y), replace=False)
    dataX = np.array(X)[indexes]
    dataY = np.array(y)[indexes]

    embedding_matrix = np.zeros((len(model1.wv.vocab), vector_dim))
    for i in range(len(model1.wv.vocab)):
        embedding_vector = model1.wv[model1.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return X, y, embedding_matrix


def read_fasta_file(fasta_file):
    seq_dict = {}
    bag_sen = list()
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        line = line.rstrip()
        if line[0] == '>':
            name = line[1:]
            seq_dict[name] = ''
        else:
            seq_dict[name] = seq_dict[name] + line.upper()
    fp.close()

    for seq in seq_dict.values():
        seq = seq.replace('T', 'U')
        bag_sen.append(seq)

    return np.asarray(bag_sen)


def Generate_Embedding(seq_posfile, seq_negfile, model):
    seqpos = read_fasta_file(seq_posfile)
    seqneg = read_fasta_file(seq_negfile)

    X, y, embedding_matrix = circRNA2Vec(10, 1, 30, model, 101, seqpos, seqneg)
    return X, y, embedding_matrix


def swish(x, beta=1):
    return (x * sigmoid(beta * x))


get_custom_objects().update({'swish': swish})


def mk_dir(dir):
    try:
        os.makedirs(dir)
    except OSError:
        print('directory already exits:', dir)


def defineExperimentPaths(basic_path, methodName, experimentID):
    experiment_name = methodName + '/' + experimentID
    MODEL_PATH = basic_path + experiment_name + '/model/'
    LOG_PATH = basic_path + experiment_name + '/logs/'
    CHECKPOINT_PATH = basic_path + experiment_name + '/checkpoints/'
    RESULT_PATH = basic_path + experiment_name + '/results/'
    #mk_dir(MODEL_PATH)
    mk_dir(CHECKPOINT_PATH)
    #mk_dir(RESULT_PATH)
    #mk_dir(LOG_PATH)
    return [MODEL_PATH, CHECKPOINT_PATH, LOG_PATH, RESULT_PATH]


def bn_activation_dropout(input):
    input_bn = BatchNormalization(axis=-1)(input)
    input_at = Activation('swish')(input_bn)
    nput_dp = Dropout(0.4)(input_at)
    return nput_dp


def ConvolutionBlock(input, f, k):
    A1 = Convolution1D(filters=f, kernel_size=k, padding='same')(input)
    A1 = bn_activation_dropout(A1)
    return A1


def InceptionA(input):
    A = ConvolutionBlock(input, 64, 1)
    B = ConvolutionBlock(input, 64, 1)
    B = ConvolutionBlock(B, 64, 5)
    C = ConvolutionBlock(input, 64, 1)
    C = ConvolutionBlock(C, 64, 7)
    C = ConvolutionBlock(C, 64, 7)
    return Concatenate(axis=-1)([A, B, C])


def MultiScale(input):
    A = ConvolutionBlock(input, 64, 1)
    C = ConvolutionBlock(input, 64, 1)
    C = ConvolutionBlock(C, 64, 3)
    D = ConvolutionBlock(input, 64, 1)
    D = ConvolutionBlock(D, 64, 5)
    D = ConvolutionBlock(D, 64, 5)
    merge = Concatenate(axis=-1)([A, C, D])
    shortcut_y = Convolution1D(filters=192, kernel_size=1, padding='same')(input)
    shortcut_y = BatchNormalization()(shortcut_y)
    result = add([shortcut_y, merge])
    result = Activation('swish')(result)
    #result = Dense(64, activation='swish')(result)
    return result

def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

def myloss(y_true, y_pred):
    loss1 = K.categorical_crossentropy(y_true, y_pred)
    loss2 = margin_loss(y_true, y_pred)
    return 0.5 * loss1 + 0.5* loss2


def createModel(embedding_matrix):
    input_row_One_Hot = 101
    input_col_One_Hot = 5

    input_row_ANF_NCP = 101
    input_col_ANF_NCP = 9

    input_row_CKSNAP_NCP = 150
    input_col_CKSNAP_NCP = 17

    input_row_PSTNPss_NCP = 101
    input_col_PSTNPss_NCP = 13



    sequence_input = Input(shape=(101, 41), name='sequence_input')
    sequence = Convolution1D(filters=128, kernel_size=7, padding='same')(sequence_input)
    sequence = BatchNormalization(axis=-1)(sequence)
    sequence = Activation('swish')(sequence)
    profile_input = Input(shape=(101,), name='profile_input')
    embedding = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                          weights=[embedding_matrix], trainable=False)(profile_input)
    profile = Convolution1D(filters=128, kernel_size=7, padding='same')(embedding)
    profile = BatchNormalization(axis=-1)(profile)
    profile = Activation('swish')(profile)
   

    profile_input1 = Input(shape=(101, 768), name='profile_input1')
    profile1 = Convolution1D(filters=128, kernel_size=1, padding='same')(profile_input1)
    profile1 = BatchNormalization(axis=-1)(profile1)
    profile1 = Activation('swish')(profile1)

    main_input = Input(shape=(input_row_One_Hot, input_col_One_Hot), name='main_input')
    mai = Convolution1D(filters=128, kernel_size=7, padding='same')(main_input)
    mai = BatchNormalization(axis=-1)(mai)
    mai1 = Activation('swish')(mai)

    input_P = Input(shape=(input_row_PSTNPss_NCP, input_col_PSTNPss_NCP),name='input_P')
    P = Convolution1D(filters=128, kernel_size=7, padding='same')(input_P)
    P = BatchNormalization(axis=-1)(P)
    main1 = Activation('swish')(P)


    seq = GlobalAveragePooling1D()(sequence)
    seq = Dense(64, activation='swish')(seq)

    seq = Dense(128, activation='sigmoid')(seq)

    seq = Dropout(0.3)(seq)
    seq = Reshape((1, 128))(seq)


    pro = GlobalAveragePooling1D()(profile)
    pro = Dense(64, activation='swish')(pro)
    pro = Dense(128, activation='sigmoid')(pro)
   
    pro = Dropout(0.3)(pro)
    pro = Reshape((1, 128))(pro)


    pro1 = GlobalAveragePooling1D()(profile1)
    pro1 = Dense(64, activation='swish')(pro1)
    pro1 = Dense(128, activation='sigmoid')(pro1)
   
    pro1 = Dropout(0.3)(pro1)
    pro1 = Reshape((1, 128))(pro1)


    main=GlobalAveragePooling1D()(main1)
    main = Dense(64, activation='swish')(main)
    main = Dense(128, activation='sigmoid')(main)
   
    main = Dropout(0.3)(main)
    main = Reshape((1, 128))(main)
   

    mai = GlobalAveragePooling1D()(mai1)
    mai = Dense(64, activation='swish')(mai)
    mai = Dense(128, activation='sigmoid')(mai)
   
    mai = Dropout(0.3)(mai)
    mai = Reshape((1, 128))(mai)


    sequence = Multiply()([sequence, seq])
    profile = Multiply()([profile, pro])
    profile1 = Multiply()([profile1, pro1])
    main = Multiply()([main1, main])
    mai = Multiply()([mai1, mai])


    overallResult= Concatenate(axis=-1)([sequence,profile,profile1,main,mai])
    overallResult =  ConvolutionBlock(overallResult, 64, 5)
    overallResult=MaxPooling1D(pool_size=5)(overallResult)

    overallResult = Dropout(0.3)(overallResult)
    overallResult = Bidirectional(LSTM(120, return_sequences=True))(overallResult)
    overallResult = Flatten()(overallResult)
    overallResult = Dense(101, activation='swish')(overallResult)
    overallResult = Dropout(0.3)(overallResult)
    ss_output = Dense(2, activation='softmax', name='ss_output', input_shape=[None, 101])(overallResult)
    return Model(inputs=[sequence_input, profile_input,input_P,profile_input1,main_input], outputs=[ss_output])


def parse_arguments(parser):
    parser.add_argument('--proteinID', type=str, default='FOX2')
    parser.add_argument('--modelType', type=str, default='/Share/home/10014/xuezhigang/iCircRBP-DHN-master11/circRNA2Vec/circRNA2Vec_model',
                        help='generate the embedding_matrix')
    parser.add_argument('--storage', type=str, default='/Share/home/10014/xuezhigang/w2/New2/')
    parser.add_argument('--userID', type=str, default='model2')
    args = parser.parse_args()
    return args







def main(parser):
    protein = parser.proteinID
    model = parser.modelType
    file_storage = parser.storage
    userid = parser.userID
    seqpos_path = '/Share/home/10014/xuezhigang/iCircRBP-DHN-master11/Datasets/circRNA-RBP/' + protein + '/positive'
    seqneg_path = '/Share/home/10014/xuezhigang/iCircRBP-DHN-master11/Datasets/circRNA-RBP/' + protein + '/negative'
    print('proteinID:', protein)
    #3mer
    Kmer = dealwithdata(protein)
    print("Kmer.shape:",Kmer.shape)
    dataX2 = dealwithSequenceAndStructure(protein)

    Embedding, dataY, embedding_matrix = Generate_Embedding(seqpos_path, seqneg_path, model)
    Embedding1 = circRNABert(protein, 3)
    print("Embedding:",Embedding.shape)
    print("dataY:",dataY.shape)
    print("embedding_matrix:",Embedding.shape)

    pos_data, pos_ids, pos_poses = analyseFixedPredict(seqpos_path, window=20, label=1)
    neg_data, neg_ids, neg_poses = analyseFixedPredict(seqneg_path, window=20, label=0)

    train_All2 = pd.concat([pos_data, neg_data])
    train_data = train_All2
    train_All = train_data
    trainX_One_Hot, trainY_One_Hot = convertRawToXY(train_All.values, train_data.values, codingMode='ENAC')

    #####################################ANF_NCP_EIIP_Onehot#####################################
    trainX_ANF_NCP, trainY_ANF_NCP = convertRawToXY(train_All.values, train_data.values,
                                                    codingMode='ANF_NCP_EIIP_Onehot')


    #####################################PSTNPss_NCP_EIIP_Onehot#####################################
    trainX_PSTNPss_NCP, trainY_PSTNPss_NCP = convertRawToXY(train_All.values, train_data.values,
                                                            codingMode='PSTNPss_NCP_EIIP_Onehot')

    indexes = np.random.choice(Kmer.shape[0], Kmer.shape[0], replace=False)
    print("indexes:",np.array(indexes).shape)

    training_idx, test_idx = indexes[:round(((Kmer.shape[0]) / 10) * 8)], indexes[round(((Kmer.shape[0]) / 10) * 8):]
    print("training_idx:",np.array(training_idx).shape)
    #print("test_idx:",test_idx)

    train_sequence, test_sequence = Kmer[training_idx, :, :], Kmer[test_idx, :, :]
    train_seqsec,test_seqsec=dataX2[training_idx, :, :],dataX2[test_idx, :, :]
    train_profile, test_profile = Embedding[training_idx, :], Embedding[test_idx, :]
    train_profile1, test_profile1 = Embedding1[training_idx, :], Embedding1[test_idx, :]

    train_onehot, testonehot = trainX_One_Hot[training_idx, :, :], trainX_One_Hot[test_idx, :, :]
    train_PSTNPss_NCP, test_PSTNPss_NCP = trainX_PSTNPss_NCP[training_idx, :, :], trainX_PSTNPss_NCP[test_idx, :, :]
    train_ANF_NCP, test_ANF_NCP = trainX_ANF_NCP[training_idx, :, :], trainX_ANF_NCP[test_idx, :, :]

    train_label, test_label = dataY[training_idx, :], dataY[test_idx, :]
    print("train_label:",np.array(train_label).shape)
    print("train_label:", train_label)

    batchSize = 100
    maxEpochs = 15
    basic_path = file_storage + userid + '/'
    methodName = protein


    # logging.basicConfig(level=logging.DEBUG)
    # sys.stdout = sys.stderr
    # logging.debug("Loading data...")
    print("Loading data...")

    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    test_y = test_label[:, 1]

    kf = KFold(5, shuffle=True)
    aucs = []
    Acc = []
    precision1 = []
    recall1 = []
    fscore1 = []
    Auprc = []
    i = 0

    for train_index, eval_index in kf.split(train_label):
        train_X1 = train_sequence[train_index]
        train_X2 = train_profile[train_index]
        train_X5 = train_profile1[train_index]
        train_X3 = train_seqsec[train_index]
        train_X7 = train_onehot[train_index]
        train_X4 = train_ANF_NCP[train_index]
        train_X6 = train_PSTNPss_NCP[train_index]

        train_y = train_label[train_index]


        eval_X1 = train_sequence[eval_index]
        eval_X2 = train_profile[eval_index]
        eval_X5 = train_profile1[eval_index]
        eval_X3 = train_seqsec[eval_index]

        eval_X7 = train_onehot[eval_index]
        eval_X4 = train_ANF_NCP[eval_index]
        eval_X6 = train_PSTNPss_NCP[eval_index]
        eval_y = train_label[eval_index]
        eval_One_Hot = trainY_One_Hot[eval_index]



        eval_y = train_label[eval_index]
        [MODEL_PATH, CHECKPOINT_PATH, LOG_PATH, RESULT_PATH] = defineExperimentPaths(basic_path, methodName,
                                                                                     str(i))
        model = createModel(embedding_matrix)
        checkpoint_weight = CHECKPOINT_PATH + "weights.best.hdf5"
        if (os.path.exists(checkpoint_weight)):
            #print("load previous best weights")
            model.load_weights(checkpoint_weight)
        else:
            print('training_network size is ', len(train_X1))
            print('validation_network size is ', len(eval_X1))
            print("begin training")
            # logging.debug("Loading network/training configuration...")
            # logging.debug("Model summary ... ")
            # model.count_params()s
            #model.summary()
            #plot_model(model,to_file='MMTM2cap.png',show_shapes=True)
            model.compile(optimizer='adam',
                          loss={'ss_output':'categorical_crossentropy'}, metrics=['accuracy'])
            logging.debug("Running training...")
            train_loss = []
            val_loss = []
            train_acc = []
            val_acc = []

            def step_decay(epoch):
                initial_lrate = 0.0005
                drop = 0.8
                epochs_drop = 5.0
                lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
                print(lrate)
                return lrate

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='auto'),
                ModelCheckpoint(checkpoint_weight,
                                monitor='val_loss',
                                verbose=2,
                                save_best_only=True,
                                mode='auto',
                                period=1),
                LearningRateScheduler(step_decay),
            ]
            startTime = time.time()
            his = model.fit(
                {'sequence_input': train_X1, 'profile_input': train_X2, 'profile_input1': train_X5,'seqsec_input':train_X3,'main_input':train_X7,'input_P':train_X6,'input_A':train_X4},
                {'ss_output': train_y},
                epochs=maxEpochs,
                batch_size=batchSize,
                callbacks=callbacks,
                verbose=2,
                validation_data=(
                    {'sequence_input': eval_X1, 'profile_input': eval_X2, 'profile_input1': eval_X5,'seqsec_input':eval_X3,'main_input':eval_X7,'input_P':eval_X6,'input_A':eval_X4},
                    {'ss_output': eval_y}),
                shuffle=True)
        endTime = time.time()
        print("----------begin prediction----------")
        ss_y_hat_test = model.predict(
            {'sequence_input': test_sequence, 'profile_input': test_profile,'profile_input1':test_profile1,'seqsec_input':test_seqsec, 'main_input':testonehot,'input_A':test_ANF_NCP,'input_P':test_PSTNPss_NCP})
        #print('ss_y_hat_test',ss_y_hat_test)

        ytrue = test_y
        ypred = ss_y_hat_test[:, 1]

        y_pred = np.argmax(ss_y_hat_test, axis=-1)
        auc1 = roc_auc_score(ytrue, ypred)
        print("AUC:", auc1)
        fpr, tpr, thresholds = roc_curve(ytrue, ypred)
        tprs.append(interp(mean_fpr, fpr, tpr))
        aucs.append(auc1)
        acc = accuracy_score(ytrue, y_pred)
        # print("ACC:",acc)
        Acc.append(acc)

        precision = precision_score(ytrue, y_pred)
        recall = recall_score(ytrue, y_pred)
        fscore = f1_score(ytrue, y_pred)
        # print("Precision:", precision)
        # print("Recall:", recall)
        # print("f1:",fscore)
        precision1.append(precision)
        recall1.append(recall)
        fscore1.append(fscore)
        precision, recall, _thresholds = precision_recall_curve(ytrue, y_pred)
        auprc = auc(recall, precision)
        print("auprc", auprc)
        Auprc.append(auprc)
        i = i + 1

    print('proteinID:', protein)
    print("mean AUC: %.4f " % np.mean(aucs))
    print("mean Auprc: %.4f " % np.mean(Auprc))
    print("mean ACC: %.4f " % np.mean(Acc))
    print("mean Precision: %.4f " % np.mean(precision1))
    print("mean Recall: %.4f " % np.mean(recall1))
    print("mean fscore: %.4f " % np.mean(fscore1))


    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    mean_acc = np.mean(Acc)
    mean_precision = np.mean(precision1)
    mean_recall = np.mean(recall1)
    mean_fscore = np.mean(fscore1)

    # np.save(basic_path + methodName + '/' + 'mean_fpr.npy',mean_fpr)
    # np.save(basic_path + methodName + '/' + 'mean_tpr.npy',mean_tpr)
    np.save(basic_path + methodName + '/' + 'mean_auc.txt',mean_auc)
    np.save(basic_path + methodName + '/' + 'mean_acc.txt',mean_acc)
    # np.save(basic_path + methodName + '/' + 'mean_precision.npy',mean_precision)
    # np.save(basic_path + methodName + '/' + 'mean_recall.npy',mean_recall)
    # np.save(basic_path + methodName + '/' + 'mean_fscore.npy',mean_fscore)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    main(args)
