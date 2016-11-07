#!/usr/bin/python
#
# author: rknaebel
#
# description:
#
import numpy as np

# KERAS: neural network lib
from keras.layers import Input, Dense, Embedding, LSTM
from keras.layers import AveragePooling1D, Reshape
from keras.optimizers import RMSprop
from keras.models import Model

from keras.preprocessing.text import text_to_word_sequence

class ActionDecisionModel(object):
    pass

class VanillaADM(ActionDecisionModel):
    def __init__(self,  seq_len, vocab_size, embedding_size,
                        hidden1_size, hidden2_size,
                        num_actions, num_objects,
                        alpha):
        self.seq_length = seq_len
        self.vocab_size = vocab_size
        self.embd_size = embedding_size
        self.h1 = hidden1_size
        self.h2 = hidden2_size
        self.action_size = num_actions
        self.object_size = num_objects

        self.qsa_model, self.qso_model = self.defineModels()

        self.qsa_model.compile(loss="mse",optimizer=RMSprop(alpha))
        self.qso_model.compile(loss="mse",optimizer=RMSprop(alpha))

    def defineModels(self):
        x = Input(shape=(self.seq_length,), dtype="int32")
        # State Representation
        w_k = Embedding(output_dim=self.embd_size,
                        input_dim=self.vocab_size,
                        input_length=self.seq_length)(x)
        x_k = LSTM(self.h1, return_sequences=True)(w_k)
        y = AveragePooling1D(pool_length=self.seq_length, stride=None)(x_k)
        v_s = Reshape((self.h1,))(y) # remove 2 axis which is 1 caused by averaging
        # Q function approximation
        q = Dense(self.h2, activation="relu")(v_s)
        # action value
        q_sa = Dense(self.action_size)(q)
        # object value
        q_so = Dense(self.object_size)(q)

        m_sa = Model(input=x,output=q_sa)
        m_so = Model(input=x,output=q_so)

        return m_sa, m_so

    def predictQval(self,s):
        qsa = self.qsa_model.predict(np.atleast_2d(s))
        qso = self.qso_model.predict(np.atleast_2d(s))
        return (qsa, qso)

    def predictAction(self,s):
        qsa, qso = self.predictQval(s)
        return (np.argmax(qsa[0]), np.argmax(qso[0]))

    def predictQmax(self,s):
        qsa, qso = self.predictQval(s)
        return (qsa.max(axis=1), qso.max(axis=1))

    def trainOnBatch(self,s_batch,target_qsa,target_qso):
        loss1 = self.qsa_model.train_on_batch(s_batch,target_qsa)
        loss2 = self.qso_model.train_on_batch(s_batch,target_qso)
        return loss1, loss2
