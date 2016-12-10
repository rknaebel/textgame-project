#!/usr/bin/python
#
# author: rknaebel
#
# description:
#
import numpy as np

# KERAS: neural network lib
import keras.backend as K
from keras.layers import Input, Dense, Embedding, LSTM
from keras.layers import AveragePooling1D, Reshape, GlobalAveragePooling1D, merge
from keras.optimizers import RMSprop, Adam, Nadam
from keras.models import Model

from ad_model import ActionDecisionModel

class NeuralQLearner(ActionDecisionModel):
    def __init__(self,  seq_len, vocab_size, embedding_size,
                        hidden1_size, hidden2_size,
                        num_actions, num_objects,
                        alpha,gamma,batch_size):
        self.seq_length = seq_len
        self.vocab_size = vocab_size
        self.embd_size = embedding_size
        self.h1 = hidden1_size
        self.h2 = hidden2_size
        self.action_size = num_actions
        self.object_size = num_objects

        self.alpha = alpha
        self.gamma = gamma
        self.batch_size = batch_size

        self.model = self.defineModels()
        self.model.compile(loss="mse",optimizer=Nadam())

    def defineModels(self):
        x = Input(shape=(self.seq_length,), dtype="uint8")
        # State Representation
        w_k = Embedding(output_dim=self.embd_size,
                        input_dim=self.vocab_size,
                        input_length=self.seq_length)(x)
        x_k = LSTM(self.h1, return_sequences=True)(w_k)
        v_s = GlobalAveragePooling1D()(x_k)
        # Q function approximation
        q_hidden = Dense(self.h2, activation="relu")(v_s)
        # action value
        qsa = Dense(self.action_size)(q_hidden)
        # object value
        qso = Dense(self.object_size)(q_hidden)

        q = merge(  [qsa,qso],
                    mode=lambda x: (K.expand_dims(x[0],2)+K.expand_dims(x[1],1))/2,
                    output_shape=lambda x: (x[0][0],x[0][1],x[1][1]))

        q_model = Model(input=x,output=q)

        return q_model

    def predictQval(self,s):
        return self.model.predict(np.atleast_2d(s))

    def predictAction(self,s):
        q = self.predictQval(s)[0]
        return np.unravel_index(q.argmax(),q.shape)

    def randomAction(self):
        act = np.random.randint(0, self.action_size)
        obj = np.random.randint(0, self.object_size)
        return (act,obj)

    def predictQmax(self,s):
        q = self.predictQval(s)
        return q.max(axis=(1,2))

    def trainOnBatch(self,s_batch,a_batch,r_batch,t_batch,s2_batch):
        # split action tuple
        act_batch, obj_batch = a_batch[:,0], a_batch[:,1]

        # Calculate targets
        target = self.predictQval(s_batch)
        qmax = self.predictQmax(s2_batch)
        # discount state values using the calculated targets
        y_i = []
        for k in xrange(self.batch_size):
            a,o = act_batch[k],obj_batch[k]
            if t_batch[k]:
                # just the true reward if game is over
                target[k,a,o] = r_batch[k]
            else:
                # reward + gamma * max a'{ Q(s', a') }
                target[k,a,o] = r_batch[k] + self.gamma * qmax[k]

        loss = self.model.train_on_batch(s_batch,target)

        return loss

    def save(self,name,overwrite):
        self.model.save("q_"+name, overwrite=overwrite)
