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


class ActionDecisionModel(object):
    def predictAction(self, s): pass
    def trainOnBatch(self,s_batch,a_batch,r_batch,t_batch,s2_batch): pass
    def save(self,name,overwrite): pass

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

    def randomAction(self):
        act = np.random.randint(0, self.action_size)
        obj = np.random.randint(0, self.object_size)
        return (act,obj)

    def predictQmax(self,s):
        qsa, qso = self.predictQval(s)
        return (qsa.max(axis=1), qso.max(axis=1))

    def trainOnBatch(self,s_batch,a_batch,r_batch,t_batch,s2_batch):
        # split action tuple
        act_batch, obj_batch = a_batch[:,0], a_batch[:,1]

        # Calculate targets
        target_qsa, target_qso = self.predictQval(s_batch)
        qsa, qso = self.predictQmax(s2_batch)

        # discount state values using the calculated targets
        y_i = []
        for k in xrange(self.batch_size):
            if t_batch[k]:
                # just the true reward if game is over
                target_qsa[k,act_batch[k]] = r_batch[k]
                target_qso[k,obj_batch[k]] = r_batch[k]
            else:
                # reward + gamma * max a'{ Q(s', a') }
                target_qsa[k,act_batch[k]] = r_batch[k] + self.gamma * qsa[k]
                target_qso[k,obj_batch[k]] = r_batch[k] + self.gamma * qso[k]

        loss1 = self.qsa_model.train_on_batch(s_batch,target_qsa)
        loss2 = self.qso_model.train_on_batch(s_batch,target_qso)
        return loss1, loss2

    def save(self,name,overwrite):
        self.qsa_model.save("qsa_"+name, overwrite=overwrite)
        self.qso_model.save("qso_"+name, overwrite=overwrite)

class RNNQLearner(ActionDecisionModel):
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

        self.model = self.defineModel()

        self.model.compile(loss="mse",optimizer=RMSprop(alpha))

    def defineModel(self):
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

        model = Model(input=x,output=[q_sa,q_so])

        return model

    def predictQval(self,s):
        qsa, qso = self.model.predict(np.atleast_2d(s))
        return (qsa, qso)

    def predictAction(self,s):
        qsa, qso = self.predictQval(s)
        return (np.argmax(qsa[0]), np.argmax(qso[0]))

    def randomAction(self):
        act = np.random.randint(0, self.action_size)
        obj = np.random.randint(0, self.object_size)
        return (act,obj)

    def predictQmax(self,s):
        qsa, qso = self.predictQval(s)
        return (qsa.max(axis=1), qso.max(axis=1))

    def trainOnBatch(self,s_batch,a_batch,r_batch,t_batch,s2_batch):
        # split action tuple
        act_batch, obj_batch = a_batch[:,0], a_batch[:,1]

        # Calculate targets
        target_qsa, target_qso = self.predictQval(s_batch)
        qsa, qso = self.predictQmax(s2_batch)

        # discount state values using the calculated targets
        y_i = []
        for k in xrange(self.batch_size):
            if t_batch[k]:
                # just the true reward if game is over
                target_qsa[k,act_batch[k]] = r_batch[k]
                target_qso[k,obj_batch[k]] = r_batch[k]
            else:
                # reward + gamma * max a'{ Q(s', a') }
                target_qsa[k,act_batch[k]] = r_batch[k] + self.gamma * qsa[k]
                target_qso[k,obj_batch[k]] = r_batch[k] + self.gamma * qso[k]

        loss = self.model.train_on_batch(s_batch,[target_qsa,target_qso])
        return loss

    def save(self,name,overwrite):
        self.qsa_model.save("qsa_"+name, overwrite=overwrite)
        self.qso_model.save("qso_"+name, overwrite=overwrite)

class ReinforceLearner(ActionDecisionModel):
    """
    Instead of measuring the absolute goodness of an action we want to know how much better than "average" it is to take an action given a state. E.g. some states are naturally bad and always give negative reward. This is called the advantage and is defined as Q(s, a) - V(s). We use that for our policy update, e.g. g_t - V(s) for REINFORCE
    """
    def predictAction(self, s): pass
    def trainOnBatch(self,s_batch,a_batch,r_batch,t_batch,s2_batch): pass
    def save(self,name,overwrite): pass

class ActorCriticLearner(object):
    """
    Two function approximators: One of the policy, one for the critic.
    This is basically TD, but for Policy Gradients
    """
    def predictAction(self, s): pass
    def trainOnBatch(self,s_batch,a_batch,r_batch,t_batch,s2_batch): pass
    def save(self,name,overwrite): pass

## A3C
# Instead of using an experience replay buffer as in DQN use multiple agents on different threads to explore the state spaces and make decorrelated updates to the actor and the critic.
