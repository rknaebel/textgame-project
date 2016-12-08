import numpy as np

# KERAS: neural network lib
from keras.layers import Input, Dense, Embedding, LSTM, SimpleRNN
from keras.layers import AveragePooling1D, Reshape, GlobalAveragePooling1D, Merge
from keras.layers import TimeDistributed
from keras.optimizers import RMSprop
from keras.models import Model

from ad_model import ActionDecisionModel

class RNNQLearner(ActionDecisionModel):
    def __init__(self,  seq_len, vocab_size, embedding_size, hist_size,
                        hidden1_size, hidden2_size,
                        num_actions, num_objects,
                        alpha,gamma,batch_size):
        self.seq_length = seq_len
        self.vocab_size = vocab_size
        self.embd_size = embedding_size
        self.hist_size = hist_size
        self.h1 = hidden1_size
        self.h2 = hidden2_size
        self.action_size = num_actions
        self.object_size = num_objects

        self.alpha = alpha
        self.gamma = gamma
        self.batch_size = batch_size

        self.model, self.embedding = self.defineModels_old()

        self.model.compile(loss="mse",optimizer=RMSprop(alpha))

    def defineModels(self):
        x = Input(shape=(self.hist_size,self.seq_length,), dtype="int32") # (STATES x SEQUENCE)
        # State Representation
        w_k = TimeDistributed(Embedding(output_dim=self.embd_size, mask_zero=True,
                        input_dim=self.vocab_size,
                        input_length=self.seq_length), name="embedding")(x) # (STATES x SEQUENCE x EMBEDDING)
        w_k = TimeDistributed(LSTM(self.h1, activation="relu", return_sequences=True), name="lstm1")(w_k) # (STATES x SEQUENCE x H1)
        v_s = TimeDistributed(LSTM(self.h1, activation="relu"), name="lstm2")(w_k) # (STATES x H1)
        embd_model = Model(input=x,output=v_s)
        # history based Q function approximation
        q = SimpleRNN(self.h2, activation="relu", name="history_rnn")(v_s) # (H2)
        # action value
        q_sa = Dense(self.action_size, name="action_dense")(q) # (ACTIONS)
        # object value
        q_so = Dense(self.object_size, name="object_dense")(q) # (OBJECTS)
        q_model = Model(input=x,output=[q_sa,q_so])

        return q_model, embd_model

    def defineModels_old(self):
        x = Input(shape=(self.hist_size,self.seq_length,), dtype="int32") # (STATES x SEQUENCE)
        # State Representation
        w_k = TimeDistributed(Embedding(output_dim=self.embd_size, mask_zero=True,
                        input_dim=self.vocab_size,
                        input_length=self.seq_length), name="embedding")(x) # (STATES x SEQUENCE x EMBEDDING)
        w_k = TimeDistributed(LSTM(self.h1, activation="relu", return_sequences=True), name="lstm1")(w_k) # (STATES x SEQUENCE x H1)
        v_s = TimeDistributed(GlobalAveragePooling1D(), name="avg")(w_k) # (STATES x H1)
        embd_model = Model(input=x,output=v_s)
        # history based Q function approximation
        q = SimpleRNN(self.h2, activation="relu", name="history_rnn")(v_s) # (H2)
        # action value
        q_sa = Dense(self.action_size, name="action_dense")(q) # (ACTIONS)
        # object value
        q_so = Dense(self.object_size, name="object_dense")(q) # (OBJECTS)
        q_model = Model(input=x,output=[q_sa,q_so])

        return q_model, embd_model

    def embedStates(self, xs):
        return self.embedding.predict(xs)

    def predictQval(self,s):
        qsa, qso = self.model.predict(np.atleast_3d(s))
        return (qsa, qso)

    def predictAction(self,s):
        qsa, qso = self.predictQval([s])
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
        self.model.save("q_"+name, overwrite=overwrite)
