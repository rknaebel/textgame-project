import numpy as np

# KERAS: neural network lib
import keras.backend as K

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from keras.layers import Input, Dense, Embedding, LSTM, SimpleRNN
from keras.layers import GlobalAveragePooling1D, merge, Flatten
from keras.layers import TimeDistributed
from keras.objectives import mean_squared_error
from keras.optimizers import RMSprop, Nadam
from keras.models import Model
from keras.utils.visualize_util import plot

from ad_model import ActionDecisionModel

class HistoryQLearner(ActionDecisionModel):
    def __init__(self,  seq_len, vocab_size, embd_size, hist_size,
                        hidden1, hidden2,
                        num_actions, num_objects,
                        alpha,gamma,exp_id="model"):
        self.seq_length = seq_len
        self.vocab_size = vocab_size
        self.embd_size = embd_size
        self.hist_size = hist_size
        self.h1 = hidden1
        self.h2 = hidden2
        self.action_size = num_actions
        self.object_size = num_objects

        self.alpha = alpha
        self.gamma = gamma

        self.epoch  = 0

        with tf.device("/cpu:0"):
            self.model_cpu = self.defineModels()
        with tf.device("/gpu:0"):
            self.model_gpu = self.defineModels()
        self.model_cpu.compile(loss="mse",optimizer=Nadam(clipvalue=0.1))

        self.model_gpu.compile(loss="mse",optimizer=Nadam(clipvalue=0.1))
        plot(self.model_cpu, show_shapes=True, to_file=exp_id+'.png')
        #
        #
        #
        sess = KTF.get_session()
        tf.summary.scalar("loss", self.model_gpu.total_loss)
        #for layer in self.model_gpu.layers:
        #    for weight in layer.weights:
        #        tf.summary.histogram(weight.name,weight)
        #    if hasattr(layer, 'output'):
        #        tf.summary.histogram('{}_out'.format(layer.name),layer.output)
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("logs",sess.graph)


    def defineModels(self):
        with tf.name_scope("input"):
            x = Input(shape=(self.hist_size,self.seq_length,), dtype="uint8") # (STATES x SEQUENCE)
        with tf.name_scope("state_embd"):
            w_k = TimeDistributed(Embedding(output_dim=self.embd_size, mask_zero=True,
                            input_dim=self.vocab_size,
                            input_length=self.seq_length), name="embedding")(x) # (STATES x SEQUENCE x EMBEDDING)
            w_k = TimeDistributed(LSTM(self.h1,  return_sequences=True), name="lstm1")(w_k) # (STATES x SEQUENCE x H1)
            v_s = TimeDistributed(LSTM(self.h1, activation="relu"), name="lstm2")(w_k) # (STATES x H1)
        # history based Q function approximation
        with tf.name_scope("q_function"):
            q_hidden = SimpleRNN(self.h2, activation="relu", name="history_rnn")(v_s) # (H2)
        # action value
        with tf.name_scope("action_value"):
            qsa = Dense(self.action_size, name="action_dense")(q_hidden) # (ACTIONS)
        # object value
        with tf.name_scope("object_value"):
            qso = Dense(self.object_size, name="object_dense")(q_hidden) # (OBJECTS)
        with tf.name_scope("merged_value"):
            q = merge(  [qsa,qso],
                        mode=lambda x: (K.expand_dims(x[0],2)+K.expand_dims(x[1],1))/2,
                        output_shape=lambda x: (x[0][0],x[0][1],x[1][1]))

        q_model = Model(input=x,output=q)

        return q_model

    def defineModels_old(self):
        with tf.device(self.device):
            x = Input(shape=(self.hist_size,self.seq_length,), dtype="uint8") # (STATES x SEQUENCE)
            # State Representation
            w_k = TimeDistributed(Embedding(output_dim=self.embd_size, mask_zero=True,
                            input_dim=self.vocab_size,
                            input_length=self.seq_length), name="embedding")(x) # (STATES x SEQUENCE x EMBEDDING)
            x_k = TimeDistributed(LSTM(self.h1, return_sequences=True), name="lstm1")(w_k) # (STATES x SEQUENCE x H1)
            v_s = TimeDistributed(GlobalAveragePooling1D(), name="avg")(x_k) # (STATES x H1)
            # history based Q function approximation
            q_hidden = SimpleRNN(self.h2, activation="relu", name="history_rnn")(v_s) # (H2)
            # action value
            qsa = Dense(self.action_size, name="action_dense")(q_hidden) # (ACTIONS)
            # object value
            qso = Dense(self.object_size, name="object_dense")(q_hidden) # (OBJECTS)

            q = merge(  [qsa,qso],
                        mode=lambda x: (K.expand_dims(x[0],2)+K.expand_dims(x[1],1))/2,
                        output_shape=lambda x: (x[0][0],x[0][1],x[1][1]))

            q_model = Model(input=x,output=q)

        return q_model

    def updateWeights(self):
        self.model_cpu.set_weights(self.model_gpu.get_weights())

    def predictQval(self,s,gpu=False):
        if gpu:
            return self.model_gpu.predict(np.atleast_3d(s))
        else:
            return self.model_cpu.predict(np.atleast_3d(s))

    def predictAction(self,s):
        q = self.predictQval([s])[0]
        return np.unravel_index(q.argmax(),q.shape)

    def randomAction(self):
        act = np.random.randint(0, self.action_size)
        obj = np.random.randint(0, self.object_size)
        return (act,obj)

    def predictQmax(self,s,gpu=False):
        q = self.predictQval(s,gpu)
        return q.max(axis=(1,2))

    def calculateTargets(self,s_batch,a_batch,r_batch,t_batch,s2_batch):
        batch_size = s_batch.shape[0]
        # split action tuple
        act_batch, obj_batch = a_batch[:,0], a_batch[:,1]
        # Calculate targets
        target = self.predictQval(s_batch,gpu=True)
        qmax = self.predictQmax(s2_batch,gpu=True)
        # discount state values using the calculated targets
        for k in xrange(batch_size):
            a,o = act_batch[k],obj_batch[k]
            if t_batch[k]:
                # just the true reward if game is over
                target[k,a,o] = r_batch[k]
            else:
                # reward + gamma * max a'{ Q(s', a') }
                target[k,a,o] = r_batch[k] + self.gamma * qmax[k]
        return target

    def trainOnBatch(self,s_batch,a_batch,r_batch,t_batch,s2_batch):
        sess = KTF.get_session()
        self.epoch += 1
        target = self.calculateTargets(s_batch,a_batch,r_batch,t_batch,s2_batch)
        loss = self.model_gpu.train_on_batch(s_batch,target)
        summary = sess.run([self.merged], feed_dict={self.model_gpu.input:s_batch, self.model_gpu.output:target})
        self.writer.add_summary(summary[0],self.epoch)
        self.writer.flush()
        return loss

    def save(self,name,overwrite):
        self.model_gpu.save("q_"+name, overwrite=overwrite)
