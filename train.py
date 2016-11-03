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

from replay_buffer import PrioritizedReplayBuffer

# GYM: Environment lib
import gym
import gym_textgame

indexer = dict()
def getIndex(word):
    if word not in indexer:
        indexer[word] = len(indexer)+1
    return indexer[word]

def sent2seq(sentence,length):
    seq = map(getIndex, text_to_word_sequence(sentence))
    return seq + [0]*(length-len(seq))


def getModels(seq_length, vocab_size, embd_size, h1, h2, action_size, object_size):
    x = Input(shape=(seq_length,), dtype="int32")
    # State Representation
    w_k = Embedding(output_dim=embd_size, input_dim=vocab_size, input_length=seq_length)(x)
    x_k = LSTM(h1, return_sequences=True)(w_k)
    y = AveragePooling1D(pool_length=seq_length, stride=None)(x_k)
    v_s = Reshape((h1,))(y) # remove 2 axis which is 1 caused by averaging
    # Q function approximation
    q = Dense(h2, activation="relu")(v_s)
    q_sa = Dense(action_size)(q)
    q_so = Dense(object_size)(q)

    m_sa = Model(x,q_sa)
    m_so = Model(x,q_so)

    return m_sa, m_so

if __name__ == "__main__":
    BUFFER_SIZE = 100000
    RANDOM_SEED = 42
    MAX_EPOCHS = 100
    MAX_EPISODES = 100
    EPISODES_PER_EPOCH = 50
    MAX_EP_STEPS = 20
    MINIBATCH_SIZE = 64
    RENDER_ENV = False
    ROUNDS_PER_LEARN = 4
    GAMMA = 0.5 # discount factor
    EPSILON_START = 1.  # exploration
    EPSILON_END   = 0.2
    EPSILON_ANNEAL_STEPS = 1e5
    ALPHA = 5e-4 # learning rate

    # layer sizes
    epsilon = EPSILON_START
    embedding_size = 20
    hidden1_size = 50
    hidden2_size = 50

    env = gym.make("HomeWorld-v0")
    # action_space = Tuple(Discrete(5), Discrete(8))
    num_actions = env.action_space.spaces[0].n
    num_objects = env.action_space.spaces[1].n
    vocab_size  = env.vocab_space
    seq_len     = 100

    qsa_model, qso_model = getModels(seq_len,vocab_size,embedding_size,
                                     hidden1_size,hidden2_size,
                                     num_actions,num_objects)
    qsa_model.compile(loss="mse",optimizer=RMSprop(ALPHA))
    qso_model.compile(loss="mse",optimizer=RMSprop(ALPHA))

    # Initialize replay memory
    replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    # Train
    #scores = []
    for epoch in range(MAX_EPOCHS):
        scores = []
        #
        # TRAIN Phase
        #
        cnt_quest_complete = 0
        for episode in range(EPISODES_PER_EPOCH):
            loss1 = 0.
            loss2 = 0.
            cnt_invalid_actions = 0
            ep_reward = 0.
            # get initial input
            s_text = env.reset()
            s = sent2seq(s_text, seq_len)
            #
            for j in xrange(MAX_EP_STEPS):
                # show textual input if so
                if RENDER_ENV: env.render()
                # choose action
                if np.random.rand() <= epsilon:
                    act = np.random.randint(0, num_actions)
                    obj = np.random.randint(0, num_objects)
                    a = (act,obj)
                else:
                    qsa = qsa_model.predict(np.atleast_2d(s))
                    qso = qso_model.predict(np.atleast_2d(s))
                    a = (np.argmax(qsa[0]), np.argmax(qso[0]))
                # anneal epsilon
                epsilon = max(0.2, epsilon-(EPSILON_START-EPSILON_END)/ EPSILON_ANNEAL_STEPS)
                # apply action, get rewards and new state s2
                s2_text, r, terminal, info = env.step(a)
                s2 = sent2seq(s2_text, seq_len)
                # add current exp to buffer
                replay_buffer.add(s, a, r, terminal, s2)
                # Keep adding experience to the memory until
                # there are at least minibatch size samples
                if  ((replay_buffer.size() > MINIBATCH_SIZE) and
                    (j % ROUNDS_PER_LEARN == 0)):
                    s_batch, a_batch, r_batch, t_batch, s2_batch = \
                        replay_buffer.sample_batch(MINIBATCH_SIZE)

                    # split action tuple
                    act_batch, obj_batch = a_batch[:,0], a_batch[:,1]

                    # Calculate targets
                    target_qsa = qsa_model.predict(s_batch)
                    target_qso = qso_model.predict(s_batch)
                    qsa = qsa_model.predict(s2_batch).max(axis=1)
                    qso = qso_model.predict(s2_batch).max(axis=1)

                    # discount state values using the calculated targets
                    y_i = []
                    for k in xrange(MINIBATCH_SIZE):
                        if t_batch[k]:
                            # just the true reward if game is over
                            target_qsa[k,act_batch[k]] = r_batch[k]
                            target_qso[k,obj_batch[k]] = r_batch[k]
                        else:
                            # reward + gamma * max a'{ Q(s', a') }
                            target_qsa[k,act_batch[k]] = r_batch[k] + GAMMA * qsa[k]
                            target_qso[k,obj_batch[k]] = r_batch[k] + GAMMA * qso[k]

                    # Update the networks each given the new target values
                    loss1 += qsa_model.train_on_batch(s_batch,target_qsa)
                    loss2 += qso_model.train_on_batch(s_batch,target_qso)

                s = s2
                ep_reward += r
                cnt_invalid_actions += 1 if r == -0.1 else 0

                if terminal:
                    cnt_quest_complete += 1
                    #print '| Reward: %.2i' % int(ep_reward)
                    break
            scores.append(ep_reward)
            print("  Episode {:03d}/{:03d}/{:03d} | L(qsa) {:.4f} | L(qso) {:.4f} | len {:02d} | inval {:02d} | quests {:02d} | r {:.2f} | {}".format(
                epoch+1, episode+1, EPISODES_PER_EPOCH, loss1, loss2, j+1, cnt_invalid_actions, cnt_quest_complete, scores[-1],
                "X" if terminal else " "))

        #
        # EVAL Phase
        #cd
        scores = []
        ep_lens = []
        invalids = []
        quests_complete = []
        for episode in range(EPISODES_PER_EPOCH):
            ep_reward = 0.
            # get initial input
            s_text = env.reset()
            s = sent2seq(s_text, seq_len)
            #
            for j in xrange(MAX_EP_STEPS):
                # show textual input if so
                if RENDER_ENV: env.render()
                # choose action
                if np.random.rand() <= 0.05:
                    act = np.random.randint(0, num_actions)
                    obj = np.random.randint(0, num_objects)
                    a = (act,obj)
                else:
                    qsa = qsa_model.predict(np.atleast_2d(s))
                    qso = qso_model.predict(np.atleast_2d(s))
                    a = (np.argmax(qsa[0]), np.argmax(qso[0]))
                # apply action, get rewards and new state s2
                s2_text, r, terminal, info = env.step(a)
                s2 = sent2seq(s2_text, seq_len)

                s = s2
                ep_reward += r

                if terminal: break

            ep_lens.append(j+1)
            invalids.append(cnt_invalid_actions)
            quests_complete.append(1 if terminal else 0)
            scores.append(ep_reward)
        print("> Evaluation {:03d} | len {:.2f} | inval {:.2f} | quests {:.2f} | r {:.2f} ".format(
            epoch+1, np.mean(ep_lens),
            np.mean(cnt_invalid_actions),
            np.mean(cnt_quest_complete),
            np.mean(scores)))
    # Save trained model weights and architecture, this will be used by the visualization code
    qsa_model.save("qsa_model.h5", overwrite=True)
    qso_model.save("qso_model.h5", overwrite=True)
    #with open("model.json", "w") as outfile:
    #    json.dump(model.to_json(), outfile)
