
from keras.layers import Input, Dense, Embedding, LSTM
from keras.layers import AveragePooling1D, Reshape
from keras.models import Model

from keras.preprocessing.text import text_to_word_sequence

indexer = dict()
def getIndex(word):
    if word not in indexer:
        indexer[word] = len(indexer)+1
    return indexer[word]

def sent2seq(sentence,length):
    seq = map(getIndex, text_to_word_sequence(sentence))
    return seq + [0]*(length-len(seq))

from replay_buffer import ReplayBuffer

import gym
import gym_textgame

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

if __name__ == "__main_":
    BUFFER_SIZE = 500
    RANDOM_SEED = 42
    MAX_EPISODES = 100
    MAX_EP_STEPS = 100
    MINIBATCH_SIZE = 100

    # parameters
    epsilon = .4  # exploration

    embedding_size = 100
    hidden1_size = 100
    hidden2_size = 100

    env = gym.make("HomeWorld-v0")
    # action_space = Tuple(Discrete(5), Discrete(8))
    num_actions = env.action_space.spaces[0].n
    num_objects = env.action_space.spaces[1].n
    vocab_size  = env.observation_space[0].n # !!TODO
    seq_len     = env.observation_space[1].n # !!TODO

    qsa_model, qso_model = getModels(seq_len,vocab_size,embedding_size,
                                     hidden1_size,hidden2_size,
                                     num_actions,num_objects)

    # Initialize experience replay memory
    replay_buffer = ExperienceReplay(BUFFER_SIZE, RANDOM_SEED)

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    # Train
    scores = []
    for e in range(MAX_EPISODES):
        loss = 0.
        game_over = False
        # get initial input
        s_text = env.reset()
        s = sent2seq(s_text)
        #s = env.observe()

        for j in xrange(MAX_EP_STEPS):
            # show textual input if so
            if RENDER_ENV: env.render()
            # choose action
            if np.random.rand() <= epsilon:
                act = np.random.randint(0, num_actions)
                obj = np.random.randint(0, num_objects)
                a = (act,obj)
            else:
                qsa = qsa_model.predict(s)
                qso = qso_model.predict(s)
                a = (np.argmax(qsa[0]), np.argmax(qso[0]))
            # apply action, get rewards and new state s2
            s2_text, r, terminal, info = env.step(a)
            s2 = sent2seq(s2_text)
            # add current exp to buffer
            replay_buffer.add(s, a, r, terminal, s2)
            #


            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)

                # split action tuple
                act_batch, obj_batch = a_batch[:,0], a_batch[:,1]

                # Calculate targets
                target_qsa = qsa_model.predict(s2_batch)
                target_qso = qso_model.predict(s2_batch)

                # discount state values using the calculated targets
                y_i = []
                for k in xrange(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append((r_batch[k],r_batch[k]))
                    else:
                        y_i.append((r_batch[k] + GAMMA * target_qsa[k],
                                    r_batch[k] + GAMMA * target_qso[k]))
                y_i = np.reshape(y_i, (MINIBATCH_SIZE, 2))

                # Update the networks each given the new target values
                qsa_model.train(s_batch,y_i[:,0])
                qso_model.train(s_batch,y_i[:,1])

                # Update the critic given the targets
                #predicted_q_value, _ = critic.train(s_batch, a_batch, y_i)

                #ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                #a_outs = actor.predict(s_batch)
                #grads = critic.action_gradients(s_batch, a_outs)
                #actor.train(s_batch, grads[0])

                # Update target networks
                #actor.update_target_network()
                #critic.update_target_network()

            s = s2
            ep_reward += r

            if terminal:

                print '| Reward: %.2i' % int(ep_reward), " | Episode", i, \
                    '| Qmax: %.4f' % (ep_ave_max_q / float(j))

                break





            # store experience
            exp_replay.remember([s, action, reward, s2], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)
            loss += model.train_on_batch(inputs, targets)
        print("Epoch {:03d}/{} | Loss {:.4f} | running avg {}".format(e, epoch, loss, scores[-1]))

    # Save trained model weights and architecture, this will be used by the visualization code
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)
