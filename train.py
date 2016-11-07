#!/usr/bin/python
#
# author: rknaebel
#
# description:
#
import numpy as np

from model import NeuralQLearner
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


if __name__ == "__main__":
    BUFFER_SIZE = 100000
    RANDOM_SEED = 42
    MAX_EPOCHS = 50 # PAPER: 100
    EPISODES_PER_EPOCH = 50
    MAX_EP_STEPS = 20
    MINIBATCH_SIZE = 64 ## PAPER: 64
    RENDER_ENV = True
    ROUNDS_PER_LEARN = 4
    GAMMA = 0.5 # discount factor
    EPSILON_START = 1.  # exploration
    EPSILON_END   = 0.2
    EPSILON_ANNEAL_STEPS = 5e4 ## PAPER: 1e5
    EPSILON_STEP = (EPSILON_START-EPSILON_END)/EPSILON_ANNEAL_STEPS
    ALPHA = 5e-4 # learning rate

    # layer sizes
    epsilon = EPSILON_START
    embedding_size = 20
    hidden1_size = 50 ## PAPER: 100
    hidden2_size = 50 ## PAPER: 100

    env = gym.make("HomeWorld-v0")
    # action_space = Tuple(Discrete(5), Discrete(8))
    num_actions = env.action_space.spaces[0].n
    num_objects = env.action_space.spaces[1].n
    vocab_size  = env.vocab_space
    seq_len     = 100

    model = NeuralQLearner( seq_len,vocab_size,embedding_size,
                            hidden1_size,hidden2_size,
                            num_actions,num_objects,ALPHA,GAMMA,MINIBATCH_SIZE)

    # Initialize replay memory
    replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    # Train
    #scores = []
    for epoch in range(MAX_EPOCHS):
        scores = []
        ep_lens = []
        invalids = []
        quests_complete = []
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
                    a = env.action_space.sample()
                else:
                    a = model.predictAction(s)
                # anneal epsilon
                epsilon = max(0.2, epsilon-EPSILON_STEP)
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
                    # Update the networks each given the new target values
                    l1, l2 = model.trainOnBatch(s_batch, a_batch, r_batch, t_batch, s2_batch)
                    loss1 += l1; loss2 += l2

                s = s2
                ep_reward += r
                cnt_invalid_actions += 1 if r == -0.1 else 0

                if terminal:
                    cnt_quest_complete += 1
                    break

            ep_lens.append(j+1)
            invalids.append(cnt_invalid_actions)
            quests_complete.append(1 if terminal else 0)
            scores.append(ep_reward)

            print("  Episode {:03d}/{:03d}/{:03d} | L(qsa) {:.4f} | L(qso) {:.4f} | len {:02d} | inval {:02d} | eps {:.4f} | r {: .2f} | {:02d}".format(
                epoch+1, episode+1, EPISODES_PER_EPOCH, loss1, loss2, ep_lens[-1], invalids[-1], epsilon, scores[-1],
                quests_complete[-1]))
        print("> Training   {:03d} | len {:02.2f} | inval {:02.2f} | quests {:02.2f} | r {: .2f} ".format(
            epoch+1, np.mean(ep_lens),
            np.mean(invalids),
            np.mean(quests_complete),
            np.mean(scores)))

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
                    a = env.action_space.sample()
                else:
                    a = model.predictAction(s)
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
            np.mean(invalids),
            np.mean(quests_complete),
            np.mean(scores)))
        # Save trained model weights and architecture, this will be used by the visualization code
        model.save("model.h5", overwrite=True)
    #with open("model.json", "w") as outfile:
    #    json.dump(model.to_json(), outfile)
