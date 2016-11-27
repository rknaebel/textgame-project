#!/usr/bin/python
#
# author: rknaebel
#
# description:
#


import numpy as np
import random

from model import RNNQLearner
from keras.preprocessing.text import text_to_word_sequence

from replay_buffer import PrioritizedReplayBuffer

# GYM: Environment lib
import gym
import gym_textgame

from arguments import getArguments

indexer = dict()
def getIndex(word):
    if word not in indexer:
        indexer[word] = len(indexer)+1
    return indexer[word]

def sent2seq(sentence,length):
    seq = map(getIndex, text_to_word_sequence(sentence))
    return seq + [0]*(length-len(seq))

def initHist(state_len=20,hist_len=5):
    states = [[0]*state_len for _ in range(hist_len)]
    return deque(states,hist_len)

if __name__ == "__main__":
    args = getArguments()

    # layer sizes
    epsilon = args.epsilon_start
    epsilon_step = (args.epsilon_start-args.epsilon_end)/args.epsilon_anneal_steps

    env = gym.make("HomeWorldHard-v0")
    env_eval = gym.make("HomeWorldHard-v0")
    # action_space = Tuple(Discrete(5), Discrete(8))
    num_actions = env.action_space.spaces[0].n
    num_objects = env.action_space.spaces[1].n
    vocab_size  = env.vocab_space
    seq_len     = 200
    hist_size   = 10


    model = RNNQLearner(seq_len,vocab_size,args.embd_size,hist_size,
                        args.hidden1,args.hidden2,
                        num_actions,num_objects,
                        args.alpha,args.gamma,args.batch_size)

    # Initialize replay memory
    replay_buffer = PrioritizedReplayBuffer(args.buffer_size, args.random_seed)

    for epoch in range(args.max_epochs):
        scores = []
        ep_lens = []
        invalids = []
        quests_complete = []
        deaths = []
        #
        # TRAIN Phase
        #
        cnt_quest_complete = 0
        for episode in range(args.episodes_per_epoch):
            loss1 = 0.
            loss2 = 0.
            cnt_invalid_actions = 0
            ep_reward = 0.
            # get initial input
            hist = initHist()
            s_text = env.reset()
            s = sent2seq(s_text, seq_len)
            #
            for j in xrange(20):
                # show textual input if so
                #if args.render: env.render()
                # choose action
                if np.random.rand() <= epsilon:
                    a = env.action_space.sample()
                else:
                    a = model.predictAction(s)
                # anneal epsilon
                epsilon = max(0.2, epsilon-epsilon_step)
                # apply action, get rewards and new state s2
                s2_text, r, terminal, info = env.step(a)
                s2 = sent2seq(s2_text, seq_len)
                # add current exp to buffer
                replay_buffer.add(s, a, r, terminal, s2)
                # Keep adding experience to the memory until
                # there are at least minibatch size samples
                if  ((replay_buffer.size() > args.batch_size) and
                    (j % 4 == 0)):
                    s_batch, a_batch, r_batch, t_batch, s2_batch = \
                        replay_buffer.sample_batch(args.batch_size)
                    # Update the networks each given the new target values
                    l, l1, l2 = model.trainOnBatch(s_batch, a_batch, r_batch, t_batch, s2_batch)
                    loss1 += l1; loss2 += l2

                s = s2
                ep_reward += r
                cnt_invalid_actions += 1 if r == -0.1 else 0

                if terminal: break

            ep_lens.append(j+1)
            invalids.append(cnt_invalid_actions)
            quests_complete.append(1 if terminal and r>=1 else 0)
            deaths.append(1 if terminal and r<=0 else 0)
            scores.append(ep_reward)

            print("  Episode {:03d}/{:03d}/{:03d} | L(qsa) {:.4f} | L(qso) {:.4f} | len {:03d} | inval {:03d} | eps {:.4f} | r {: .2f} | {:1d} {:1d}".format(
                epoch+1, episode+1, args.episodes_per_epoch, loss1, loss2, ep_lens[-1], invalids[-1], epsilon, scores[-1],
                quests_complete[-1],deaths[-1]))
        print("> Training   {:03d} | len {:02.2f} | inval {:02.2f} | quests {:02.2f} | deaths {:.2f} | r {: .2f} ".format(
            epoch+1, np.mean(ep_lens),
            np.mean(invalids),
            np.mean(quests_complete),
            np.mean(deaths),
            np.mean(scores)))

        #
        # EVAL Phase
        #
        scores = []
        ep_lens = []
        invalids = []
        quests_complete = []
        for episode in range(args.episodes_per_epoch):
            ep_reward = 0.
            cnt_invalid_actions = 0
            # get initial input
            seed = random.random()
            env_eval.seed(seed)
            s_text = env_eval.reset()
            s = sent2seq(s_text, seq_len)
            #
            for j in xrange(args.max_ep_steps):
                # show textual input if so
                if args.render: env_eval.render()
                # choose action
                if np.random.rand() <= 0.05:
                    a = env_eval.action_space.sample()
                else:
                    a = model.predictAction(s)
                # apply action, get rewards and new state s2
                s2_text, r, terminal, info = env_eval.step(a)
                s2 = sent2seq(s2_text, seq_len)

                s = s2
                ep_reward += r
                cnt_invalid_actions += 1 if r == -0.1 else 0

                if terminal: break


            ep_lens.append(j+1)
            invalids.append(cnt_invalid_actions)
            quests_complete.append(1 if terminal and r>=1 else 0)
            deaths.append(1 if terminal and r<=0 else 0)
            scores.append(ep_reward)
        print("> Evaluation {:03d} | len {:.2f} | inval {:.2f} | quests {:.2f} | deaths {:.2f} | r {: .2f} ".format(
            epoch+1, np.mean(ep_lens),
            np.mean(invalids),
            np.mean(quests_complete),
            np.mean(deaths),
            np.mean(scores)))
        # Save trained model weights and architecture, this will be used by the visualization code
        model.save("model.h5", overwrite=True)
    #with open("model.json", "w") as outfile:
    #    json.dump(model.to_json(), outfile)
