#!/usr/bin/python
#
# author: rknaebel
#
# description:
#


import numpy as np
import random
from collections import deque

from models import RNNQLearner
from keras.preprocessing.text import text_to_word_sequence

from replay_buffer import PrioritizedReplayBuffer

# GYM: Environment lib
import gym
import gym_textgame

from arguments import getArguments

from datetime import datetime
from elasticsearch import Elasticsearch

indexer = dict()
def getIndex(word):
    if word not in indexer:
        indexer[word] = len(indexer)+1
    return indexer[word]

def sent2seq(sentence,length):
    seq = map(getIndex, text_to_word_sequence(sentence))
    return seq + [0]*(length-len(seq))

def initHist(state,hist_size=5):
    states = [np.zeros(len(state), dtype="int") for _ in range(hist_size)]
    history = deque(states,hist_size)
    history.append(state)
    return history

import copy
def addHistoryState(hist,state):
    hist2 = copy.copy(hist)
    hist2.append(state)
    return hist2

def initDB():
    # elasticsearch
    dt = datetime.now()
    es_index = "rnn-" + dt.strftime("%y-%m-%d-%H-%M")
    es = Elasticsearch()
    return (es, es_index)

def sendDocDB(handle,doc):
    es, idx = handle
    doc["timestamp"] = datetime.now()
    #print idx, doc
    es.index(index=idx, doc_type="textgame_result", body=doc)

if __name__ == "__main__":
    args = getArguments()

    es = initDB()

    # layer sizes
    epsilon = args.epsilon_start
    epsilon_step = (args.epsilon_start-args.epsilon_end)/args.epsilon_anneal_steps

    env = gym.make(args.env)
    env_eval = gym.make(args.env)
    # action_space = Tuple(Discrete(5), Discrete(8))
    num_actions = env.action_space.spaces[0].n
    num_objects = env.action_space.spaces[1].n
    vocab_size  = env.vocab_space
    seq_len     = env.seq_length
    hist_size   = args.history_size

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
        tr_ctr = 0
        for episode in range(args.episodes_per_epoch):
            loss1 = 0.
            loss2 = 0.
            cnt_invalid_actions = 0
            ep_reward = 0.
            # get initial input
            s_text = env.reset()
            s = sent2seq(s_text, seq_len)
            h = initHist(s,hist_size)
            #
            for j in xrange(args.max_ep_steps):
                tr_ctr += 1
                # show textual input if so
                #if args.render: env.render()
                # choose action
                if np.random.rand() <= epsilon:
                    a = env.action_space.sample()
                else:
                    a = model.predictAction(h)
                # anneal epsilon
                epsilon = max(0.2, epsilon-epsilon_step)
                # apply action, get rewards and new state s2
                s2_text, r, terminal, info = env.step(a)
                s2 = sent2seq(s2_text, seq_len)
                h2 = addHistoryState(h,s2)
                # add current exp to buffer
                replay_buffer.add(h, a, r, terminal, h2)
                # Keep adding experience to the memory until
                # there are at least minibatch size samples
                if  ((replay_buffer.size() > args.batch_size) and
                    (tr_ctr % 4 == 0)):
                    h_batch, a_batch, r_batch, t_batch, h2_batch = \
                        replay_buffer.sample_batch(args.batch_size)
                    # Update the networks each given the new target values
                    l, l1, l2 = model.trainOnBatch(h_batch, a_batch, r_batch, t_batch, h2_batch)
                    loss1 += l1; loss2 += l2
                    tr_ctr = 0

                s = s2
                h1 = h2
                ep_reward += r
                cnt_invalid_actions += 1 if r == -0.1 else 0

                if terminal: break

            ep_lens.append(j+1)
            invalids.append(cnt_invalid_actions)
            quests_complete.append(1 if terminal and r>=1 else 0)
            deaths.append(1 if terminal and r<=0 else 0)
            scores.append(ep_reward)

            #print("  Episode {:03d}/{:03d}/{:03d} | L(qsa) {:.4f} | L(qso) {:.4f} | len {:03d} | inval {:03d} | eps {:.4f} | r {: .2f} | {:1d} {:1d}".format(
            #    epoch+1, episode+1, args.episodes_per_epoch, loss1, loss2, ep_lens[-1], invalids[-1], epsilon, scores[-1],
            #    quests_complete[-1],deaths[-1]))
            sendDocDB(es, { "epoch" : epoch+1, "episode" : episode+1,
                            "length" : ep_lens[-1], "invalids" : invalids[-1],
                            "epsilon" : epsilon, "reward" : scores[-1],
                            "quest_complete" : quests_complete[-1],
                            "death" : deaths[-1], "mode" : "train"})
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
            h = initHist(s,hist_size)
            #
            for j in xrange(args.max_ep_steps):
                # show textual input if so
                if args.render: env_eval.render()
                # choose action
                if np.random.rand() <= 0.05:
                    a = env_eval.action_space.sample()
                else:
                    a = model.predictAction(h)
                # apply action, get rewards and new state s2
                s2_text, r, terminal, info = env_eval.step(a)
                s2 = sent2seq(s2_text, seq_len)
                h2 = addHistoryState(h,s2)

                s = s2
                h = h2
                ep_reward += r
                cnt_invalid_actions += 1 if r == -0.1 else 0

                if terminal: break


            ep_lens.append(j+1)
            invalids.append(cnt_invalid_actions)
            quests_complete.append(1 if terminal and r>=1 else 0)
            deaths.append(1 if terminal and r<=0 else 0)
            scores.append(ep_reward)
            sendDocDB(es, { "epoch" : epoch+1, "episode" : episode+1,
                            "length" : ep_lens[-1], "invalids" : invalids[-1],
                            "epsilon" : 0.05, "reward" : scores[-1],
                            "quest_complete" : quests_complete[-1],
                            "death" : deaths[-1], "mode" : "eval"})
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
