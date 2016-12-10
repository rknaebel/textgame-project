#!/usr/bin/python
#
# author: rknaebel
#
# description:
#

import numpy as np
import csv
import random

from models import NeuralQLearner
from preprocess import sent2seq

from replay_buffer import PrioritizedReplayBuffer

# GYM: Environment lib
import gym
import gym_textgame

from arguments import getArguments

if __name__ == "__main__":
    args = getArguments()

    # layer sizes
    epsilon = args.epsilon_start
    epsilon_step = (args.epsilon_start-args.epsilon_end)/args.epsilon_anneal_steps

    env = gym.make(args.env)
    env_eval = gym.make(args.env)
    num_actions = env.num_actions
    num_objects = env.num_objects
    vocab_size  = env.vocab_space
    seq_len     = env.seq_length

    if args.csv:
        train_csv_file = open("{}_train.csv".format(args.csv), "wb")
        train_csv = csv.writer(train_csv_file)
        eval_csv_file  = open("{}_eval.csv".format(args.csv), "wb")
        eval_csv = csv.writer(eval_csv_file)

    model = NeuralQLearner( seq_len,vocab_size,
                            args.embd_size,args.hidden1,args.hidden2,
                            num_actions,num_objects,
                            args.alpha,args.gamma,args.batch_size)

    # Initialize replay memory
    replay_buffer = PrioritizedReplayBuffer(args.buffer_size, args.random_seed)
    step_ctr = 0
    for epoch in range(args.max_epochs):
        scores = []
        ep_lens = []
        invalids = []
        quests_complete = []
        #
        # TRAIN Phase
        #
        for episode in range(args.episodes_per_epoch):
            loss = 0.
            cnt_invalid_actions = 0
            ep_reward = 0.
            # get initial input
            s_text = env.reset()
            s = sent2seq(s_text, seq_len)
            #
            for j in xrange(args.max_ep_steps):
                step_ctr += 1
                # show textual input if so
                if args.render: env.render()
                # choose action
                if np.random.rand() <= epsilon:
                    a = model.randomAction()
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
                    (step_ctr % args.rounds_per_learn == 0)):
                    s_batch, a_batch, r_batch, t_batch, s2_batch = \
                        replay_buffer.sample_batch(args.batch_size)
                    # Update the networks each given the new target values
                    l = model.trainOnBatch(s_batch, a_batch, r_batch, t_batch, s2_batch)
                    loss += l
                    step_ctr = 0

                s = s2
                ep_reward += r
                cnt_invalid_actions += 1 if r == -0.1 else 0
                if terminal: break

            ep_lens.append(j+1)
            invalids.append(cnt_invalid_actions)
            quests_complete.append(int(r >= 1))
            scores.append(ep_reward)

            if args.csv:
                train_csv.writerow((epoch+1, episode+1, args.episodes_per_epoch, loss, ep_lens[-1], invalids[-1], epsilon, scores[-1],
                quests_complete[-1]))
        print("> Training   {:03d} | len {:02.2f} | inval {:02.2f} | quests {:02.2f} | r {: .2f} ".format(
            epoch+1, np.mean(ep_lens),
            np.mean(invalids),
            np.mean(quests_complete),
            np.mean(scores)))

        #
        # EVAL Phase
        #
        scores = []
        ep_lens = []
        invalids = []
        quests_complete = []
        for episode in range(args.episodes_per_epoch):
            cnt_invalid_actions = 0
            ep_reward = 0.
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
            scores.append(ep_reward)
            quests_complete.append(1 if terminal else 0)
            if args.csv:
                eval_csv.writerow((epoch+1, episode+1, args.episodes_per_epoch, loss1, loss2, ep_lens[-1], invalids[-1], scores[-1],
                quests_complete[-1]))
        print("> Evaluation {:03d} | len {:.2f} | inval {:.2f} | quests {:.2f} | r {: .2f} ".format(
            epoch+1, np.mean(ep_lens),
            np.mean(invalids),
            np.mean(quests_complete),
            np.mean(scores)))
        # Save trained model weights and architecture, this will be used by the visualization code
        model.save("model.h5", overwrite=True)
