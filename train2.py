#!/usr/bin/python
#
# author: rknaebel
#
# description:
#
import numpy as np
import random

# GYM: Environment lib
import gym
import gym_textgame

from models import HistoryQLearner
from replay_buffer import PrioritizedReplayBuffer
from preprocess import sent2seq, initHist, addHistoryState

from arguments import getArguments
from utils import initDB, sendDocDB, sendModelDB

if __name__ == "__main__":
    args = getArguments()

    es = initDB()

    # layer sizes
    epsilon = args.epsilon_start
    epsilon_step = (args.epsilon_start-args.epsilon_end)/args.epsilon_anneal_steps

    env = gym.make(args.env)
    env_eval = gym.make(args.env)
    num_actions = env.num_actions
    num_objects = env.num_objects
    vocab_size  = env.vocab_space
    seq_len     = env.seq_length
    hist_size   = args.history_size

    model = HistoryQLearner(seq_len,vocab_size,args.embd_size,hist_size,
                        args.hidden1,args.hidden2,
                        num_actions,num_objects,
                        args.alpha,args.gamma)

    # Initialize replay memory
    replay_buffer = PrioritizedReplayBuffer(args.buffer_size, args.random_seed)

    sendModelDB(es,args,args.exp_id)
    #sendWeigthsDB(es,model)
    step_ctr = 0
    for epoch in range(args.max_epochs):
        scores = []
        ep_lens = []
        invalids = []
        quests_complete = []
        deaths = []
        #
        # TRAIN Phase
        #
        for episode in range(args.episodes_per_epoch):
            loss = 0.
            plan = []
            cnt_invalid_actions = 0
            ep_reward = 0.
            # get initial input
            init_s_text = env.reset()
            s = sent2seq(init_s_text, seq_len)
            h = initHist(s,hist_size)
            #
            for j in xrange(args.max_ep_steps):
                step_ctr += 1
                # show textual input if so
                #if args.render: env.render()
                # choose action
                if np.random.rand() <= epsilon:
                    a = model.randomAction()
                else:
                    a = model.predictAction(h)
                plan.append(env.get_action(a))
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
                    (step_ctr % args.rounds_per_learn == 0)):
                    h_batch, a_batch, r_batch, t_batch, h2_batch = \
                        replay_buffer.sample_batch(args.batch_size)
                    # Update the networks each given the new target values
                    l = model.trainOnBatch(h_batch, a_batch, r_batch, t_batch, h2_batch)
                    loss += l
                    step_ctr = 0

                s = s2
                h1 = h2
                ep_reward += r
                cnt_invalid_actions += 1 if r == -0.1 else 0

                if terminal: break

            ep_lens.append(j+1)
            invalids.append(cnt_invalid_actions)
            quests_complete.append(int(terminal and r >= 1))
            deaths.append(int(terminal and r <= 0))
            scores.append(ep_reward)

            sendDocDB(es, { "epoch" : epoch+1, "episode" : episode+1,
                            "length" : ep_lens[-1], "invalids" : invalids[-1],
                            "epsilon" : epsilon, "reward" : scores[-1],
                            "quest_complete" : quests_complete[-1],
                            "death" : deaths[-1], "mode" : "train",
                            "init_state" : init_s_text, "plan" : plan}, args.exp_id)
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
            plan = []
            ep_reward = 0.
            cnt_invalid_actions = 0
            # get initial input
            seed = random.random()
            env_eval.seed(seed)
            init_s_text = env_eval.reset()
            s = sent2seq(init_s_text, seq_len)
            h = initHist(s,hist_size)
            #
            for j in xrange(args.max_ep_steps):
                # show textual input if so
                if args.render: env_eval.render()
                # choose action
                if np.random.rand() <= 0.05:
                    a = model.randomAction()
                else:
                    a = model.predictAction(h)
                plan.append(env.get_action(a))
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
            quests_complete.append(int(terminal and r >= 1))
            deaths.append(int(terminal and r <= 0))
            scores.append(ep_reward)
            sendDocDB(es, { "epoch" : epoch+1, "episode" : episode+1,
                            "length" : ep_lens[-1], "invalids" : invalids[-1],
                            "epsilon" : 0.05, "reward" : scores[-1],
                            "quest_complete" : quests_complete[-1],
                            "death" : deaths[-1], "mode" : "eval",
                            "init_state" : init_s_text, "plan" : plan}, args.exp_id)
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
