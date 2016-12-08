#!/usr/bin/python
#
# author:
#
# date:
# description:
#
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--retrain", action="store_true", dest="retrain",
                    default=False)

parser.add_argument("--model", action="store", dest="model",
                    default="./model_checkpoints/savepoint.pkl")

parser.add_argument("--csv", action="store", dest="csv",
                    default="")

parser.add_argument("--exp_id", action="store", dest="exp_id",
                    default="")

parser.add_argument("--env", action="store", dest="env",
                    default="HomeWorld-v0")

#parser.add_argument("--sequence_len", action="store", dest="seq_len",
#                    default=100)

parser.add_argument("--embd", action="store", dest="embd_size",
                    default=20, type=int)

parser.add_argument("--h1", action="store", dest="hidden1",
                    default=100, type=int) # 50

parser.add_argument("--h2", action="store", dest="hidden2",
                    default=100, type=int) # 50

parser.add_argument("--buffer_size", action="store", dest="buffer_size",
                    default=100000, type=int)

parser.add_argument("--random_seed", action="store", dest="random_seed",
                    default=42, type=int)

parser.add_argument("--history_size", action="store", dest="history_size",
                    default=10, type=int)

parser.add_argument("--max_epochs", action="store", dest="max_epochs",
                    default=100, type=int)

parser.add_argument("--episodes_per_epoch", action="store", dest="episodes_per_epoch",
                    default=50, type=int)

parser.add_argument("--max_ep_steps", action="store", dest="max_ep_steps",
                    default=20, type=int)

parser.add_argument("--render", action="store", dest="render",
                    default=False, type=bool)
#
# learning parameters
#
parser.add_argument("--batch_size", action="store", dest="batch_size",
                    default=64, type=int)

parser.add_argument("--rounds_per_learn", action="store", dest="rounds_per_learn",
                    default=4, type=int)

parser.add_argument("--alpha", action="store", dest="alpha",
                    default=5e-5, type=float)

parser.add_argument("--gamma", action="store", dest="gamma",
                    default=5e-1, type=float)

#
# epsilon annealing things
#
parser.add_argument("--epsilon_start", action="store", dest="epsilon_start",
                    default=1.0, type=float)

parser.add_argument("--epsilon_end", action="store", dest="epsilon_end",
                    default=0.2, type=float)

parser.add_argument("--epsilon_anneal_steps", action="store", dest="epsilon_anneal_steps",
                    default=1e5, type=float) # 5e4


def getArguments():
    return parser.parse_args()
