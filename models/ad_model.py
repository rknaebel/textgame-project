#!/usr/bin/python
#
# author: rknaebel
#
# description:
#

class ActionDecisionModel(object):
    def predictAction(self, s): pass
    def trainOnBatch(self,s_batch,a_batch,r_batch,t_batch,s2_batch): pass
    def save(self,name,overwrite): pass

class ReinforceLearner(ActionDecisionModel):
    """
    Instead of measuring the absolute goodness of an action we want to know how much better than "average" it is to take an action given a state. E.g. some states are naturally bad and always give negative reward. This is called the advantage and is defined as Q(s, a) - V(s). We use that for our policy update, e.g. g_t - V(s) for REINFORCE
    """
    def predictAction(self, s): pass
    def trainOnBatch(self,s_batch,a_batch,r_batch,t_batch,s2_batch): pass
    def save(self,name,overwrite): pass

class ActorCriticLearner(object):
    """
    Two function approximators: One of the policy, one for the critic.
    This is basically TD, but for Policy Gradients
    """
    def predictAction(self, s): pass
    def trainOnBatch(self,s_batch,a_batch,r_batch,t_batch,s2_batch): pass
    def save(self,name,overwrite): pass

## A3C
# Instead of using an experience replay buffer as in DQN use multiple agents on different threads to explore the state spaces and make decorrelated updates to the actor and the critic.
