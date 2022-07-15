#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ! conda install pytorch torchvision torchaudio cpuonly -c pytorch


# # Import

# In[1]:


import os
import argparse

import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import get_device, set_seed, tournament, reorganize, Logger


# # Training 

# In[1]:


#### import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import get_device, set_seed, tournament, reorganize, Logger


# Check whether gpu is available
device = get_device()

# Seed numpy, torch, random
set_seed(7)

# Make the environment with seed
env = rlcard.make("bridge", config={'seed': 7})

# Initialize the agent and use random agents as opponents
from rlcard.agents import DQNAgent
agent1 = DQNAgent(num_actions=env.num_actions,
                 state_shape=env.state_shape[0],
                 mlp_layers=[700 , 700],
                 device=device,
                 name='Player 1')

agent2 = RandomAgent(num_actions=env.num_actions, name='Player 2')

agent3 = DQNAgent(num_actions=env.num_actions,
                  state_shape=env.state_shape[0],
                  mlp_layers=[700 , 700],
                  device=device,
                  name='Player 3',
                 )

agent4 = RandomAgent(num_actions=env.num_actions, name='Player 4')


'''
agents = [agent]
for _ in range(env.num_players):
    agents.append(RandomAgent(num_actions=env.num_actions))
'''

agents = [agent1, agent2, agent3, agent4]

env.set_agents(agents)

# for episode in range(2000):
for episode in range(10):
    #print('\n episode:', episode)
    trajectories, payoffs = env.run(is_training=True)
    #break
    # Training
    trajectories = reorganize(trajectories, payoffs)
    for ts in trajectories[0]:
        agent1.feed(ts)
        #agent.feed(ts)
        #rl_loss = agent.train_rl()
        #sl_loss = agent.train_sl()
    for ts in trajectories[2]:
        agent3.feed(ts)


# In[ ]:





# In[ ]:





# # Testing

# In[2]:


# Simulation
'''
def run_1_simulation():
    trajectories, payoffs = env.run()
    if payoffs[0] > payoffs[1]:
        return 0
    else:
        return 1


# In[3]:


l = [run_1_simulation() for i in range(2000)] 

sum(l) / len(l)


# In[4]:


1 - sum(l) / len(l)
'''


# In[ ]:





# In[2]:





# # Game Play

# Play Game for 1 fateiyah

# In[2]:


agent1.verbose = True
agent2.verbose = True
agent3.verbose = True


# In[ ]:





# In[4]:


##### import os
import argparse

import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import get_device, set_seed, tournament, reorganize, Logger
from rlcard.agents.human_agent_bridge import HumanAgentBridge

env2 = rlcard.make("bridge", config={'seed': 7, 'verbose': True})

agents = [
          agent1,
          agent2,
          agent3,
          #RandomAgent(num_actions=env2.num_actions, name='Player 1', verbose=True),
          #RandomAgent(num_actions=env2.num_actions, name='Player 2', verbose=True),
          #RandomAgent(num_actions=env2.num_actions, name='Player 3', verbose=True),
          HumanAgentBridge(num_actions=env2.num_actions, name='Player 4', verbose=True)
         ]

env2.set_agents(agents)

_, payoffs = env2.run()


# In[ ]:





# In[ ]:





# In[5]:


if payoffs[1] > payoffs[0]:
    print('Your Team: Won')
else:
    print('Your Team: Lose')


# In[ ]:




