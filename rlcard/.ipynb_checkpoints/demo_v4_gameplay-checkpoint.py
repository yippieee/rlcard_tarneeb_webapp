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
from rlcard.agents import DQNAgent
from rlcard.agents.human_agent_bridge import HumanAgentBridge


# # Setting Up Agents

# In[2]:


# Initialize the agent and use random agents as opponents
# Seed numpy, torch, random
set_seed(7)

env2 = rlcard.make("bridge", config={'seed': 7, 'verbose': True})

agent1 = torch.load('agent1DQN_model1.pth')
agent2 = RandomAgent(num_actions=env2.num_actions, name='Player 2')

agent3 = torch.load('agent2DQN_model1.pth')
agent4 = HumanAgentBridge(num_actions=env2.num_actions, name='Player 4', verbose=True)


# # Game Play

# Play Game for 1 fateiyah

# In[5]:


agent1.verbose = True
agent2.verbose = True
agent3.verbose = True


# In[10]:


agents = [
          agent1,
          agent2,
          agent3,
          agent4
         ]


env2.set_agents(agents)

_, payoffs = env2.run()


# In[ ]:





# In[ ]:





# In[9]:


if payoffs[1] > payoffs[0]:
    print('Your Team: Won')
else:
    print('Your Team: Lose')


# In[ ]:





# In[ ]:




