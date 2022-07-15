#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ! conda install pytorch torchvision torchaudio cpuonly -c pytorch


# # Import

# In[11]:


import os
import argparse

import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import get_device, set_seed, tournament, reorganize, Logger


# # Training 

# In[12]:


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

#for episode in range(10):
for episode in range(2000):
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





# # Saving Trained Model

# In[13]:


save_path_1 = os.path.join('', 'agent1DQN_model1.pth')
save_path_2 = os.path.join('', 'agent2DQN_model1.pth')


# In[14]:


#saving model 
torch.save(agents[0], save_path_1)
torch.save(agents[2], save_path_2)


# In[31]:


print('Saved Location for DQN Agent 1:', save_path_1)
print('Saved Location for DQN Agent 2:', save_path_2)


# # Loading Trained Model

# In[15]:


# #loading models
# ag1 = torch.load('agent1DQN_model1.pth')
# ag3 = torch.load('agent2DQN_model1.pth')


# # Testing

# ## Testing for current/ live trained model

# In[16]:


# # Simulation

# def run_1_simulation():
#     trajectories, payoffs = env.run()
#     if payoffs[0] > payoffs[1]:
#         return 0
#     else:
#         return 1


# In[25]:


# using direct trained agent


# In[17]:


# l1 = [run_1_simulation() for i in range(2000)] 

# sum(l1) / len(l1)


# In[18]:


# 1 - sum(l1) / len(l1)


# In[ ]:





# In[ ]:





# ## Testing simulation for loaded models

# In[26]:


# # Check whether gpu is available
# device = get_device()

# # Seed numpy, torch, random
# set_seed(7)

# # Make the environment with seed
# load_env = rlcard.make("bridge", config={'seed': 7})

# # Initialize the agent and use random agents as opponents
# from rlcard.agents import DQNAgent
# agent1 = torch.load('agent1DQN_model1.pth')
# agent2 = RandomAgent(num_actions=load_env.num_actions, name='Player 2')

# agent3 = torch.load('agent2DQN_model1.pth')
# agent4 = RandomAgent(num_actions=load_env.num_actions, name='Player 4')

# agents = [agent1, agent2, agent3, agent4]

# load_env.set_agents(agents)


# In[27]:


# # Simulation

# def run_1_simulation():
#     trajectories, payoffs = load_env.run()
#     if payoffs[0] > payoffs[1]:
#         return 0
#     else:
#         return 1


# In[28]:


# l2 = [run_1_simulation() for i in range(2000)] 

# sum(l2) / len(l2)


# In[30]:


# 1 - sum(l2) / len(l2)


# In[ ]:





# # Game Play

# Play Game for 1 fateiyah

# In[19]:


# agent1.verbose = True
# agent2.verbose = True
# agent3.verbose = True


# In[ ]:





# In[22]:


# ##### import os
# import argparse

# import torch

# import rlcard
# from rlcard.agents import RandomAgent
# from rlcard.utils import get_device, set_seed, tournament, reorganize, Logger
# from rlcard.agents.human_agent_bridge import HumanAgentBridge

# env2 = rlcard.make("bridge", config={'seed': 7, 'verbose': True})

# agents = [
#           agent1,
#           agent2,
#           agent3,
#           #RandomAgent(num_actions=env2.num_actions, name='Player 1', verbose=True),
#           #RandomAgent(num_actions=env2.num_actions, name='Player 2', verbose=True),
#           #RandomAgent(num_actions=env2.num_actions, name='Player 3', verbose=True),
#           HumanAgentBridge(num_actions=env2.num_actions, name='Player 4', verbose=True)
#          ]


# env2.set_agents(agents)

# _, payoffs = env2.run()


# In[ ]:





# In[ ]:





# In[23]:


# if payoffs[1] > payoffs[0]:
#     print('Your Team: Won')
# else:
#     print('Your Team: Lose')


# In[ ]:




