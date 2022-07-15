''' DQN agent

The code is derived from https://github.com/dennybritz/reinforcement-learning/blob/master/DQN/dqn.py

Copyright (c) 2019 Matthew Judell
Copyright (c) 2019 DATA Lab at Texas A&M University
Copyright (c) 2016 Denny Britz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import random
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from copy import deepcopy

from rlcard.utils.utils import remove_illegal

from rlcard.games.bridge.utils.bridge_card import BridgeCard
from rlcard.games.bridge.utils.action_event import ActionEvent

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'legal_actions', 'done'])


class DQNAgent(object):
	'''
	Approximate clone of rlcard.agents.dqn_agent.DQNAgent
	that depends on PyTorch instead of Tensorflow
	'''
	def __init__(self,
				 replay_memory_size=20000,
				 replay_memory_init_size=100,
				 update_target_estimator_every=1000,
				 discount_factor=0.99,
				 epsilon_start=1.0,
				 epsilon_end=0.1,
				 epsilon_decay_steps=20000,
				 batch_size=32,
				 num_actions=2,
				 state_shape=None,
				 train_every=1,
				 mlp_layers=None,
				 learning_rate=0.00005,
				 device=None,
				 name='player', 
				 verbose=False):

		'''
		Q-Learning algorithm for off-policy TD control using Function Approximation.
		Finds the optimal greedy policy while following an epsilon-greedy policy.

		Args:
			replay_memory_size (int): Size of the replay memory
			replay_memory_init_size (int): Number of random experiences to sample when initializing
			  the reply memory.
			update_target_estimator_every (int): Copy parameters from the Q estimator to the
			  target estimator every N steps
			discount_factor (float): Gamma discount factor
			epsilon_start (float): Chance to sample a random action when taking an action.
			  Epsilon is decayed over time and this is the start value
			epsilon_end (float): The final minimum value of epsilon after decaying is done
			epsilon_decay_steps (int): Number of steps to decay epsilon over
			batch_size (int): Size of batches to sample from the replay memory
			evaluate_every (int): Evaluate every N steps
			num_actions (int): The number of the actions
			state_space (list): The space of the state vector
			train_every (int): Train the network every X steps.
			mlp_layers (list): The layer number and the dimension of each layer in MLP
			learning_rate (float): The learning rate of the DQN agent.
			device (torch.device): whether to use the cpu or gpu
		'''
		self.use_raw = False
		self.replay_memory_init_size = replay_memory_init_size
		self.update_target_estimator_every = update_target_estimator_every
		self.discount_factor = discount_factor
		self.epsilon_decay_steps = epsilon_decay_steps
		self.batch_size = batch_size
		self.num_actions = num_actions
		self.train_every = train_every
		self.verbose = verbose
		self.name = name

		# Torch device
		if device is None:
			self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		else:
			self.device = device

		# Total timesteps
		self.total_t = 0

		# Total training step
		self.train_t = 0

		# The epsilon decay scheduler
		self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

		# Create estimators
		self.q_estimator = Estimator(num_actions=num_actions, learning_rate=learning_rate, state_shape=state_shape, \
			mlp_layers=mlp_layers, device=self.device)
		self.target_estimator = Estimator(num_actions=num_actions, learning_rate=learning_rate, state_shape=state_shape, \
			mlp_layers=mlp_layers, device=self.device)

		# Create replay memory
		self.memory = Memory(replay_memory_size, batch_size)

	def feed(self, ts):
		''' Store data in to replay buffer and train the agent. There are two stages.
			In stage 1, populate the memory without training
			In stage 2, train the agent every several timesteps

		Args:
			ts (list): a list of 5 elements that represent the transition
		'''
		(state, action, reward, next_state, done) = tuple(ts)
		self.feed_memory(state['obs'], action, reward, next_state['obs'], list(next_state['legal_actions'].keys()), done)
		self.total_t += 1
		tmp = self.total_t - self.replay_memory_init_size
		if tmp>=0 and tmp%self.train_every == 0:
			#import pdb; pdb.set_trace()
			self.train()

	def step(self, state):
		''' Predict the action for genrating training data but
			have the predictions disconnected from the computation graph

		Args:
			state (numpy.array): current state

		Returns:
			action (int): an action id
		'''
		if self.verbose:
			_print_state(state, self.name)
		#import pdb; pdb.set_trace()
		q_values = self.predict(state)
		epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
		legal_actions = list(state['legal_actions'].keys())
		probs = np.ones(len(legal_actions), dtype=float) * epsilon / len(legal_actions)
		best_action_idx = legal_actions.index(np.argmax(q_values))
		probs[best_action_idx] += (1.0 - epsilon)
		action_idx = np.random.choice(np.arange(len(probs)), p=probs)

		choice = legal_actions[action_idx]
		if self.verbose:
			print('Final Choice:', num_to_name(choice))
		return choice

	def eval_step(self, state):
		''' Predict the action for evaluation purpose.

		Args:
			state (numpy.array): current state

		Returns:
			action (int): an action id
			info (dict): A dictionary containing information
		'''
		if self.verbose:
			_print_state(state, self.name)
		q_values = self.predict(state)
		best_action = np.argmax(q_values)
		
		if self.verbose:
			print('Final Choice:', num_to_name(best_action))
		
		info = {}
		info['values'] = {state['raw_legal_actions'][i]: float(q_values[list(state['legal_actions'].keys())[i]]) 
								for i in range(len(state['legal_actions']))
						 }

		return best_action, info

	def predict(self, state):
		''' Predict the masked Q-values

		Args:
			state (numpy.array): current state

		Returns:
			q_values (numpy.array): a 1-d array where each entry represents a Q value
		'''
		#import pdb; pdb.set_trace()
		q_values = self.q_estimator.predict_nograd(np.expand_dims(state['obs'], 0))[0]
		masked_q_values = -np.inf * np.ones(self.num_actions, dtype=float)
		legal_actions = list(state['legal_actions'].keys())
		
		masked_q_values[legal_actions] = q_values[legal_actions]

		return masked_q_values

	def train(self):
		''' Train the network

		Returns:
			loss (float): The loss of the current batch.
		'''
		state_batch, action_batch, reward_batch, next_state_batch, legal_actions_batch, done_batch = self.memory.sample()

		# Calculate best next actions using Q-network (Double DQN)
		q_values_next = self.q_estimator.predict_nograd(next_state_batch)
		legal_actions = []
		for b in range(self.batch_size):
			legal_actions.extend([i + b * self.num_actions for i in legal_actions_batch[b]])
		masked_q_values = -np.inf * np.ones(self.num_actions * self.batch_size, dtype=float)
		masked_q_values[legal_actions] = q_values_next.flatten()[legal_actions]
		masked_q_values = masked_q_values.reshape((self.batch_size, self.num_actions))
		best_actions = np.argmax(masked_q_values, axis=1)

		# Evaluate best next actions using Target-network (Double DQN)
		q_values_next_target = self.target_estimator.predict_nograd(next_state_batch)
		target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
			self.discount_factor * q_values_next_target[np.arange(self.batch_size), best_actions]

		# Perform gradient descent update
		state_batch = np.array(state_batch)

		loss = self.q_estimator.update(state_batch, action_batch, target_batch)
		print('\rINFO - Step {}, rl-loss: {}'.format(self.total_t, loss), end='')

		# Update the target estimator
		if self.train_t % self.update_target_estimator_every == 0:
			self.target_estimator = deepcopy(self.q_estimator)
			print("\nINFO - Copied model parameters to target network.")

		self.train_t += 1

	def feed_memory(self, state, action, reward, next_state, legal_actions, done):
		''' Feed transition to memory

		Args:
			state (numpy.array): the current state
			action (int): the performed action ID
			reward (float): the reward received
			next_state (numpy.array): the next state after performing the action
			legal_actions (list): the legal actions of the next state
			done (boolean): whether the episode is finished
		'''
		self.memory.save(state, action, reward, next_state, legal_actions, done)

	def set_device(self, device):
		self.device = device
		self.q_estimator.device = device
		self.target_estimator.device = device

class Estimator(object):
	'''
	Approximate clone of rlcard.agents.dqn_agent.Estimator that
	uses PyTorch instead of Tensorflow.	 All methods input/output np.ndarray.

	Q-Value Estimator neural network.
	This network is used for both the Q-Network and the Target Network.
	'''

	def __init__(self, num_actions=2, learning_rate=0.001, state_shape=None, mlp_layers=None, device=None):
		''' Initilalize an Estimator object.

		Args:
			num_actions (int): the number output actions
			state_shape (list): the shape of the state space
			mlp_layers (list): size of outputs of mlp layers
			device (torch.device): whether to use cpu or gpu
		'''
		self.num_actions = num_actions
		self.learning_rate=learning_rate
		self.state_shape = state_shape
		self.mlp_layers = mlp_layers
		self.device = device

		# set up Q model and place it in eval mode
		qnet = EstimatorNetwork(num_actions, state_shape, mlp_layers)
		qnet = qnet.to(self.device)
		self.qnet = qnet
		self.qnet.eval()

		# initialize the weights using Xavier init
		for p in self.qnet.parameters():
			if len(p.data.shape) > 1:
				nn.init.xavier_uniform_(p.data)

		# set up loss function
		self.mse_loss = nn.MSELoss(reduction='mean')

		# set up optimizer
		self.optimizer =  torch.optim.Adam(self.qnet.parameters(), lr=self.learning_rate)

	def predict_nograd(self, s):
		''' Predicts action values, but prediction is not included
			in the computation graph.  It is used to predict optimal next
			actions in the Double-DQN algorithm.

		Args:
		  s (np.ndarray): (batch, state_len)

		Returns:
		  np.ndarray of shape (batch_size, NUM_VALID_ACTIONS) containing the estimated
		  action values.
		'''
		with torch.no_grad():
			s = torch.from_numpy(s).float().to(self.device)
			q_as = self.qnet(s).cpu().numpy()
		return q_as

	def update(self, s, a, y):
		''' Updates the estimator towards the given targets.
			In this case y is the target-network estimated
			value of the Q-network optimal actions, which
			is labeled y in Algorithm 1 of Minh et al. (2015)

		Args:
		  s (np.ndarray): (batch, state_shape) state representation
		  a (np.ndarray): (batch,) integer sampled actions
		  y (np.ndarray): (batch,) value of optimal actions according to Q-target

		Returns:
		  The calculated loss on the batch.
		'''
		self.optimizer.zero_grad()

		self.qnet.train()

		s = torch.from_numpy(s).float().to(self.device)
		a = torch.from_numpy(a).long().to(self.device)
		y = torch.from_numpy(y).float().to(self.device)

		# (batch, state_shape) -> (batch, num_actions)
		q_as = self.qnet(s)

		# (batch, num_actions) -> (batch, )
		Q = torch.gather(q_as, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)

		# update model
		batch_loss = self.mse_loss(Q, y)
		batch_loss.backward()
		self.optimizer.step()
		batch_loss = batch_loss.item()

		self.qnet.eval()

		return batch_loss


class EstimatorNetwork(nn.Module):
	''' The function approximation network for Estimator
		It is just a series of tanh layers. All in/out are torch.tensor
	'''

	def __init__(self, num_actions=2, state_shape=None, mlp_layers=None):
		''' Initialize the Q network

		Args:
			num_actions (int): number of legal actions
			state_shape (list): shape of state tensor
			mlp_layers (list): output size of each fc layer
		'''
		super(EstimatorNetwork, self).__init__()

		self.num_actions = num_actions
		self.state_shape = state_shape
		self.mlp_layers = mlp_layers

		# build the Q network
		layer_dims = [np.prod(self.state_shape)] + self.mlp_layers
		fc = [nn.Flatten()]
		fc.append(nn.BatchNorm1d(layer_dims[0]))
		for i in range(len(layer_dims)-1):
			fc.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias=True))
			fc.append(nn.Tanh())
		fc.append(nn.Linear(layer_dims[-1], self.num_actions, bias=True))
		self.fc_layers = nn.Sequential(*fc)

	def forward(self, s):
		''' Predict action values

		Args:
			s  (Tensor): (batch, state_shape)
		'''
		return self.fc_layers(s)

class Memory(object):
	''' Memory for saving transitions
	'''

	def __init__(self, memory_size, batch_size):
		''' Initialize
		Args:
			memory_size (int): the size of the memroy buffer
		'''
		self.memory_size = memory_size
		self.batch_size = batch_size
		self.memory = []

	def save(self, state, action, reward, next_state, legal_actions, done):
		''' Save transition into memory

		Args:
			state (numpy.array): the current state
			action (int): the performed action ID
			reward (float): the reward received
			next_state (numpy.array): the next state after performing the action
			legal_actions (list): the legal actions of the next state
			done (boolean): whether the episode is finished
		'''
		if len(self.memory) == self.memory_size:
			self.memory.pop(0)
		transition = Transition(state, action, reward, next_state, legal_actions, done)
		self.memory.append(transition)

	def sample(self):
		''' Sample a minibatch from the replay memory

		Returns:
			state_batch (list): a batch of states
			action_batch (list): a batch of actions
			reward_batch (list): a batch of rewards
			next_state_batch (list): a batch of states
			done_batch (list): a batch of dones
		'''
		samples = random.sample(self.memory, self.batch_size)
		return map(np.array, zip(*samples))

		

def _print_state(state, name):
	try:
		l = state['obs']
		l1_hand_rep			  = l[0:208]
		l2_pile_rep			  = l[208:415+1]
		l3_otherplayer_rep	  = l[416:467+1]

		l4_dealer_rep	   = l[468:471+1]
		l5_current_player	  = l[472:475+1]

		l6_playcard_phase	  = l[476:476+1]
		l7_bidding_amount_rep = l[477:620+1]
		# l8_last_bid_amt_rep	  = l[621:656+1]

		l8_contract_bid_rep	  = l[621:627+1]
		l9_contract_trump_rep = l[628:632+1]
		l10_highest_bidder_rep = l[633:]

		suits = ['C', 'D', 'H', 'S']
		ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
		all_cards = [r+s for s in suits for r in ranks]
		print('\n-------------- STATE / DQN Agent / {} --------------'.format(name))
		#for pid, i in enumerate(range(0, len(l1_hand_rep), 52)):
		print('1. My Hand Rep:              ', [all_cards[idx%52] for idx, x in enumerate(l1_hand_rep) if x])
		# import pdb; pdb.set_trace()
		print('2. Pile Rep:                 ', [all_cards[idx%52] for idx, x in enumerate(l2_pile_rep) if x])
		print('3. Other Player Rep:         ', [all_cards[idx] for idx, x in enumerate(l3_otherplayer_rep) if x])
		player_id = [idx for idx, x in enumerate(l4_dealer_rep) if x][0]+1
		print('4. Dealer Rep:               ', '[Player {}]'.format(player_id) )
		player_id = [idx for idx, x in enumerate(l5_current_player) if x][0]+1
		print('5. Current Player:           ', '[Player {}]'.format(player_id) )
		print('6. Is PlayCard phase:        ', l6_playcard_phase)
		
		bid_action_id = [idx for idx, x in enumerate(l7_bidding_amount_rep) if x]
		# print('bid_action_id:', bid_action_id)
		if bid_action_id:
			bid_action = [bid_to_name(x) for x in bid_action_id]
		else:
			bid_action = '-'
		print('7. Last 3 Bidding Rep:       ', bid_action)
		
		# last_bid_action_id = [idx for idx, x in enumerate(l8_last_bid_amt_rep) if x]
		# print('last_bid_action_id:', last_bid_action_id)
		# last_bid_action = [num_to_name(last_bid_action_id[0])] if last_bid_action_id else '-'
		# print('8. Last bidding rep: ', last_bid_action)
		
		contract_bid_id = [idx for idx, x in enumerate(l8_contract_bid_rep) if x]
		contract_bid_name = [contract_bid_id[0]+7] if contract_bid_id else '-'
		print('8. Contract Bid rep:         ', contract_bid_name)
		
		contract_trump_id = [idx for idx, x in enumerate(l9_contract_trump_rep) if x]
		contract_trump_name = contract_trump_id[0] if contract_trump_id else '-'
		contract_trump_name = [['Club', 'Diamond', 'Hearts', 'Spade', 'NoTrump'][contract_trump_name]] if contract_trump_id else '-'
		print('9. Contract Trump rep:       ', contract_trump_name)

		highest_bidder_id = [idx for idx, x in enumerate(l10_highest_bidder_rep) if x]
		highest_bidder_name = ['Player {}'.format(highest_bidder_id[0]+1)] if highest_bidder_id else '-'
		print('10. Contract Highest Bidder: ', highest_bidder_name)
		
		#print('legal_actions: ', state['legal_actions'])
		legal_actions = [num_to_name(action_id) for action_id in  state['legal_actions']]
		print('\nPossible Legal Actions:', legal_actions)
	except Exception as e:
		print(e)
		import pdb; pdb.set_trace()

		
def bid_to_name(action_id):
	player_id = action_id // 36 + 1
	action_name = num_to_name(action_id % 36)
	return 'Player{}_{}'.format(player_id, action_name)
	

def num_to_name(action_id):
	#print('action_id:', action_id)
	#import pdb; pdb.set_trace()
	if ActionEvent.first_bid_action_id <= action_id <= ActionEvent.last_bid_action_id:
		#bid_amount = 7 + (action_id - 1) % 7
		#bid_suit_id = (action_id - 1) // 7

		bid_amount = 7 + (action_id) % 7
		bid_suit_id = (action_id) // 7
		bid_suit = BridgeCard.suits[bid_suit_id] if bid_suit_id < 4 else "nt"               # [C, D, H, S, nt]
		bid_suit = {'C': 'Club', 'S': 'Spade', 'D': 'Diamond', 'H': 'Hearts', 'nt': 'NoTrump'}[bid_suit]
		return f'{bid_amount}_{bid_suit}'
	elif action_id==35:
		return 'Pass'
	# If 37 <= action_id <= 88
	elif ActionEvent.first_play_card_action_id <= action_id <= ActionEvent.last_play_card_action_id:
		card_id = action_id - ActionEvent.first_play_card_action_id
		card = BridgeCard.card(card_id=card_id)
		return card
	return '-'