import numpy as np
from rlcard.games.bridge.utils.bridge_card import BridgeCard
from rlcard.games.bridge.utils.action_event import ActionEvent

class RandomAgent(object):
	''' A random agent. Random agents is for running toy examples on the card games
	'''

	def __init__(self, num_actions, name='player', verbose=False):
		''' Initilize the random agent

		Args:
			num_actions (int): The size of the ouput action space
		'''
		self.use_raw = False
		self.num_actions = num_actions
		self.verbose = verbose
		self.name = name

	#@staticmethod
	def step(self, state):
		''' Predict the action given the curent state in gerenerating training data.

		Args:
			state (dict): An dictionary that represents the current state

		Returns:
			action (int): The action predicted (randomly chosen) by the random agent
		'''
		if self.verbose:
			_print_state(state, self.name)
		choice = np.random.choice(list(state['legal_actions'].keys()))
		if self.verbose:
			print('\nFinal Choice:', num_to_name(choice))
		return choice

	def eval_step(self, state):
		''' Predict the action given the current state for evaluation.
			Since the random agents are not trained. This function is equivalent to step function

		Args:
			state (dict): An dictionary that represents the current state

		Returns:
			action (int): The action predicted (randomly chosen) by the random agent
			probs (list): The list of action probabilities
		'''
		probs = [0 for _ in range(self.num_actions)]
		for i in state['legal_actions']:
			try:
				probs[i-1] = 1/len(state['legal_actions'])
			except:
				import pdb; pdb.set_trace()

		info = {}
		#if i > 36:
		#	import pdb; pdb.set_trace()
		'''
		info['probs'] = {state['raw_legal_actions'][i]: probs[list(state['legal_actions'].keys())[i]-1] 
							for i in range(len(state['legal_actions']))
						}
		'''
		try: 
			info['probs'] = {state['raw_legal_actions'][i]: probs[list(state['legal_actions'].keys())[i]] 
							for i in range(len(state['legal_actions']))
						}
		except:
			import pdb; pdb.set_trace()
		return self.step(state), info

def _print_state(state, name):
	try:
		l = state['obs']
		#print(len(l))
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
		print('\n-------------- STATE / Random Agent / {} --------------'.format(name))
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