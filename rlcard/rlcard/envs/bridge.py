'''
	File name: envs/bridge.py
	Author: William Hale
	Date created: 11/26/2021
'''

import numpy as np
from collections import OrderedDict

from rlcard.envs import Env

from rlcard.games.bridge import Game

from rlcard.games.bridge.game import BridgeGame
from rlcard.games.bridge.utils.action_event import ActionEvent
from rlcard.games.bridge.utils.bridge_card import BridgeCard
from rlcard.games.bridge.utils.move import CallMove, PlayCardMove, MakeBidMove

#	[] Why no_bid_action_id in bidding_rep ?
#		It allows the bidding always to start with North.
#		If North is not the dealer, then he must call 'no_bid'.
#		Until the dealer is reached, 'no_bid' must be the call.
#		I think this might help because it keeps a player's bid in a fixed 'column'.
#		Note: the 'no_bid' is only inserted in the bidding_rep, not in the actual game.
#
#	[] Why current_player_rep ?
#		Explanation here.
#
#	[] Note: hands_rep maintain the hands by N, E, S, W.
#
#	[] Note: trick_rep maintains the trick cards by N, E, S, W.
#	   The trick leader can be deduced since play is in clockwise direction.
#
#	[] Note: is_bidding_rep can be deduced from bidding_rep.
#	   I think I added is_bidding_rep before bidding_rep and thus it helped in early testing.
#	   My early testing had just the player's hand: I think the model conflated the bidding phase with the playing phase in this situation.
#	   Although is_bidding_rep is not needed, keeping it may improve learning.
#
#	[] Note: bidding_rep uses the action_id instead of one hot encoding.
#	   I think one hot encoding would make the input dimension significantly larger.
#


########################### Env #################################

'''
Player's
   1. Extract state
   2. Action
   2. Player's payoff
   
State  ->  Action  ->  Reward
'''


class BridgeEnv(Env):
	''' Bridge Environment
	'''
	def __init__(self, config):
		#import pdb; pdb.set_trace()
		self.name = 'bridge'
		self.verbose = config.pop('verbose', False)
		self.game = Game(verbose=self.verbose)
		super().__init__(config=config)
		self.bridgePayoffDelegate = DefaultBridgePayoffDelegate(self.verbose)
		self.bridgeStateExtractor = DefaultBridgeStateExtractor()
		state_shape_size = self.bridgeStateExtractor.get_state_shape_size()
		self.state_shape = [[1, state_shape_size] for _ in range(self.num_players)]
		self.action_shape = [None for _ in range(self.num_players)]

	def get_payoffs(self):
		''' 
		#2
		Get the payoffs of players.

		Returns:
			(list): A list of payoffs for each player.
			
		e.g. [1, 9, 2, 9]
		'''
		return self.bridgePayoffDelegate.get_payoffs(game=self.game)

	def get_perfect_information(self):
		''' 
		#2
		[Get] Get the perfect information of the current state

		Returns:
			(dict): A dictionary of all the perfect information of the current state
		'''
		return self.game.round.get_perfect_information()

	def _extract_state(self, state):  # wch: don't use state 211126
		''' 
		#2
		Extract useful information from state for RL.

		Args:
			state (dict): The raw state

		Returns:
			(numpy.array): The extracted state
		'''
		return self.bridgeStateExtractor.extract_state(game=self.game)

	def _decode_action(self, action_id):
		''' 
		
		Decode Action id to the action in the game.

		Args:
			action_id (int): The id of the action

		Returns:
			(ActionEvent): The action that will be passed to the game engine.
		'''
		return ActionEvent.from_action_id(action_id=action_id)

	def _get_legal_actions(self):
		''' Get all legal actions for current state.

		Returns:
			(list): A list of legal actions' id.
		'''
		raise NotImplementedError  # wch: not needed


	   
	
########################### Helper Functions / 1. Extract Payoff #################################

class BridgePayoffDelegate(object):

	def get_payoffs(self, game: BridgeGame):
		''' Get the payoffs of players. Must be implemented in the child class.

		Returns:
			(list): A list of payoffs for each player.

		Note: Must be implemented in the child class.
		'''
		raise NotImplementedError

		
 

class DefaultBridgePayoffDelegate(BridgePayoffDelegate):

	def __init__(self, verbose):
		self.make_bid_bonus = 2
		self.verbose = verbose

	def get_payoffs(self, game: BridgeGame):
		''' 
		Get the payoffs of each players.

		Returns:
			(list): A list of payoffs for each player.
		'''
		# If currently in 1. Bidding phase
		#				  2. or PlayCard phase
		contract_bid_move = game.round.contract_bid_move
		if contract_bid_move:
			# Declarer - the player who places the final / highest bid	  .... e.g. p0
			# Dummy	   - Teammate of the declarer ............................ e.g. p2
			# Defender - players of the other team of declarer ............... e.g. p1, p3

			# i.e. all players = Bidding team (=Declarer + Dummy) + Defender team
			
			# Contract - the highest / final bid ........ i.e. contract of the PlayCard phase
			declarer = contract_bid_move.player
			
			# Bid amount	->	   No. of games to be won in contract
			
			'''
			# for Bridge
			bid_trick_count = contract_bid_move.action.bid_amount + 6
			'''
			
			# for Tarneeb
			'''
			Bid amount ∈ [7, 13]
			'''
			bid_trick_count = contract_bid_move.action.bid_amount
			
			# 1. Compute Total_wins of declarer and defender
			won_trick_counts = game.round.won_trick_counts
			
			declarer_won_trick_count = won_trick_counts[declarer.player_id % 2]
			defender_won_trick_count = won_trick_counts[(declarer.player_id + 1) % 2]

			
			# 2. Total_wins	  ->  Payoff 
			'''
			# for bridge
			if bid_trick_count <= declarer_won_trick_count:
				declarer_payoff = bid_trick_count + self.make_bid_bonus	 
			else:
				declarer_payoff = declarer_won_trick_count - bid_trick_count
			
			defender_payoff = defender_won_trick_count
			'''
			
			# for Tarneeb
			'''
			Payoff Rules:
				1. The bidder's team tries to take at least as many tricks as they bid. 
				   If their bid is less than 13 and succeed, they score the number of tricks they won, and the other team scores nothing. 
			
				2. If the bidding team takes fewer tricks than they bid, they lose the amount of their bid, and the other team scores the number of tricks they won.

				3. Winning all 13 tricks is called kaboot. If the bid was less than 13, kaboot brings a bonus of 3 points, so 16 points in total instead of 13.

				4. If a team bids 13 tricks and wins them all, they score 26 points. 
				   If they lose any tricks, they score minus 16 and the other team scores double the number the tricks that they win.

				5. Further hands are played until one team achieves a cumulative score of 41 points (=terminal score) or more, and wins the game.
				
				Ref: https://www.pagat.com/auctionwhist/tarneeb.html
			'''
			#import pdb; pdb.set_trace()
			if bid_trick_count < 13:
				# 1.
				if declarer_won_trick_count >= bid_trick_count:
					declarer_payoff = declarer_won_trick_count
					defender_payoff = 0
				# 2. 
				else:
					declarer_payoff = declarer_won_trick_count - bid_trick_count
					defender_payoff = defender_won_trick_count
				# 3.
				if declarer_won_trick_count == 13:
					declarer_payoff = 16   
			elif bid_trick_count == 13:
				# 4. 
				if declarer_won_trick_count == 13:
					declarer_payoff = 26
					defender_payoff = 0
				# 5.
				else:
					declarer_payoff = -16
					defender_payoff = 2*defender_won_trick_count
			else:
				raise Exception("Unexpected case")
				  
			
			# 3. Team_payoff ->	 Individual payoff
			team_wins = []
			payoffs = []
			for player_id in range(4):
				if player_id % 2 == declarer.player_id % 2:
					wins = declarer_won_trick_count
					payoff = declarer_payoff  
				else:
					wins = defender_won_trick_count
					payoff = defender_payoff
				payoffs.append(payoff)
				team_wins.append(wins)
		else:
			team_wins = [0, 0, 0, 0]
			payoffs = [0, 0, 0, 0]
		if self.verbose:
			print('\nWinning Trick Count:')
			print('*		Player 1:', team_wins[0])
			print('*		Player 2:', team_wins[1])
			print('*		Player 3:', team_wins[2])
			print('*		Player 4:', team_wins[3])

			print('\nPayoffs:')
			print('*		Player 1:', payoffs[0])
			print('*		Player 2:', payoffs[1])
			print('*		Player 3:', payoffs[2])
			print('*		Player 4:', payoffs[3])
			print()
		return np.array(payoffs)




########################### Helper Functions / 2. Extract State #################################	 

class BridgeStateExtractor(object):	 # interface

	def get_state_shape_size(self) -> int:
		raise NotImplementedError

	def extract_state(self, game: BridgeGame):
		''' Extract useful information from state for RL. Must be implemented in the child class.

		Args:
			game (BridgeGame): The game

		Returns:
			(numpy.array): The extracted state
		'''
		raise NotImplementedError

	@staticmethod
	def get_legal_actions(game: BridgeGame):
		''' Get all legal actions for current state.

		Returns:
			(OrderedDict): A OrderedDict of legal actions' id.
		'''
		legal_actions = game.judger.get_legal_actions()
		legal_actions_ids = {action_event.action_id: None for action_event in legal_actions}
		return OrderedDict(legal_actions_ids)


class DefaultBridgeStateExtractor(BridgeStateExtractor):

	def __init__(self):
		super().__init__()
		#self.max_bidding_rep_index = 40  # Note: max of 40 calls
		#self.last_bid_rep_size = 1 + 35 + 3  # no_bid, bid, pass, dbl, rdbl
		self.last_bid_rep_size = (1 +  7*5)

	def get_state_shape_size(self) -> int:
		state_shape_size = 0
		
		# 1) Cards representation
		# 1.1 Cards / In hand / Current player
		state_shape_size += 4 * 52						   # 1.	 hands_rep_size			: Rep of hand of  each player = [ [1, 0, 1,	 ...... 0],
														   #															  [0, 1, 0,	 ...... 0]
														   #															  [0, 0, 1,	 ...... 0]
														   #															  [0, 0, 0,	 ...... 1] ]

		# 1.2 Cards / In pile
		state_shape_size += 4 * 52						   # 2.	 trick_rep_size			: Rep of trick pile (i.e. the cards being currently played) = 
														   #													   e.g. [ [1, 0, 0,	 ...... 0],
														   #															  [0, 1, 0,	 ...... 0]
														   #															  [0, 0, 1,	 ...... 0]
														   #															  [0, 0, 0,	 ...... 1] ]
		
		# 1.3 Cards / In hand / Other players
		state_shape_size += 52							   # 3.	 hidden_cards_rep_size	 : Rep of cards played of other players (other then current player)
														   #														e.g. [1, 0, 0,	.... 0]		(52 items)
		
		
		# (only for bridge)
		# state_shape_size += 4							   # 4.	 vul_rep_size
		
		# 2) Player representation
		# 2.1 Player / Highest bidder
		state_shape_size += 4							   # 5.	 dealer_rep_size		 : Rep of highest bidder finally
														   #														  e.g. [0, 0, 1, 0]		(i.e. p3)
		
		# 2.2 Player / Current player
		state_shape_size += 4							   # 6.	 current_player_rep_size : Rep of current player (who will play now)
														   #														  e.g. [0, 1, 0, 0]		(i.e. p2)

		# 3) Bidding Phase representation
		# 3.1 Bidding phase
		state_shape_size += 1							   # 7.	 is_bidding_rep_size	 : Rep of bidding phase ongoing
														   #															e.g. 1				   (i.e. yes bidding phase)
														   #															e.g. 0				   (i.e. no bidding phase / yes playcard phase)
		# 3.2 Bid amount representation
		#state_shape_size += self.max_bidding_rep_index		# 8.  bidding_rep_size		   : Rep of bidding amount of one round for each player
		state_shape_size += 4*(1 +  7*5)  					#									   e.g. [p1_pass, p1_7_c, ... p1_13_c, 
		                                                    #                                                     p1_7_d, ... p1_13_d,
		                                                    #                                                     p1_7_s, ... p1_13_s,
		                                                    #                                                     p1_7_h, ... p1_13_h,
		                                                    #                                                     p1_7_nt, ... p1_13_nt,
															#											 (similarly for p2),
															#											 (similarly for p3),
															#											 (similarly for p4)]
		
		# 3.3 Last bid_amount representation
		#state_shape_size += self.last_bid_rep_size			# 9.  last_bid_rep_size			: Rep of last bid amount
		#state_shape_size += (1 +  7*5)      				#															   e.g. [px_pass, px_7_c, ... px_13_c,
		                     								#															                  px_7_d, ... px_13_d, 
		                     								#															                  px_7_s, ... px_13_s, 
		                     								#															                  px_7_h, ... px_13_h
		                     								#															                  px_7_nt, ... px_13_nt ]

		# 3.4 Final contract / bid_amount representation
		state_shape_size += 7							   # 10. bid_amount_rep_size	   : Rep of current bid amount
														   #															  e.g. [py_7, py_8, py_9, py_10, py_11, py_12, py_13]
		
		# 3.5 Final contract / Trump suite representation
		state_shape_size += 5							   # 11. trump_suit_rep_size	   : Rep of trump suite
														   #																	Spade, Diamond, Club, Hearts, No-Trump
														   #															   e.g. [0	  , 1	   , 0	 , 0	 , 0]
		
		# 3.6 Final contract / Highest bidder representation
		state_shape_size += 4
		
		return state_shape_size

	def extract_state(self, game: BridgeGame):
		''' Extract useful information from state for RL.

		Args:
			game (BridgeGame): The game

		Returns:
			(numpy.array): The extracted state
		'''
		#import pdb; pdb.set_trace()
		extracted_state = {}
		legal_actions: OrderedDict = self.get_legal_actions(game=game)
		raw_legal_actions = list(legal_actions.keys())
		current_player = game.round.get_current_player()
		current_player_id = current_player.player_id

		# 1) Cards representation
		# 1.1 Cards / In hand / Current player
		'''
		# If Bridge
		hands_rep = [np.zeros(52, dtype=int) for _ in range(4)]
		if not game.is_over():
			# A. For current player: 
			#						 Extract from "game.round.players"	->	 hands_rep
			for card in game.round.players[current_player_id].hand:
				hands_rep[current_player_id][card.card_id] = 1
			
			# B. For dummy player: 
			#						 Extract from "game.round.get_dummy()"	->	 State representation
			if game.round.is_bidding_over():
				dummy = game.round.get_dummy()
				other_known_player = dummy if dummy.player_id != current_player_id else game.round.get_declarer()
				for card in other_known_player.hand:
					hands_rep[other_known_player.player_id][card.card_id] = 1
		'''
		
		# If Tarneeb
		hands_rep = [np.zeros(52, dtype=int) for _ in range(4)]
		if not game.is_over():
			# A. For current player: 
			#						 Extract from "game.round.players"	->	 hands_rep
			for card in game.round.players[current_player_id].hand:
				hands_rep[current_player_id][card.card_id] = 1
		#hands_rep = np.concatenate(hands_rep)


		# 1.2 Cards / In pile										 
		#					   Core:		 
		#							Reformat: "game.round.get_trick_moves()"   ->	State representation
		trick_pile_rep = [np.zeros(52, dtype=int) for _ in range(4)]
		# If: Phase = PlayCard 
		if game.round.is_bidding_over() and not game.is_over():
			
			# Get past moves / card played	(in ongoing round)
			trick_moves = game.round.get_trick_moves()
			
			# Reformat: "game.round.get_trick_moves()"	 ->	  State representation
			for move in trick_moves:
				player = move.player
				card = move.card
				trick_pile_rep[player.player_id][card.card_id] = 1
		#trick_pile_rep = np.concatenate(trick_pile_rep)

		# 1.3 Cards / In hand / Other player
		# construct hidden_card_rep (during trick taking phase)
		hidden_cards_rep = np.zeros(52, dtype=int)
		if not game.is_over():
			# Case #1: Phase = PlayCard phase
			if game.round.is_bidding_over():
				
				# if Bridge:
				'''
				declarer = game.round.get_declarer()
				if current_player_id % 2 == declarer.player_id % 2:
					hidden_player_ids = [(current_player_id + 1) % 2, (current_player_id + 3) % 2]
				else:
					hidden_player_ids = [declarer.player_id, (current_player_id + 2) % 2]
				for hidden_player_id in hidden_player_ids:
					for card in game.round.players[hidden_player_id].hand:
						hidden_cards_rep[card.card_id] = 1
				'''
				
				# if Tarneeb:
				for player in game.round.players:
					if player.player_id != current_player_id:
						for card in player.hand:
							hidden_cards_rep[card.card_id] = 1
			# Case #2: Phase = Bidding phase
			else:
				# hidden cards = card of all other players
				# e.g. Current plaayers = p2
				#	   Other players	= [p1, p3, p4]
				for player in game.round.players:
					if player.player_id != current_player_id:
						for card in player.hand:
							hidden_cards_rep[card.card_id] = 1

		# construct vul_rep
		# for bridge:
		# vul_rep = np.array(game.round.tray.vul, dtype=int)

		
		# 2) Player representation
		# 2.1 Player / Dealer Rep
		dealer_rep = np.zeros(4, dtype=int)
		# If Bridge:
		'''
		dealer_rep[game.round.tray.dealer_id] = 1										 # Reformat:  "game.round.tray.dealer_id"	->	 State representation
		'''
		# If Tarneeb:
		#import pdb; pdb.set_trace()
		dealer_rep[game.round.tray.dealer_id] = 1
		#dealer_rep[game.round.get_declarer()] = 1

		
		# 2.2 Player / Current player
		current_player_rep = np.zeros(4, dtype=int)
		current_player_rep[current_player_id] = 1										 # Reformat:  "current_player_id"	->	 State representation


		# 3) Bidding Phase representation
		# 3.1 Bidding phase
		is_bidding_over_rep = np.array([1] if game.round.is_bidding_over() else [0])			  # Reformat:  "game.round.is_bidding_over()"	->	 State representation

		
		# 3.2 Bid amount representation
		#bidding_rep = np.zeros(self.max_bidding_rep_index, dtype=int)
		#bidding_rep = np.zeros(4*8, dtype=int)
		bidding_rep = [np.zeros(1+7*5, dtype=int) for _ in range(4)]
		#bidding_rep_index = game.round.dealer_id  # no_bid_action_ids allocated at start so that north always 'starts' the bidding
		# If bridge:
		'''
		for move in game.round.move_sheet:
			#if bidding_rep_index >= self.max_bidding_rep_index:
			#	 break
			if isinstance(move, PlayCardMove):
				break
			elif isinstance(move, CallMove):
				bidding_rep[bidding_rep_index] = move.action.action_id			# check: move.action.action_id ?
				# bidding_rep_index += 1
		'''
		
		# If Tarneeb:
		# Highest bid till now
		# Last 3 bids till now
		if not game.round.is_bidding_over():
			if len(game.round.move_sheet):
				# this_round_moves = len(game.round.move_sheet) % 4
				#import pdb; pdb.set_trace()
				# for move in game.round.move_sheet[-this_round_moves:]:
				for move in game.round.move_sheet[-3:]:
					if isinstance(move, CallMove):
						#import pdb; pdb.set_trace()
						bidding_rep[move.player.player_id][move.action.action_id] = 1
						#bidding_rep[current_player_id] = move.action.action_id			 # check: move.action.action_id ?
						# bidding_rep_index += 1
						#break
		#bidding_rep = np.concatenate(bidding_rep)
		
		# 3.3 Last bid_amount representation
		#last_bid_rep = np.zeros(self.last_bid_rep_size, dtype=int)
		#last_move = game.round.move_sheet[-1]
		#if isinstance(last_move, CallMove):
		#	last_bid_rep[last_move.action.action_id-1] = 1			   # check: ActionEvent.no_bid_action_id ?

		
		# 3.4 Final contract / bid_amount representation		  : Reformat "contract_bid_move.action.bid_amount"	 ->	  State representation
		# & 
		# 3.5 Final contract / Trump suite representation		  : Reformat "contract_bid_move.action.bid_suit"	 ->	  State representation
		bid_amount_rep = np.zeros(7, dtype=int)
		trump_suit_rep = np.zeros(5, dtype=int)
		highest_bidder_rep = np.zeros(4, dtype=int)
		#if game.round.is_bidding_over() and (not game.is_over()) and (game.round.play_card_count == 0):
		if game.round.is_bidding_over() and (not game.is_over()):
			contract_bid_move = game.round.contract_bid_move
			#print('contract_bid_move:', contract_bid_move)
			#import pdb; pdb.set_trace()
			if contract_bid_move:
				#import pdb; pdb.set_trace()
				bid_amount_rep[contract_bid_move.action.bid_amount-7] = 1
				
				bid_suit = contract_bid_move.action.bid_suit
				try:
					bid_suit_index = 4 if not bid_suit else BridgeCard.suits.index(bid_suit)    # 4 meaning No-Trump
				except:
					import pdb; pdb.set_trace()
				trump_suit_rep[bid_suit_index] = 1
				#import pdb; pdb.set_trace()
				highest_bidder_rep[game.round.highest_bidder] = 1

		# Concatenate individual state list	 ->	 one master list 
		rep = []
		rep += hands_rep
		rep += trick_pile_rep
		rep.append(hidden_cards_rep)
		#'''
		rep.append(dealer_rep)
		rep.append(current_player_rep)
		rep.append(is_bidding_over_rep)
		rep += bidding_rep
		#rep.append(last_bid_rep)
		#'''
		rep.append(bid_amount_rep)
		rep.append(trump_suit_rep)
		rep.append(highest_bidder_rep)

		#rep = [np.concatenate(item) for item in rep]
		obs = np.concatenate(rep)
		#import pdb; pdb.set_trace()
		
		extracted_state['obs'] = obs
		extracted_state['legal_actions'] = legal_actions
		extracted_state['raw_legal_actions'] = raw_legal_actions
		extracted_state['raw_obs'] = obs
		
		
		return extracted_state
