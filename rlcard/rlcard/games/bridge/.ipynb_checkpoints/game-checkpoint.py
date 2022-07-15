'''
	File name: bridge/game.py
	Author: William Hale
	Date created: 11/25/2021
'''

from typing import List

import numpy as np

from .judger import BridgeJudger
from .round import BridgeRound
from .utils.action_event import ActionEvent, CallActionEvent, PlayCardAction
from .utils.move import MakePassMove


'''
# Tarneeb Strucute
								
						round_1	  
					 /			 
		  Fatiyeh_1	 -	round_2
		/			 \
	   /				round_3				  
	  /
Game  -	  Fatiyeh_2
	  \
	   \ 
		  Fatiyeh_3
		  
						 (there are 13 rounds (a.k.a tricks) / per Fatiyeh)
						 
		  (there are those no.
		   of Fatiyah untill one of the team
		   reaches terrminal score of 41
		   and is declared as the winner)
		  
(there is 1 game)
'''


'''
Entry point: 
	*init_game()  = Distributes 13 cards to each player
	
	*step()       = take next step
	
	(mainly calls round.py and maintains state)
'''

class BridgeGame:
	''' Game class. This class will interact with outer environment.
	'''

	def __init__(self, allow_step_back=False, verbose=False):
		'''Initialize the class BridgeGame
		'''
		#np.random.seed(7)
		self.verbose = verbose
		self.allow_step_back: bool = allow_step_back
		self.np_random = np.random.RandomState()
		self.judger: BridgeJudger = BridgeJudger(game=self)
		self.actions: [ActionEvent] = []  # must reset in init_game
		self.round: BridgeRound or None = None	# must reset in init_game
		self.num_players: int = 4
		self.board_id = None 

	def init_game(self):
		'''
		Initialize all characters in the game and start round 1
		'''
		if self.verbose:
			print('\n**************************************** Setup ****************************************')
		if self.board_id is None:
			self.board_id = self.np_random.choice([1, 2, 3, 4])
		self.actions: List[ActionEvent] = []
		self.round = BridgeRound(num_players=self.num_players, board_id=self.board_id, np_random=self.np_random, verbose=self.verbose)
		for player_id in range(4):
			player = self.round.players[player_id]
			self.round.dealer.deal_cards(player=player, num=13)
			#print('HAND / ', player_id, ':' , player.hand)
		#import pdb; pdb.set_trace()
		current_player_id = self.round.current_player_id
		state = self.get_state(player_id=current_player_id)
		if self.verbose:
			print('\nSeating Order:')
			print('```p1```')
			print('p2````p4')
			print('```p3```')
			print('\n**************************************** Game Start ****************************************')
		return state, current_player_id

	def step(self, action: ActionEvent):
		'''
		#1 
		[Put] Perform game action and return next player number, and the state for next player
		'''
		# Phase: Bidding phase
		if isinstance(action, CallActionEvent):
			#print('Step / Bid Phase :' , action)
			self.round.make_call(action=action)
			if self.is_4_consecutive_pass():
				if self.verbose:
					print('************************ Reset Game ***********************************')
				self.np_random = np.random.RandomState()
				return self.init_game()				
		# Phase: PlayCard phase
		elif isinstance(action, PlayCardAction):
			#print('Step / PlayCard Phase :' , action)
			self.round.play_card(action=action)
		else:
			raise Exception(f'Unknown step action={action}')
		self.actions.append(action)
		
		next_player_id = self.round.current_player_id
		next_state = self.get_state(player_id=next_player_id)
		return next_state, next_player_id

	def is_4_consecutive_pass(self):
		'''
		If all four players pass on their first turn to speak, the hand is thrown in, and the cards are shuffled and redealt by the same dealer.
		'''
		all_pass = False
		if (len(self.round.move_sheet)>=4):
			all_pass = all([True if isinstance(mv, MakePassMove) else False for mv in self.round.move_sheet[-4:]])
		return all_pass

	def get_num_players(self) -> int:
		''' 
		#2
		[Get] Return the total number of players in the game
		'''
		return self.num_players

	@staticmethod
	def get_num_actions() -> int:
		''' 
		#2
		[Get] Return the total number of possible actions in the game
		'''
		return ActionEvent.get_num_actions()

	def get_player_id(self):
		''' 
		#2
		[Get] Return the current player
		'''
		return self.round.current_player_id

	def is_over(self) -> bool:
		''' 
		#2
		[Get] Return whether the current game is over
		'''
		return self.round.is_over()

	def get_state(self, player_id: int):  # wch: not really used
		''' 
		#2
		[Get] Given a player, return its state

		Return:
			state (dict): The information of the state
		'''
		state = {}
		if not self.is_over():
			state['player_id'] = player_id
			state['current_player_id'] = self.round.current_player_id
			state['hand'] = self.round.players[player_id].hand
		else:
			state['player_id'] = player_id
			state['current_player_id'] = self.round.current_player_id
			state['hand'] = self.round.players[player_id].hand
		return state
