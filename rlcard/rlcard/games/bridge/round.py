'''
	File name: bridge/round.py
	Author: William Hale
	Date created: 11/25/2021
'''

from typing import List

from .dealer import BridgeDealer
from .player import BridgePlayer

from .utils.action_event import CallActionEvent, PassAction, BidAction, PlayCardAction
from .utils.move import BridgeMove, DealHandMove, PlayCardMove, MakeBidMove, MakePassMove, CallMove
from .utils.tray import Tray


'''
Handle a round
1. __init__() = Init player for round

[Main]
2. playCard() = current player plays a card	  ->   if he's the last, then compute the winner too
2. make_call() = 

[Info extract]
3. get_current_player(), get_trick_moves() = Info functions for round

[History]
4. round.move_sheet: History of all move - in one Fateiyeh
'''


class BridgeRound:

	@property
	def dealer_id(self) -> int:
		#print('Dealer ID / ', self.tray.dealer_id)
		return self.tray.dealer_id

	# If Bridge
	'''
	@property
	def vul(self):
		return self.tray.vul
	'''

	@property
	def board_id(self) -> int:
		return self.tray.board_id

	@property
	def round_phase(self):
		if self.is_over():
			result = 'game over'
		elif self.is_bidding_over():
			result = 'play card'
		else:
			result = 'make bid'
		return result

	def __init__(self, num_players: int, board_id: int, np_random, verbose):
		''' Initialize the round class

			The round class maintains the following instances:
				1) dealer: the dealer of the round; dealer has trick_pile
				2) players: the players in the round; each player has his own hand_pile
				3) current_player_id: the id of the current player who has the move
				4) doubling_cube: 2 if contract is doubled; 4 if contract is redoubled; else 1
				5) play_card_count: count of PlayCardMoves
				5) move_sheet: history of the moves of the players (including the deal_hand_move)

			The round class maintains a list of moves made by the players in self.move_sheet.
			move_sheet is similar to a chess score sheet.
			I didn't want to call it a score_sheet since it is not keeping score.
			I could have called move_sheet just moves, but that might conflict with the name moves used elsewhere.
			I settled on the longer name "move_sheet" to indicate that it is the official list of moves being made.

		Args:
			num_players: int
			board_id: int
			np_random
		'''
		self.verbose = verbose
		
		# 1. Board init
		tray = Tray(board_id=board_id)
		# 2. Dealer init
		# If Bridge:
		dealer_id = tray.dealer_id
		if self.verbose:
			print('Dealer ID:', dealer_id+1)
		self.tray = tray
		self.np_random = np_random
		# If Bridge:
		self.dealer: BridgeDealer = BridgeDealer(self.np_random, self.verbose)
		'''
		To begin setup, a dealer must be chosen, this is done randomly. The dealer will shuffle the cards and deal 13 cards to each player one at a time. 
		Bidding will begin first with the player to the right of the dealer and continues counterclockwise around the group.
		'''

		# 3. Player init
		self.players: List[BridgePlayer] = []
		for player_id in range(num_players):
			self.players.append(BridgePlayer(player_id=player_id, np_random=self.np_random))
		self.current_player_id: int = (dealer_id + 1)%4

		# If Bridge:
		#self.doubling_cube: int = 1
		self.play_card_count: int = 0

		self.contract_bid_move: MakeBidMove or None = None
		self.won_trick_counts = [0, 0]	# count of won tricks by side
		
		# 4. Init move sheet
		self.move_sheet: List[BridgeMove] = []
		#import pdb; pdb.set_trace()
		self.move_sheet.append(DealHandMove(dealer=self.players[dealer_id], shuffled_deck=self.dealer.shuffled_deck))
		self.highest_bidder = None

	def is_bidding_over(self) -> bool:
		''' Return whether the current bidding is over
		'''
		#import pdb; pdb.set_trace()
		is_bidding_over = True
		if len(self.move_sheet) < 5:
			is_bidding_over = False
		else:
			last_make_pass_moves: List[MakePassMove] = []
			for move in reversed(self.move_sheet):
				if isinstance(move, MakePassMove):
					last_make_pass_moves.append(move)
					if len(last_make_pass_moves) == 3:
						break
				elif isinstance(move, CallMove):
					is_bidding_over = False
					break
				else:
					break
		return is_bidding_over

	def is_over(self) -> bool:
		''' 
		# 2
		[Get] Return whether the current game is over
		e.g. True / Falses
		Steps: 1. Is bidding over
			   2. 
		'''
		is_over = True
		if not self.is_bidding_over():
			is_over = False
		elif self.contract_bid_move:
			for player in self.players:
				if player.hand:
					is_over = False
					break
		return is_over

	def get_current_player(self) -> BridgePlayer or None:
		"""
		# 2
		[Get] Name of current player
		e.g. input = 1
			 output = p1
		e.g. input = 2
			 output = p2
		"""
		current_player_id = self.current_player_id
		return None if current_player_id is None else self.players[current_player_id]

	def get_trick_moves(self) -> List[PlayCardMove]:
		"""
		# 2
		[Get] Return previous moves of round
		e.g. if currently turn of p2
			 then rerturn = [move_of_p0, move_of_p1]
		"""
		trick_moves: List[PlayCardMove] = []
		if self.is_bidding_over():
			if self.play_card_count > 0:
				trick_pile_count = self.play_card_count % 4
				if trick_pile_count == 0:
					trick_pile_count = 4  # wch: note this
				for move in self.move_sheet[-trick_pile_count:]:
					if isinstance(move, PlayCardMove):
						trick_moves.append(move)
				if len(trick_moves) != trick_pile_count:
					raise Exception(f'get_trick_moves: count of trick_moves={[str(move.card) for move in trick_moves]} does not equal {trick_pile_count}')
		return trick_moves

	def get_trump_suit(self) -> str or None:
		"""
		# 2
		[Get] Return trump of the round
		e.g. output = "D"  (where D denotes Dimond)
		"""
		trump_suit = None
		if self.contract_bid_move:
			trump_suit = self.contract_bid_move.action.bid_suit
		return trump_suit

	def make_call(self, action: CallActionEvent):
		"""
		# 1
		[Put] Perform bid card action
		output = None
		"""
		# when current_player takes CallActionEvent step, the move is recorded and executed
		#import pdb; pdb.set_trace()
		current_player = self.players[self.current_player_id]
		if isinstance(action, PassAction):
			#import pdb; pdb.set_trace()
			current_player.has_passed_bid = True
			self.players[self.current_player_id] = current_player
			
			self.move_sheet.append(MakePassMove(current_player))
		elif isinstance(action, BidAction):
			#self.doubling_cube = 1
			make_bid_move = MakeBidMove(current_player, action)
			self.contract_bid_move = make_bid_move
			self.highest_bidder = self.current_player_id
			self.move_sheet.append(make_bid_move)

		#if self.is_bidding_over():
		if not self.is_over():
			#self.current_player_id = self.get_left_defender().player_id
			self.current_player_id = (self.current_player_id + 1) % 4
	
	def play_card(self, action: PlayCardAction):
		"""
		# 1
		[Put] Perform play card action
		output = None
		"""
		# when current_player takes PlayCardAction step, the move is recorded and executed
		current_player = self.players[self.current_player_id]
		self.move_sheet.append(PlayCardMove(current_player, action))
		card = action.card
		#print('   Curret Player / ', self.current_player_id)
		#print('   		Card / ', card)
		current_player.remove_card_from_hand(card=card)
		self.play_card_count += 1
		# update current_player_id
		trick_moves = self.get_trick_moves()
		
		# if round completed or not
		if len(trick_moves) == 4:
			trump_suit = self.get_trump_suit()
			winning_card = trick_moves[0].card
			trick_winner = trick_moves[0].player
			for move in trick_moves[1:]:
				trick_card = move.card
				trick_player = move.player
				if trick_card.suit == winning_card.suit:
					if trick_card.card_id > winning_card.card_id:
						winning_card = trick_card
						trick_winner = trick_player
				elif trick_card.suit == trump_suit:
					winning_card = trick_card
					trick_winner = trick_player
			self.current_player_id = trick_winner.player_id
			self.won_trick_counts[trick_winner.player_id % 2] += 1
			if self.verbose:
				print('\n ************************************** ROUND END (Winner: Player {})***********************************'.format(self.current_player_id+1))
		else:
			self.current_player_id = (self.current_player_id + 1) % 4

	def get_declarer(self) -> BridgePlayer or None:
		"""
		# 2
		[Get] Return dealer name
		e.g. output = "p2"
		"""
		declarer = None
		if self.contract_bid_move:
			trump_suit = self.contract_bid_move.action.bid_suit
			side = self.contract_bid_move.player.player_id % 2
			for move in self.move_sheet:
				if isinstance(move, MakeBidMove) and move.action.bid_suit == trump_suit and move.player.player_id % 2 == side:
					declarer = move.player
					break
		return declarer

	# If Bridge
	'''
	def get_dummy(self) -> BridgePlayer or None:
		"""
		# 2
		Return team member of dealer
		e.g. if dealer = p1
			  then dummy = p3
		"""
		dummy = None
		declarer = self.get_declarer()
		if declarer:
			dummy = self.players[(declarer.player_id + 2) % 4]
		return dummy
	'''

	def get_left_defender(self) -> BridgePlayer or None:
		"""
		# 2
		[Get] Return left of dealer
		e.g. if dealer = p1
			  then dummy = p2
		"""
		left_defender = None
		declarer = self.get_declarer()
		if declarer:
			left_defender = self.players[(declarer.player_id + 1) % 4]
		return left_defender

	def get_right_defender(self) -> BridgePlayer or None:
		"""
		# 2
		[Get] Return right of dealer
		e.g. if dealer = p1
			  then dummy = p0
		"""
		right_defender = None
		declarer = self.get_declarer()
		if declarer:
			right_defender = self.players[(declarer.player_id + 3) % 4]
		return right_defender

	def get_perfect_information(self):
		"""
		# 3
		[Get]
		"""
		state = {}
		last_call_move = None
		if not self.is_bidding_over() or self.play_card_count == 0:
			last_move = self.move_sheet[-1]
			if isinstance(last_move, CallMove):
				last_call_move = last_move
		trick_moves = [None, None, None, None]
		if self.is_bidding_over():
			for trick_move in self.get_trick_moves():
				trick_moves[trick_move.player.player_id] = trick_move.card
		state['move_count'] = len(self.move_sheet)
		state['tray'] = self.tray
		state['current_player_id'] = self.current_player_id
		state['round_phase'] = self.round_phase
		state['last_call_move'] = last_call_move
		state['doubling_cube'] = self.doubling_cube
		state['contact'] = self.contract_bid_move if self.is_bidding_over() and self.contract_bid_move else None
		state['hands'] = [player.hand for player in self.players]
		state['trick_moves'] = trick_moves
		return state

	def print_scene(self):
		"""
		# 3
		[Get]
		"""
		print(f'===== Board: {self.tray.board_id} move: {len(self.move_sheet)} player: {self.players[self.current_player_id]} phase: {self.round_phase} =====')
		print(f'dealer={self.players[self.tray.dealer_id]}')
		print(f'vul={self.vul}')
		if not self.is_bidding_over() or self.play_card_count == 0:
			last_move = self.move_sheet[-1]
			last_call_text = f'{last_move}' if isinstance(last_move, CallMove) else 'None'
			print(f'last call: {last_call_text}')
		if self.is_bidding_over() and self.contract_bid_move:
			bid_suit = self.contract_bid_move.action.bid_suit
			doubling_cube = self.doubling_cube
			if not bid_suit:
				bid_suit = 'NT'
			doubling_cube_text = "" if doubling_cube == 1 else "dbl" if doubling_cube == 2 else "rdbl"
			print(f'contract: {self.contract_bid_move.player} {self.contract_bid_move.action.bid_amount}{bid_suit} {doubling_cube_text}')
		for player in self.players:
			print(f'{player}: {[str(card) for card in player.hand]}')
		if self.is_bidding_over():
			trick_pile = ['None', 'None', 'None', 'None']
			for trick_move in self.get_trick_moves():
				trick_pile[trick_move.player.player_id] = trick_move.card
			print(f'trick_pile: {[str(card) for card in trick_pile]}')
