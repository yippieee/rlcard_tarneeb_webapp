'''
	File name: bridge/utils/action_event.py
	Author: William Hale
	Date created: 11/25/2021
'''

from .bridge_card import BridgeCard

# Case: If Bridge
# ====================================
# Action_ids:
#		0 -> no_bid_action_id
#		1 to 35 -> bid_action_id (bid amount by suit or NT)
#		36 -> pass_action_id
#		37 -> dbl_action_id
#		38 -> rdbl_action_id
#		39 to 90 -> play_card_action_id
# ====================================

# Case: If Tarneeb
# ====================================
# Action_ids:
#		0 -> no_bid / pass action_id
#		1 to 28 -> bid_action_id (bid amount by suit)	= 4*7	   =  (assuming BridgeCard.suits = ['c', 'd', 's', 'h'])
#		29 to 80 -> play_card_action_id					= 28 + 52-1
# ====================================

'''
Action_ID (as number) to Action_identifier (as object)
e.g.  3		->	  <object> PlayCardAction(card=card)
'''


# Interface: for all actions
class ActionEvent(object):	# Interface

	#no_bid_action_id = 
	'''
	first_bid_action_id = 1
	last_bid_action_id = (7*5)                    #
	pass_action_id = 36
	first_play_card_action_id = 37
	last_play_card_action_id = 37 + 52 - 1
	'''
	first_bid_action_id = 0
	last_bid_action_id = (7*5) - 1
	pass_action_id = 35
	first_play_card_action_id = 36
	last_play_card_action_id = 36 + 52 - 1

	def __init__(self, action_id: int):
		self.action_id = action_id

	def __eq__(self, other):
		# Comp
		result = False
		if isinstance(other, ActionEvent):
			result = self.action_id == other.action_id
		return result

	@staticmethod
	def from_action_id(action_id: int):
		# Action_id -> Action_object
		# e.g. Input = 3
		#	   Output = <object> PlayCardAction(card=card)
		
		#import pd; pdb.set_trace()
		#print ('ActionID : ', action_id)
		# If action_id == 36
		if action_id == ActionEvent.pass_action_id:
			return PassAction()
		
		# If 1 <= action_id <= 35
		elif ActionEvent.first_bid_action_id <= action_id <= ActionEvent.last_bid_action_id:
			#bid_amount = 7 + (action_id - 1) % 7
			#bid_suit_id = (action_id - 1) // 7

			bid_amount = 7 + (action_id) % 7
			bid_suit_id = (action_id) // 7
			bid_suit = BridgeCard.suits[bid_suit_id] if bid_suit_id < 4 else "nt"               # [c, d, s, h, nt]
			return BidAction(bid_amount, bid_suit)
		
		# If 37 <= action_id <= 88
		elif ActionEvent.first_play_card_action_id <= action_id <= ActionEvent.last_play_card_action_id:
			card_id = action_id - ActionEvent.first_play_card_action_id
			card = BridgeCard.card(card_id=card_id)
			return PlayCardAction(card=card)
		else:
			raise Exception(f'ActionEvent from_action_id: invalid action_id={action_id}')

	@staticmethod
	def get_num_actions():
		''' Return the total number of possible actions in the game
		'''
		#return 88  # 1 + 35 + 3 + 52	# no_bid, 35 bids, pass, dbl, rdl, 52 play_card
		return 88  # 1 + 35 + 3 + 52	# no_bid, 35 bids, pass, dbl, rdl, 52 play_card


class CallActionEvent(ActionEvent):	 # Interface
	pass

# Action_identifier: 1.1 Bid phase / Pass
class PassAction(CallActionEvent):

	def __init__(self):
		super().__init__(action_id=ActionEvent.pass_action_id)

	def __str__(self):
		return "pass"

	def __repr__(self):
		return "pass"


# Action_identifier: 1.2 Bid phase / Place bid
class BidAction(CallActionEvent):

	def __init__(self, bid_amount: int, bid_suit: str or None):
		suits = BridgeCard.suits
		#if bid_suit and bid_suit not in suits:
		#	raise Exception(f'BidAction has invalid suit: {bid_suit}')
		if bid_suit in suits:
			bid_suit_id = suits.index(bid_suit)
		else:
			bid_suit_id = 4
		#if Bridge:
		'''
		bid_action_id = bid_suit_id + 5 * (bid_amount - 1) + ActionEvent.first_bid_action_id
		'''
		
		#if Tarneeb:
		#bid_action_id = bid_suit_id*7 + (bid_amount - 7 + 1)
		bid_action_id = bid_suit_id*7 + (bid_amount - 7)
		super().__init__(action_id = bid_action_id)
		self.bid_amount = bid_amount
		self.bid_suit = bid_suit

	def __str__(self):
		bid_suit = self.bid_suit
		if not bid_suit:
			bid_suit = 'NT'
		return f'{self.bid_amount}{bid_suit}'

	def __repr__(self):
		return self.__str__()


# Action_identifier: 2. Play Card Phase
class PlayCardAction(ActionEvent):

	def __init__(self, card: BridgeCard):
		play_card_action_id = ActionEvent.first_play_card_action_id + card.card_id
		super().__init__(action_id=play_card_action_id)
		self.card: BridgeCard = card

	def __str__(self):
		return f"{self.card}"

	def __repr__(self):
		return f"{self.card}"
