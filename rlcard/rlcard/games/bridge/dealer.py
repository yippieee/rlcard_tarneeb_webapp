'''
	File name: bridge/dealer.py
	Author: William Hale
	Date created: 11/25/2021
'''

from typing import List

from .player import BridgePlayer
from .utils.bridge_card import BridgeCard


'''
Re-assignment of old cards:

1. Random extract n cards from stock file
2. and give it a particular player
'''

class BridgeDealer:
	''' Initialize a BridgeDealer dealer class
	'''
	def __init__(self, np_random, verbose=False):
		''' set shuffled_deck, set stock_pile
		'''
		self.np_random = np_random
		self.verbose = verbose
		# 1. Create a deck
		self.shuffled_deck: List[BridgeCard] = BridgeCard.get_deck()  # keep a copy of the shuffled cards at start of new hand
			
		# 2. Shuffle the deck
		self.np_random.shuffle(self.shuffled_deck)
		
		# 3. Shuffled deck is stock pile
		self.stock_pile: List[BridgeCard] = self.shuffled_deck.copy()

	def deal_cards(self, player: BridgePlayer, num: int):
		''' Deal some cards from stock_pile to one player

		Args:
			player (BridgePlayer): The BridgePlayer object
			num (int): The number of cards to be dealt
		'''
		
		# Give 3 cards from stock pile to a particular player
		for idx, _ in enumerate(range(num)):
			player.hand.append(self.stock_pile.pop())
		if self.verbose:
			print('\nPlayer{}: {}'.format(player.player_id+1, player.hand))