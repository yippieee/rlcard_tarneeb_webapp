'''
    File name: bridge/utils/bridge_card.py
    Author: William Hale
    Date created: 11/25/2021
'''

from rlcard.games.base import Card

'''
Handle deck of cards:
'''

class BridgeCard(Card):

    suits = ['C', 'D', 'H', 'S']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

    @staticmethod
    def card(card_id: int):
        # Card ID -> Cand
        # e.g. Input=2 .... Output=s3
        return _deck[card_id]

    @staticmethod
    def get_deck() -> [Card]:
        # Return deck of cards
        # e.g. [Card_1, Card_2, Card_3, ............ (52 elements)]
        return _deck.copy()

    def __init__(self, suit: str, rank: str):
        # suit card -> suit index
        # e.g. input  = s
        #      output = 2               (assuming BridgeCard.suits = ['c', 'd', 's', 'h'])
        
        # card's value -> value index
        # e.g. input = A 
        #      output = 13              (assuming BridgeCard.ranks = [2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A])
        super().__init__(suit=suit, rank=rank)
        suit_index = BridgeCard.suits.index(self.suit)
        rank_index = BridgeCard.ranks.index(self.rank)
        self.card_id = 13 * suit_index + rank_index

    def __str__(self):
        return f'{self.rank}{self.suit}'

    def __repr__(self):
        return f'{self.rank}{self.suit}'


# deck is always in order from 2C, ... KC, AC, 2D, ... KD, AD, 2H, ... KH, AH, 2S, ... KS, AS
_deck = [BridgeCard(suit=suit, rank=rank) for suit in BridgeCard.suits for rank in BridgeCard.ranks]  # want this to be read-only
