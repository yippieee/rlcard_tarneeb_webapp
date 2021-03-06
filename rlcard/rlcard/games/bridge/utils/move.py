'''
    File name: bridge/utils/move.py
    Author: William Hale
    Date created: 11/25/2021
'''

# 
# These classes are used to keep a move_sheet history of the moves in a round.
#

from .action_event import ActionEvent, BidAction, PassAction, PlayCardAction
from .bridge_card import BridgeCard

from ..player import BridgePlayer



################### 1. Interface #############################################

class BridgeMove(object):         # Interface 1: Generic
    pass


class PlayerMove(BridgeMove):     # Interface 2.1: Phase: PlayCard phase

    def __init__(self, player: BridgePlayer, action: ActionEvent):
        super().__init__()
        self.player = player
        self.action = action

        
class CallMove(PlayerMove):        # Interface 2.2: Phase: Bidding phase

    def __init__(self, player: BridgePlayer, action: ActionEvent):
        super().__init__(player=player, action=action)



################# 2. Moves ####################################################
## 2.1 Move = ?
class DealHandMove(BridgeMove):

    def __init__(self, dealer: BridgePlayer, shuffled_deck: [BridgeCard]):
        super().__init__()
        self.dealer = dealer
        self.shuffled_deck = shuffled_deck

    def __str__(self):
        shuffled_deck_text = " ".join([str(card) for card in self.shuffled_deck])
        return f'{self.dealer} deal shuffled_deck=[{shuffled_deck_text}]'


## 2.2 Move = Bid phase / Pass
class MakePassMove(CallMove):

    def __init__(self, player: BridgePlayer):
        super().__init__(player=player, action=PassAction())

    def __str__(self):
        return f'{self.player} {self.action}'


## 2.3 Move = Bid phase / Make bid
class MakeBidMove(CallMove):

    def __init__(self, player: BridgePlayer, bid_action: BidAction):
        super().__init__(player=player, action=bid_action)
        self.action = bid_action  # Note: keep type as BidAction rather than ActionEvent

    def __str__(self):
        return f'{self.player} bids {self.action}'


# 2.4: Move = Play Card Phase
class PlayCardMove(PlayerMove):

    def __init__(self, player: BridgePlayer, action: PlayCardAction):
        super().__init__(player=player, action=action)
        self.action = action  # Note: keep type as PlayCardAction rather than ActionEvent

    @property
    def card(self):
        return self.action.card

    def __str__(self):
        return f'{self.player} plays {self.action}'
