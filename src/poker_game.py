"""
Full Texas Hold'em Poker Game Engine
Implements proper poker rules, hand evaluation, and game flow
"""
import numpy as np
from enum import IntEnum
from collections import Counter
from itertools import combinations


class Rank(IntEnum):
    """Card ranks (2-A)"""
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


class Suit(IntEnum):
    """Card suits"""
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


class HandRank(IntEnum):
    """Poker hand rankings"""
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9


class Card:
    """Represents a playing card"""

    RANK_NAMES = {
        2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
        9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'
    }

    SUIT_SYMBOLS = {
        Suit.CLUBS: '♣',
        Suit.DIAMONDS: '♦',
        Suit.HEARTS: '♥',
        Suit.SPADES: '♠'
    }

    def __init__(self, rank, suit):
        """
        Args:
            rank: Card rank (2-14)
            suit: Card suit (0-3)
        """
        self.rank = rank
        self.suit = suit

    def __str__(self):
        return f"{self.RANK_NAMES[self.rank]}{self.SUIT_SYMBOLS[self.suit]}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))


class Deck:
    """Standard 52-card deck"""

    def __init__(self, seed=None):
        """Initialize and shuffle deck"""
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        """Reset and shuffle deck"""
        self.cards = [
            Card(rank, suit)
            for rank in range(2, 15)
            for suit in range(4)
        ]
        self.rng.shuffle(self.cards)

    def deal(self, n=1):
        """Deal n cards from deck"""
        if len(self.cards) < n:
            raise ValueError("Not enough cards in deck")
        dealt = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt if n > 1 else dealt[0]


class HandEvaluator:
    """Evaluates poker hands"""

    @staticmethod
    def evaluate_hand(cards):
        """
        Evaluate a poker hand (5-7 cards)

        Args:
            cards: List of Card objects (5-7 cards)

        Returns:
            (hand_rank, tiebreakers): Tuple of HandRank and list of tiebreaker values
        """
        if len(cards) < 5:
            raise ValueError("Need at least 5 cards to evaluate")

        # Find best 5-card combination
        best_rank = HandRank.HIGH_CARD
        best_tiebreakers = []

        for combo in combinations(cards, 5):
            rank, tiebreakers = HandEvaluator._evaluate_5_cards(list(combo))
            if (rank, tiebreakers) > (best_rank, best_tiebreakers):
                best_rank = rank
                best_tiebreakers = tiebreakers

        return best_rank, best_tiebreakers

    @staticmethod
    def _evaluate_5_cards(cards):
        """Evaluate exactly 5 cards"""
        ranks = sorted([c.rank for c in cards], reverse=True)
        suits = [c.suit for c in cards]

        is_flush = len(set(suits)) == 1
        is_straight, straight_high = HandEvaluator._is_straight(ranks)

        # Count rank occurrences
        rank_counts = Counter(ranks)
        counts = sorted(rank_counts.values(), reverse=True)
        unique_ranks = sorted(rank_counts.keys(), key=lambda r: (rank_counts[r], r), reverse=True)

        # Royal Flush
        if is_flush and is_straight and straight_high == 14:
            return HandRank.ROYAL_FLUSH, [14]

        # Straight Flush
        if is_flush and is_straight:
            return HandRank.STRAIGHT_FLUSH, [straight_high]

        # Four of a Kind
        if counts == [4, 1]:
            quad = [r for r, c in rank_counts.items() if c == 4][0]
            kicker = [r for r, c in rank_counts.items() if c == 1][0]
            return HandRank.FOUR_OF_A_KIND, [quad, kicker]

        # Full House
        if counts == [3, 2]:
            trips = [r for r, c in rank_counts.items() if c == 3][0]
            pair = [r for r, c in rank_counts.items() if c == 2][0]
            return HandRank.FULL_HOUSE, [trips, pair]

        # Flush
        if is_flush:
            return HandRank.FLUSH, ranks

        # Straight
        if is_straight:
            return HandRank.STRAIGHT, [straight_high]

        # Three of a Kind
        if counts == [3, 1, 1]:
            trips = [r for r, c in rank_counts.items() if c == 3][0]
            kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
            return HandRank.THREE_OF_A_KIND, [trips] + kickers

        # Two Pair
        if counts == [2, 2, 1]:
            pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
            kicker = [r for r, c in rank_counts.items() if c == 1][0]
            return HandRank.TWO_PAIR, pairs + [kicker]

        # One Pair
        if counts == [2, 1, 1, 1]:
            pair = [r for r, c in rank_counts.items() if c == 2][0]
            kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
            return HandRank.PAIR, [pair] + kickers

        # High Card
        return HandRank.HIGH_CARD, ranks

    @staticmethod
    def _is_straight(ranks):
        """Check if ranks form a straight"""
        sorted_ranks = sorted(set(ranks), reverse=True)

        # Check for regular straight
        if len(sorted_ranks) == 5:
            if sorted_ranks[0] - sorted_ranks[4] == 4:
                return True, sorted_ranks[0]

        # Check for A-2-3-4-5 (wheel)
        if sorted_ranks == [14, 5, 4, 3, 2]:
            return True, 5

        return False, 0

    @staticmethod
    def hand_description(rank, tiebreakers):
        """Get human-readable hand description"""
        rank_names = {
            2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six',
            7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten',
            11: 'Jack', 12: 'Queen', 13: 'King', 14: 'Ace'
        }

        if rank == HandRank.ROYAL_FLUSH:
            return "Royal Flush"
        elif rank == HandRank.STRAIGHT_FLUSH:
            return f"Straight Flush, {rank_names[tiebreakers[0]]} high"
        elif rank == HandRank.FOUR_OF_A_KIND:
            return f"Four of a Kind, {rank_names[tiebreakers[0]]}s"
        elif rank == HandRank.FULL_HOUSE:
            return f"Full House, {rank_names[tiebreakers[0]]}s over {rank_names[tiebreakers[1]]}s"
        elif rank == HandRank.FLUSH:
            return f"Flush, {rank_names[tiebreakers[0]]} high"
        elif rank == HandRank.STRAIGHT:
            return f"Straight, {rank_names[tiebreakers[0]]} high"
        elif rank == HandRank.THREE_OF_A_KIND:
            return f"Three of a Kind, {rank_names[tiebreakers[0]]}s"
        elif rank == HandRank.TWO_PAIR:
            return f"Two Pair, {rank_names[tiebreakers[0]]}s and {rank_names[tiebreakers[1]]}s"
        elif rank == HandRank.PAIR:
            return f"Pair of {rank_names[tiebreakers[0]]}s"
        else:
            return f"High Card, {rank_names[tiebreakers[0]]}"


class PokerGame:
    """
    Heads-up Texas Hold'em game
    """

    def __init__(self, small_blind=1, big_blind=2, starting_stack=100, seed=None):
        """
        Args:
            small_blind: Small blind amount
            big_blind: Big blind amount
            starting_stack: Starting stack for both players
            seed: Random seed
        """
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.starting_stack = starting_stack

        self.deck = Deck(seed=seed)
        self.rng = np.random.RandomState(seed)

        # Game state
        self.player_stack = starting_stack
        self.ai_stack = starting_stack
        self.player_hole = []
        self.ai_hole = []
        self.board = []
        self.pot = 0
        self.street = 'preflop'
        self.player_bet = 0
        self.ai_bet = 0
        self.button = 'player'  # 'player' or 'ai'
        self.to_act = None
        self.hand_over = False
        self.folded = None
        self.last_action = None  # Track last action to detect consecutive checks

    def start_hand(self):
        """Start a new hand"""
        # Reset deck
        self.deck.reset()

        # Reset hand state
        self.board = []
        self.pot = 0
        self.player_bet = 0
        self.ai_bet = 0
        self.hand_over = False
        self.folded = None
        self.street = 'preflop'
        self.last_action = None

        # Rotate button
        self.button = 'ai' if self.button == 'player' else 'player'

        # Deal hole cards
        self.player_hole = self.deck.deal(2)
        self.ai_hole = self.deck.deal(2)

        # Post blinds
        if self.button == 'player':
            # Player is button/small blind
            self.player_stack -= self.small_blind
            self.player_bet = self.small_blind
            self.ai_stack -= self.big_blind
            self.ai_bet = self.big_blind
            self.to_act = 'player'
        else:
            # AI is button/small blind
            self.ai_stack -= self.small_blind
            self.ai_bet = self.small_blind
            self.player_stack -= self.big_blind
            self.player_bet = self.big_blind
            self.to_act = 'ai'

        self.pot = self.player_bet + self.ai_bet

        return {
            'street': self.street,
            'player_hole': self.player_hole,
            'board': self.board,
            'pot': self.pot,
            'player_stack': self.player_stack,
            'ai_stack': self.ai_stack,
            'player_bet': self.player_bet,
            'ai_bet': self.ai_bet,
            'to_act': self.to_act
        }

    def get_valid_actions(self):
        """Get list of valid actions for current player"""
        if self.hand_over:
            return []

        to_call = abs(self.player_bet - self.ai_bet)
        acting_stack = self.player_stack if self.to_act == 'player' else self.ai_stack

        actions = []

        # Can always fold if there's a bet to call
        if to_call > 0:
            actions.append('fold')

        # Check if no bet, call if there's a bet
        if to_call == 0:
            actions.append('check')
        elif to_call <= acting_stack:
            actions.append('call')

        # Can raise if have chips
        if acting_stack > to_call:
            # Minimum raise is the size of the previous bet/raise
            min_raise = max(self.big_blind, to_call * 2)
            if min_raise <= acting_stack:
                actions.append('raise')

        # Can always go all-in if have chips
        if acting_stack > 0:
            actions.append('all_in')

        return actions

    def act(self, action, raise_amount=None):
        """
        Execute an action

        Args:
            action: 'fold', 'check', 'call', 'raise', 'all_in'
            raise_amount: Amount to raise (if action is 'raise')

        Returns:
            Game state dict
        """
        if action not in self.get_valid_actions():
            raise ValueError(f"Invalid action: {action}")

        to_call = abs(self.player_bet - self.ai_bet)

        if self.to_act == 'player':
            if action == 'fold':
                self.folded = 'player'
                self.hand_over = True
                self.ai_stack += self.pot
                return self._get_state()

            elif action == 'check':
                # If both players check consecutively, move to next street
                if self.player_bet == self.ai_bet and self.last_action == 'check':
                    self._next_street()
                else:
                    # Switch to other player's turn
                    self.to_act = 'ai'
                    self.last_action = 'check'
                return self._get_state()

            elif action == 'call':
                amount = min(to_call, self.player_stack)
                self.player_stack -= amount
                self.player_bet += amount
                self.pot += amount
                self.last_action = 'call'

                # Move to next street
                if self.player_bet == self.ai_bet:
                    self._next_street()
                else:
                    self.to_act = 'ai'
                return self._get_state()

            elif action == 'raise':
                if raise_amount is None or raise_amount < self.big_blind:
                    raise_amount = self.big_blind * 3

                amount = min(to_call + raise_amount, self.player_stack)
                self.player_stack -= amount
                self.player_bet += amount
                self.pot += amount
                self.to_act = 'ai'
                self.last_action = 'raise'
                return self._get_state()

            elif action == 'all_in':
                amount = self.player_stack
                self.player_stack = 0
                self.player_bet += amount
                self.pot += amount
                self.to_act = 'ai'
                self.last_action = 'all_in'
                return self._get_state()

        else:  # AI's turn
            if action == 'fold':
                self.folded = 'ai'
                self.hand_over = True
                self.player_stack += self.pot
                return self._get_state()

            elif action == 'check':
                # If both players check consecutively, move to next street
                if self.player_bet == self.ai_bet and self.last_action == 'check':
                    self._next_street()
                else:
                    # Switch to other player's turn
                    self.to_act = 'player'
                    self.last_action = 'check'
                return self._get_state()

            elif action == 'call':
                amount = min(to_call, self.ai_stack)
                self.ai_stack -= amount
                self.ai_bet += amount
                self.pot += amount
                self.last_action = 'call'

                if self.player_bet == self.ai_bet:
                    self._next_street()
                else:
                    self.to_act = 'player'
                return self._get_state()

            elif action == 'raise':
                if raise_amount is None or raise_amount < self.big_blind:
                    raise_amount = self.big_blind * 3

                amount = min(to_call + raise_amount, self.ai_stack)
                self.ai_stack -= amount
                self.ai_bet += amount
                self.pot += amount
                self.to_act = 'player'
                self.last_action = 'raise'
                return self._get_state()

            elif action == 'all_in':
                amount = self.ai_stack
                self.ai_stack = 0
                self.ai_bet += amount
                self.pot += amount
                self.to_act = 'player'
                self.last_action = 'all_in'
                return self._get_state()

    def _next_street(self):
        """Advance to next street"""
        # Reset bets for new street
        self.player_bet = 0
        self.ai_bet = 0
        self.last_action = None

        if self.street == 'preflop':
            # Deal flop
            self.board.extend(self.deck.deal(3))
            self.street = 'flop'
            self.to_act = 'ai' if self.button == 'player' else 'player'

        elif self.street == 'flop':
            # Deal turn
            self.board.append(self.deck.deal())
            self.street = 'turn'
            self.to_act = 'ai' if self.button == 'player' else 'player'

        elif self.street == 'turn':
            # Deal river
            self.board.append(self.deck.deal())
            self.street = 'river'
            self.to_act = 'ai' if self.button == 'player' else 'player'

        elif self.street == 'river':
            # Showdown
            self._showdown()

    def _showdown(self):
        """Determine winner at showdown"""
        self.hand_over = True

        # Evaluate hands
        player_hand = self.player_hole + self.board
        ai_hand = self.ai_hole + self.board

        player_rank, player_tiebreakers = HandEvaluator.evaluate_hand(player_hand)
        ai_rank, ai_tiebreakers = HandEvaluator.evaluate_hand(ai_hand)

        # Compare hands
        if (player_rank, player_tiebreakers) > (ai_rank, ai_tiebreakers):
            self.player_stack += self.pot
        elif (player_rank, player_tiebreakers) < (ai_rank, ai_tiebreakers):
            self.ai_stack += self.pot
        else:
            # Split pot
            self.player_stack += self.pot // 2
            self.ai_stack += self.pot // 2

    def _get_state(self):
        """Get current game state"""
        return {
            'street': self.street,
            'player_hole': self.player_hole,
            'ai_hole': self.ai_hole if self.hand_over else None,
            'board': self.board,
            'pot': self.pot,
            'player_stack': self.player_stack,
            'ai_stack': self.ai_stack,
            'player_bet': self.player_bet,
            'ai_bet': self.ai_bet,
            'to_act': self.to_act,
            'hand_over': self.hand_over,
            'folded': self.folded,
            'valid_actions': self.get_valid_actions()
        }

    def get_winner(self):
        """Get hand winner (only valid after hand is over)"""
        if not self.hand_over:
            return None

        if self.folded:
            return 'ai' if self.folded == 'player' else 'player'

        player_hand = self.player_hole + self.board
        ai_hand = self.ai_hole + self.board

        player_rank, player_tiebreakers = HandEvaluator.evaluate_hand(player_hand)
        ai_rank, ai_tiebreakers = HandEvaluator.evaluate_hand(ai_hand)

        if (player_rank, player_tiebreakers) > (ai_rank, ai_tiebreakers):
            return 'player'
        elif (player_rank, player_tiebreakers) < (ai_rank, ai_tiebreakers):
            return 'ai'
        else:
            return 'tie'

    def get_hand_description(self, player='player'):
        """Get hand description for player"""
        if player == 'player':
            cards = self.player_hole + self.board
        else:
            cards = self.ai_hole + self.board

        if len(cards) < 5:
            return "Not enough cards"

        rank, tiebreakers = HandEvaluator.evaluate_hand(cards)
        return HandEvaluator.hand_description(rank, tiebreakers)
