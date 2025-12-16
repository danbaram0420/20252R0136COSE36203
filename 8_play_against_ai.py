"""
Step 8: Play Poker Against Trained AI
Interactive poker game where you play heads-up against the trained AI model
"""
import sys
sys.path.append('src')

import torch
from pathlib import Path
from poker_game import PokerGame, HandEvaluator
from ai_agent import PokerAIAgent
import time


def print_separator(char='=', length=60):
    """Print a separator line"""
    print(char * length)


def print_game_state(game, show_ai_cards=False):
    """
    Print current game state in a nice format

    Args:
        game: PokerGame instance
        show_ai_cards: Whether to show AI's hole cards
    """
    print_separator()
    print(f"STREET: {game.street.upper()}")
    print_separator()

    # Player info
    print(f"\n🧑 YOU:")
    print(f"  Hole: {game.player_hole[0]} {game.player_hole[1]}")
    print(f"  Stack: {game.player_stack} BB")
    print(f"  Bet: {game.player_bet} BB")

    # Board
    if game.board:
        print(f"\n🃏 BOARD: {' '.join(str(c) for c in game.board)}")
    else:
        print(f"\n🃏 BOARD: (no cards yet)")

    # AI info
    print(f"\n🤖 AI:")
    if show_ai_cards and game.ai_hole:
        print(f"  Hole: {game.ai_hole[0]} {game.ai_hole[1]}")
    else:
        print(f"  Hole: 🂠 🂠 (hidden)")
    print(f"  Stack: {game.ai_stack} BB")
    print(f"  Bet: {game.ai_bet} BB")

    # Pot
    print(f"\n💰 POT: {game.pot} BB")
    print_separator()


def get_player_action(game):
    """
    Get action from player

    Args:
        game: PokerGame instance

    Returns:
        (action, raise_amount): Tuple of action string and optional raise amount
    """
    valid_actions = game.get_valid_actions()

    print("\n🎯 Your turn!")
    print(f"Valid actions: {', '.join(valid_actions)}")

    while True:
        action = input("\nChoose your action: ").strip().lower()

        if action not in valid_actions:
            print(f"❌ Invalid action. Please choose from: {', '.join(valid_actions)}")
            continue

        raise_amount = None
        if action == 'raise':
            pot = game.pot
            to_call = abs(game.player_bet - game.ai_bet)
            min_raise = max(game.big_blind, to_call)

            print(f"\nCurrent pot: {pot} BB")
            print(f"Amount to call: {to_call} BB")
            print(f"Minimum raise: {min_raise} BB")
            print(f"Your stack: {game.player_stack} BB")

            while True:
                try:
                    raise_input = input(f"\nRaise amount (or press Enter for {pot} BB): ").strip()
                    if raise_input == '':
                        raise_amount = pot
                        break
                    raise_amount = int(raise_input)
                    if raise_amount < min_raise:
                        print(f"❌ Raise must be at least {min_raise} BB")
                        continue
                    if raise_amount + to_call > game.player_stack:
                        print(f"❌ You only have {game.player_stack} BB")
                        continue
                    break
                except ValueError:
                    print("❌ Please enter a valid number")
                    continue

        return action, raise_amount


def print_hand_result(game, ai_dialogue=None):
    """
    Print hand result

    Args:
        game: PokerGame instance
        ai_dialogue: AI's dialogue (if multimodal)
    """
    print_separator('=')
    print("HAND RESULT")
    print_separator('=')

    # Show AI cards
    print(f"\n🤖 AI reveals: {game.ai_hole[0]} {game.ai_hole[1]}")

    if game.folded:
        if game.folded == 'player':
            print(f"\n❌ You folded. AI wins {game.pot} BB")
        else:
            print(f"\n✅ AI folded. You win {game.pot} BB!")
    else:
        # Showdown
        if game.board:
            print(f"\n🃏 Final Board: {' '.join(str(c) for c in game.board)}")

        player_desc = game.get_hand_description('player')
        ai_desc = game.get_hand_description('ai')

        print(f"\n🧑 Your hand: {player_desc}")
        print(f"🤖 AI's hand: {ai_desc}")

        winner = game.get_winner()
        if winner == 'player':
            print(f"\n✅ You win {game.pot} BB!")
        elif winner == 'ai':
            print(f"\n❌ AI wins {game.pot} BB")
        else:
            print(f"\n🤝 Split pot! ({game.pot // 2} BB each)")

    if ai_dialogue:
        print(f"\n💬 AI says: \"{ai_dialogue}\"")

    print_separator('=')


def play_game(agent, starting_stack=100, n_hands=None):
    """
    Play poker against AI

    Args:
        agent: PokerAIAgent instance
        starting_stack: Starting stack for both players
        n_hands: Number of hands to play (None for unlimited)
    """
    game = PokerGame(
        small_blind=1,
        big_blind=2,
        starting_stack=starting_stack,
        seed=None  # Random seed
    )

    hand_count = 0
    player_wins = 0
    ai_wins = 0

    print_separator('=')
    print("WELCOME TO POKER AI CHALLENGE!")
    print_separator('=')
    print(f"\nYou're playing heads-up No-Limit Texas Hold'em")
    print(f"Starting stack: {starting_stack} BB each")
    print(f"Blinds: {game.small_blind}/{game.big_blind} BB")
    if n_hands:
        print(f"Number of hands: {n_hands}")
    else:
        print(f"Number of hands: Unlimited (until one player is out)")
    print("\nType 'quit' at any time to exit")
    print_separator('=')

    input("\nPress Enter to start...")

    while True:
        # Check if game should end
        if game.player_stack <= 0:
            print("\n💥 You're out of chips! AI wins the match!")
            break
        if game.ai_stack <= 0:
            print("\n🎉 AI is out of chips! You win the match!")
            break
        if n_hands and hand_count >= n_hands:
            print(f"\n⏰ {n_hands} hands completed!")
            break

        # Start new hand
        hand_count += 1
        print(f"\n\n{'='*60}")
        print(f"HAND #{hand_count}")
        print(f"Score - You: {player_wins} | AI: {ai_wins}")
        print(f"Stacks - You: {game.player_stack} BB | AI: {game.ai_stack} BB")
        print(f"{'='*60}\n")

        game.start_hand()

        # Print initial state
        print_game_state(game, show_ai_cards=False)

        ai_last_dialogue = None

        # Play hand
        while not game.hand_over:
            state = game._get_state()

            if game.to_act == 'player':
                # Player's turn
                action, raise_amount = get_player_action(game)

                if action == 'quit':
                    print("\n👋 Thanks for playing!")
                    return

                game.act(action, raise_amount)

            else:
                # AI's turn
                print("\n🤖 AI is thinking...")
                time.sleep(0.5)  # Dramatic pause

                state_for_ai = game._get_state()
                state_for_ai['ai_hole'] = game.ai_hole  # Give AI access to its own cards

                action, dialogue = agent.get_action(
                    state_for_ai,
                    game.get_valid_actions(),
                    deterministic=True
                )

                # Get raise amount if needed
                raise_amount = None
                if action == 'raise':
                    raise_amount = agent.get_raise_amount(state_for_ai)

                # Show AI action
                if dialogue:
                    print(f"💬 AI: \"{dialogue}\"")

                to_call = abs(game.player_bet - game.ai_bet)
                if action == 'fold':
                    print(f"🤖 AI folds")
                elif action == 'check':
                    print(f"🤖 AI checks")
                elif action == 'call':
                    print(f"🤖 AI calls {to_call} BB")
                elif action == 'raise':
                    print(f"🤖 AI raises to {raise_amount} BB")
                elif action == 'all_in':
                    print(f"🤖 AI goes all-in for {game.ai_stack} BB!")

                ai_last_dialogue = dialogue
                game.act(action, raise_amount)

                # Print state after AI action
                if not game.hand_over:
                    print_game_state(game, show_ai_cards=False)

        # Hand is over
        winner = game.get_winner()
        if winner == 'player':
            player_wins += 1
        elif winner == 'ai':
            ai_wins += 1

        print_hand_result(game, ai_last_dialogue)

        # Ask to continue
        if n_hands is None or hand_count < n_hands:
            response = input("\nPlay another hand? (y/n): ").strip().lower()
            if response != 'y':
                print("\n👋 Thanks for playing!")
                break

    # Final score
    print(f"\n\n{'='*60}")
    print("FINAL SCORE")
    print(f"{'='*60}")
    print(f"Hands played: {hand_count}")
    print(f"You won: {player_wins} hands ({player_wins/hand_count*100:.1f}%)")
    print(f"AI won: {ai_wins} hands ({ai_wins/hand_count*100:.1f}%)")
    print(f"Ties: {hand_count - player_wins - ai_wins} hands")
    print(f"\nFinal stacks:")
    print(f"  You: {game.player_stack} BB ({game.player_stack - starting_stack:+d} BB)")
    print(f"  AI: {game.ai_stack} BB ({game.ai_stack - starting_stack:+d} BB)")

    if game.player_stack > game.ai_stack:
        print(f"\n🎉 Congratulations! You won by {game.player_stack - game.ai_stack} BB!")
    elif game.ai_stack > game.player_stack:
        print(f"\n😔 AI won by {game.ai_stack - game.player_stack} BB. Better luck next time!")
    else:
        print(f"\n🤝 It's a tie!")

    print(f"{'='*60}")


def main():
    print("\n" + "="*60)
    print("STEP 8: PLAY POKER AGAINST TRAINED AI")
    print("="*60)

    # Define all available models
    models_info = {
        '1': {
            'name': 'RL Multimodal (Best)',
            'path': 'checkpoints/rl_multimodal_best.pt',
            'type': 'rl_multimodal',
            'dialogue': True,
            'description': 'PPO-trained with game state + dialogue (76,669 BB profit)'
        },
        '2': {
            'name': 'RL Baseline',
            'path': 'checkpoints/rl_baseline_best.pt',
            'type': 'rl_baseline',
            'dialogue': False,
            'description': 'PPO-trained with game state only (23,827 BB profit)'
        },
        '3': {
            'name': 'Supervised Multimodal',
            'path': 'checkpoints/multimodal_best.pt',
            'type': 'supervised_multimodal',
            'dialogue': True,
            'description': 'Supervised learning with game state + dialogue (33,975 BB profit)'
        },
        '4': {
            'name': 'Supervised Baseline',
            'path': 'checkpoints/baseline_best.pt',
            'type': 'supervised_baseline',
            'dialogue': False,
            'description': 'Supervised learning with game state only (-1,858 BB profit)'
        }
    }

    # Check which models are available
    available_models = {}
    print("\nChecking for trained models...")
    for key, info in models_info.items():
        if Path(info['path']).exists():
            available_models[key] = info
            print(f"  ✓ {info['name']} - Available")
        else:
            print(f"  ✗ {info['name']} - Not found")

    if not available_models:
        print("\n❌ Error: No trained models found!")
        print("\n   Please run steps 1-6 first to train models:")
        print("   1. python 1_preprocess_data.py")
        print("   2. python 2_train_baseline.py")
        print("   3. python 3_generate_dialogues.py")
        print("   4. python 4_train_multimodal.py")
        print("   5. python 5_train_rl_baseline.py")
        print("   6. python 6_train_rl_multimodal.py")
        return

    # Let user choose model
    print("\n" + "="*60)
    print("SELECT MODEL TO PLAY AGAINST")
    print("="*60)
    for key, info in available_models.items():
        print(f"\n{key}. {info['name']}")
        print(f"   {info['description']}")

    while True:
        choice = input("\nSelect model (1-4): ").strip()
        if choice in available_models:
            selected_model = available_models[choice]
            break
        else:
            print(f"❌ Invalid choice. Please select from: {', '.join(available_models.keys())}")

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Load AI agent
    print(f"\nLoading {selected_model['name']}...")
    try:
        agent = PokerAIAgent(
            model_path=selected_model['path'],
            model_type=selected_model['type'],
            device=device,
            use_dialogue=selected_model['dialogue']
        )
        print(f"✓ {selected_model['name']} loaded successfully!")
    except Exception as e:
        print(f"\n❌ Error loading AI agent: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease ensure all required files are present.")
        return

    # Game settings
    print("\n" + "="*60)
    print("GAME SETTINGS")
    print("="*60)

    while True:
        try:
            starting_stack = input("\nStarting stack (default 100 BB): ").strip()
            if starting_stack == '':
                starting_stack = 100
            else:
                starting_stack = int(starting_stack)
            if starting_stack <= 0:
                print("❌ Stack must be positive")
                continue
            break
        except ValueError:
            print("❌ Please enter a valid number")

    while True:
        try:
            n_hands = input("Number of hands (press Enter for unlimited): ").strip()
            if n_hands == '':
                n_hands = None
            else:
                n_hands = int(n_hands)
                if n_hands <= 0:
                    print("❌ Number of hands must be positive")
                    continue
            break
        except ValueError:
            print("❌ Please enter a valid number")

    # Start game
    try:
        play_game(agent, starting_stack=starting_stack, n_hands=n_hands)
    except KeyboardInterrupt:
        print("\n\n👋 Game interrupted. Thanks for playing!")
    except Exception as e:
        print(f"\n\n❌ Error during game: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
