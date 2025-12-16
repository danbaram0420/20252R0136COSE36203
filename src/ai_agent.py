"""
AI Agent for playing poker
Loads trained models and makes decisions
"""
import torch
import numpy as np
import pickle
from pathlib import Path
from models import MultimodalActorCritic, SimpleActorCritic, PokerMLP, MultimodalPokerModel
from generate_text import generate_dialogue_template
from transformers import AutoTokenizer, AutoModel


class PokerAIAgent:
    """AI agent that uses trained model to play poker"""

    def __init__(
        self,
        model_path='checkpoints/rl_multimodal_best.pt',
        model_type='rl_multimodal',
        device='cuda',
        use_dialogue=True
    ):
        """
        Args:
            model_path: Path to model checkpoint
            model_type: 'rl_multimodal', 'rl_baseline', 'supervised_multimodal', 'supervised_baseline'
            device: Device to run model on
            use_dialogue: Whether to use dialogue (for multimodal models)
        """
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_dialogue = use_dialogue and 'multimodal' in model_type
        self.is_rl = model_type.startswith('rl_')

        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Load PCA transformers
        if model_type in ['rl_multimodal', 'rl_baseline']:
            pca_path = 'checkpoints/rl_baseline_pca.pkl' if model_type == 'rl_baseline' else 'checkpoints/rl_multimodal_game_pca.pkl'
            with open(pca_path, 'rb') as f:
                self.game_pca = pickle.load(f)
            print(f"Loaded game PCA from {pca_path}")
        else:
            pca_path = 'checkpoints/baseline_pca.pkl' if model_type == 'supervised_baseline' else 'checkpoints/multimodal_game_pca.pkl'
            with open(pca_path, 'rb') as f:
                self.game_pca = pickle.load(f)
            print(f"Loaded game PCA from {pca_path}")

        # Load text components for multimodal models
        if self.use_dialogue:
            # Load text PCA
            text_pca_path = 'checkpoints/rl_multimodal_text_pca.pkl' if 'rl' in model_type else 'checkpoints/multimodal_text_pca.pkl'
            with open(text_pca_path, 'rb') as f:
                self.text_pca = pickle.load(f)
            print(f"Loaded text PCA from {text_pca_path}")

            # Load text encoder
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self.text_encoder = AutoModel.from_pretrained('distilbert-base-uncased').to(self.device)
            self.text_encoder.eval()
            print("Loaded text encoder (DistilBERT)")

        # Create model architecture
        if model_type == 'rl_multimodal':
            self.model = MultimodalActorCritic(
                game_input_dim=256,
                text_input_dim=256,
                hidden_dim=128,
                fusion_dim=256,
                n_actions=6,
                dropout=0.1
            ).to(self.device)
        elif model_type == 'rl_baseline':
            self.model = SimpleActorCritic(
                input_dim=256,
                hidden_dim=256,
                n_actions=6,
                dropout=0.1
            ).to(self.device)
        elif model_type == 'supervised_baseline':
            self.model = PokerMLP(
                input_dim=256,  # After PCA
                hidden_dims=[2048, 1024],
                output_dim=6,
                dropout=0.2
            ).to(self.device)
        elif model_type == 'supervised_multimodal':
            self.model = MultimodalPokerModel(
                game_input_dim=256,
                text_input_dim=256,
                game_hidden_dims=[2048, 1024],
                fusion_hidden_dims=[384, 192],
                output_dim=6,
                dropout=0.2
            ).to(self.device)
        else:
            raise ValueError(f"Model type {model_type} not supported. Use one of: rl_multimodal, rl_baseline, supervised_multimodal, supervised_baseline")

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Loaded {model_type} model")

        # Action mapping
        self.action_names = ['fold', 'call', 'raise', 'raise', 'raise', 'all_in']
        self.action_map = {
            0: 'fold',
            1: 'call',  # or check
            2: 'raise',  # small
            3: 'raise',  # medium
            4: 'raise',  # large
            5: 'all_in'
        }

    def game_state_to_features(self, game_state):
        """
        Convert game state to feature vector (377 dimensions)

        Args:
            game_state: Dict with keys 'player_hole', 'board', 'pot', 'player_stack', etc.

        Returns:
            features: np.ndarray of shape (377,)
        """
        features = np.zeros(377, dtype=np.float32)

        # Helper to encode cards
        def card_to_index(card):
            """Convert card to index (0-51)"""
            rank_idx = card.rank - 2  # 2-14 -> 0-12
            suit_idx = card.suit  # 0-3
            return rank_idx * 4 + suit_idx

        # Encode hole cards (104 dims - 2 cards × 52 one-hot)
        if 'ai_hole' in game_state and game_state['ai_hole']:
            hole_cards = game_state['ai_hole']
        else:
            hole_cards = []

        for i, card in enumerate(hole_cards[:2]):
            card_idx = card_to_index(card)
            features[i * 52 + card_idx] = 1.0

        # Encode board cards (260 dims - 5 cards × 52 one-hot)
        board_cards = game_state.get('board', [])
        for i, card in enumerate(board_cards[:5]):
            card_idx = card_to_index(card)
            features[104 + i * 52 + card_idx] = 1.0

        # Position (6 dims - one-hot for position 0-5, use 0 for heads-up)
        features[364] = 1.0

        # Street (4 dims - one-hot for preflop/flop/turn/river)
        street_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
        street_idx = street_map.get(game_state.get('street', 'preflop'), 0)
        features[370 + street_idx] = 1.0

        # Numeric features (3 dims - pot, stack, bet_to_call)
        pot = game_state.get('pot', 0)
        stack = game_state.get('ai_stack', 100)
        ai_bet = game_state.get('ai_bet', 0)
        player_bet = game_state.get('player_bet', 0)
        bet_to_call = max(0, player_bet - ai_bet)

        # Normalize
        features[374] = min(pot / 100.0, 5.0)
        features[375] = min(stack / 100.0, 5.0)
        features[376] = min(bet_to_call / 100.0, 5.0)

        return features

    def generate_dialogue(self, game_state, action_idx):
        """Generate dialogue for the action"""
        if not self.use_dialogue:
            return ""

        # Use template-based dialogue
        dialogue = generate_dialogue_template(game_state, action_idx)
        return dialogue

    def embed_text(self, text):
        """Embed text using DistilBERT and PCA"""
        if not self.use_dialogue:
            return None

        # Tokenize
        encoded = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Encode
        with torch.no_grad():
            outputs = self.text_encoder(**encoded)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # Apply PCA
        embedding_pca = self.text_pca.transform(cls_embedding)

        return embedding_pca[0]

    def get_action(self, game_state, valid_actions, deterministic=True):
        """
        Get action for current game state

        Args:
            game_state: Dict with game state
            valid_actions: List of valid action strings from game engine
            deterministic: Whether to use deterministic policy

        Returns:
            action: Action string ('fold', 'check', 'call', 'raise', 'all_in')
            dialogue: Dialogue string (if multimodal)
        """
        # Convert game state to features
        features = self.game_state_to_features(game_state)

        # Apply PCA
        features_pca = self.game_pca.transform(features.reshape(1, -1))
        features_tensor = torch.FloatTensor(features_pca).to(self.device)

        # Get action from model
        with torch.no_grad():
            if self.is_rl:
                # RL models have get_action method
                if self.use_dialogue:
                    # Generate placeholder dialogue (will be replaced after action selection)
                    temp_dialogue = "Thinking..."
                    text_embedding = self.embed_text(temp_dialogue)
                    text_tensor = torch.FloatTensor(text_embedding).unsqueeze(0).to(self.device)

                    action_idx, _, _ = self.model.get_action(features_tensor, text_tensor, deterministic=deterministic)
                    action_idx = action_idx.item()
                else:
                    action_idx, _, _ = self.model.get_action(features_tensor, deterministic=deterministic)
                    action_idx = action_idx.item()
            else:
                # Supervised models output logits directly
                if self.use_dialogue:
                    # Generate placeholder dialogue (will be replaced after action selection)
                    temp_dialogue = "Thinking..."
                    text_embedding = self.embed_text(temp_dialogue)
                    text_tensor = torch.FloatTensor(text_embedding).unsqueeze(0).to(self.device)

                    logits = self.model(features_tensor, text_tensor)
                else:
                    logits = self.model(features_tensor)

                # Get action with highest probability
                if deterministic:
                    action_idx = torch.argmax(logits, dim=1).item()
                else:
                    # Sample from softmax distribution
                    probs = torch.softmax(logits, dim=1)
                    action_idx = torch.multinomial(probs, 1).item()

        # Map model action to game action
        model_action = self.action_map[action_idx]

        # Ensure action is valid
        if model_action == 'call' and 'check' in valid_actions and 'call' not in valid_actions:
            action = 'check'
        elif model_action not in valid_actions:
            # Fallback to valid action
            if 'check' in valid_actions:
                action = 'check'
            elif 'call' in valid_actions:
                action = 'call'
            elif 'fold' in valid_actions:
                action = 'fold'
            else:
                action = valid_actions[0]
        else:
            action = model_action

        # Generate dialogue after action selection
        dialogue = self.generate_dialogue(game_state, action_idx) if self.use_dialogue else ""

        return action, dialogue

    def get_raise_amount(self, game_state):
        """Get raise amount based on pot size"""
        pot = game_state.get('pot', 0)
        # Default to pot-sized raise
        return max(pot, game_state.get('player_bet', 0) - game_state.get('ai_bet', 0) + pot)
