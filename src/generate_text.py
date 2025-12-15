"""
Text generation for poker dialogues using LLM
"""
import torch
import numpy as np
from tqdm import tqdm
import json
import os


# ============================================================================
# State Reconstruction
# ============================================================================

def features_to_state(feature_vector):
    """
    Convert 377-dim feature vector back to game state
    
    Feature breakdown:
    - [0:104]: Hole cards (52*2 one-hot)
    - [104:364]: Board cards (52*5 one-hot)
    - [364:370]: Position (6 one-hot)
    - [370:374]: Street (4 one-hot)
    - [374:377]: pot, stack, bet_to_call (normalized)
    
    Returns:
        state dict with readable fields
    """
    # Decode hole cards
    hole_cards = []
    for i in range(2):
        card_vec = feature_vector[i*52:(i+1)*52]
        card_idx = np.argmax(card_vec)
        if card_vec[card_idx] > 0:
            hole_cards.append(index_to_card(card_idx))
    
    # Decode board cards
    board_cards = []
    for i in range(5):
        card_vec = feature_vector[104 + i*52:104 + (i+1)*52]
        card_idx = np.argmax(card_vec)
        if card_vec[card_idx] > 0:
            board_cards.append(index_to_card(card_idx))
    
    # Decode position
    position_vec = feature_vector[364:370]
    position = np.argmax(position_vec)
    
    # Decode street
    street_vec = feature_vector[370:374]
    streets = ['preflop', 'flop', 'turn', 'river']
    street = streets[np.argmax(street_vec)]
    
    # Denormalize numeric features
    pot = int(feature_vector[374] * 10000)
    stack = int(feature_vector[375] * 10000)
    bet_to_call = int(feature_vector[376] * 10000)
    
    return {
        'hole_cards': hole_cards,
        'board_cards': board_cards,
        'position': int(position),
        'street': street,
        'pot': pot,
        'stack': stack,
        'bet_to_call': bet_to_call
    }


def index_to_card(idx):
    """Convert 0-51 index to card string (e.g., 'Tc', 'As')"""
    ranks = '23456789TJQKA'
    suits = 'cdhs'
    rank = ranks[idx // 4]
    suit = suits[idx % 4]
    return rank + suit


# ============================================================================
# Prompt Creation
# ============================================================================

def create_dialogue_prompt(state):
    """
    Create prompt from game state ONLY (no action information)
    
    Args:
        state: Game state dict from features_to_state()
        
    Returns:
        prompt: String prompt for LLM
    """
    # Format cards
    hole_str = ' '.join(state['hole_cards']) if state['hole_cards'] else 'unknown'
    board_str = ' '.join(state['board_cards']) if state['board_cards'] else 'none yet'
    
    # Position names
    position_names = ['BTN', 'SB', 'BB', 'UTG', 'MP', 'CO']
    position_str = position_names[state['position']] if state['position'] < 6 else f"P{state['position']}"
    
    prompt = f"""You're playing 6-max No-Limit Hold'em poker.

Position: {position_str}
Your cards: {hole_str}
Board: {board_str}
Street: {state['street']}

Pot: ${state['pot']}
Your stack: ${state['stack']}
Bet to call: ${state['bet_to_call']}

You're about to act. What would you say out loud at the table?
(Natural poker talk, under 10 words. Don't mention specific actions like fold/call/raise.)

Examples:
✓ "Not feeling this one."
✓ "Let's see what happens."
✓ "Time to make a move."
✓ "I like my hand here."

Say:"""
    
    return prompt


# ============================================================================
# Generation Functions
# ============================================================================

def generate_dialogues_hf(
    features,
    labels,
    model_name='microsoft/Phi-3-mini-4k-instruct',
    max_tokens=30,
    temperature=0.8,
    batch_size=16,
    output_file='data/text/dialogues.jsonl',
    use_cache=True
):
    """
    Generate dialogues using HuggingFace transformers
    
    Args:
        features: np.ndarray of shape (N, 377)
        labels: np.ndarray of shape (N,) - for saving only, NOT used in generation
        model_name: HuggingFace model name
        max_tokens: Max tokens per generation
        temperature: Sampling temperature
        batch_size: Batch size for generation
        output_file: Path to save generated dialogues
        use_cache: Whether to use cached results
        
    Returns:
        dialogues: List of generated dialogue strings
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Check cache
    if use_cache and os.path.exists(output_file):
        print(f"Loading cached dialogues from {output_file}")
        dialogues = []
        with open(output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                dialogues.append(data['dialogue'])
        print(f"✓ Loaded {len(dialogues)} cached dialogues")
        return dialogues
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print(f"Generating dialogues using {model_name}...")
    print(f"Total samples: {len(features)}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map='auto' if device == 'cuda' else None,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if device == 'cpu':
        model = model.to(device)
    
    dialogues = []
    
    # Generate in batches
    for i in tqdm(range(0, len(features), batch_size), desc='Generating'):
        batch_features = features[i:i+batch_size]
        
        # Reconstruct states from features
        states = [features_to_state(f) for f in batch_features]
        
        # Create prompts (NO action label used)
        prompts = [create_dialogue_prompt(state) for state in states]
        
        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        for j, output in enumerate(outputs):
            input_length = inputs['input_ids'][j].shape[0]
            generated_ids = output[input_length:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Clean
            dialogue = clean_dialogue(generated_text)
            dialogues.append(dialogue)
    
    # Save to file
    with open(output_file, 'w') as f:
        for i, dialogue in enumerate(dialogues):
            data = {
                'index': i,
                'dialogue': dialogue,
                'action': int(labels[i])  # Saved for reference, not used in generation
            }
            f.write(json.dumps(data) + '\n')
    
    print(f"✓ Generated {len(dialogues)} dialogues")
    print(f"✓ Saved to {output_file}")
    
    return dialogues


def clean_dialogue(text):
    """Clean generated dialogue"""
    text = text.strip()
    
    # Remove quotes
    text = text.replace('"', '').replace("'", '')
    
    # Take first sentence only
    if '.' in text:
        text = text.split('.')[0] + '.'
    if '\n' in text:
        text = text.split('\n')[0]
    
    # Remove meta-commentary
    bad_phrases = [
        'would say', 'might say', 'could say', 'player', 'given',
        'based on', 'according to', 'in this situation', 'here is'
    ]
    for phrase in bad_phrases:
        if phrase in text.lower():
            return fallback_dialogue()
    
    # Check length
    if len(text.split()) > 12 or len(text.split()) < 2:
        return fallback_dialogue()
    
    return text


def fallback_dialogue():
    """Simple fallback dialogues"""
    fallbacks = [
        "Let me think.",
        "Interesting spot.",
        "Okay then.",
        "Here we go.",
        "Let's play this."
    ]
    return np.random.choice(fallbacks)


# ============================================================================
# Rule-based Fallback (if LLM unavailable)
# ============================================================================

RULE_BASED_TEMPLATES = {
    0: ["Not for me.", "I'm out.", "Too much.", "Pass.", "Next hand."],
    1: ["Let's see.", "I'm in.", "Okay.", "Call.", "Sure."],
    2: ["Let me raise a bit.", "Bump it up.", "Making it interesting.", "A little more."],
    3: ["Time to play.", "I like this.", "Let's build it.", "Raising here."],
    4: ["Big bet.", "Let's play for real.", "This is my hand.", "Serious now."],
    5: ["Everything.", "All of it.", "Let's go.", "This is it.", "I'm committed."]
}

def generate_dialogues_rule_based(labels, output_file='data/text/dialogues.jsonl'):
    """
    Generate dialogues using rule-based templates
    
    Args:
        labels: np.ndarray of shape (N,)
        output_file: Path to save generated dialogues
        
    Returns:
        dialogues: List of generated dialogue strings
    """
    print(f"Generating dialogues using rule-based templates...")
    print(f"Total samples: {len(labels)}")
    
    dialogues = []
    rng = np.random.RandomState(42)
    
    for label in tqdm(labels, desc='Generating'):
        templates = RULE_BASED_TEMPLATES[label]
        dialogue = rng.choice(templates)
        dialogues.append(dialogue)
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for i, dialogue in enumerate(dialogues):
            data = {
                'index': i,
                'dialogue': dialogue,
                'action': int(labels[i])
            }
            f.write(json.dumps(data) + '\n')
    
    print(f"✓ Generated {len(dialogues)} dialogues")
    print(f"✓ Saved to {output_file}")
    
    return dialogues


# ============================================================================
# Loading Utilities
# ============================================================================

def load_dialogues(input_file='data/text/dialogues.jsonl'):
    """Load generated dialogues from file"""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Dialogue file not found: {input_file}")
    
    dialogues = []
    actions = []
    
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            dialogues.append(data['dialogue'])
            actions.append(data['action'])
    
    print(f"✓ Loaded {len(dialogues)} dialogues from {input_file}")
    return dialogues, np.array(actions)