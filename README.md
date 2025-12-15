# Poker AI: Multimodal Decision Making with Reinforcement Learning

Advanced AI agent for Texas Hold'em poker combining game state and dialogue information using both supervised learning and reinforcement learning (PPO).

## Overview

This project implements and compares **4 different poker AI models**:

1. **Supervised Baseline**: Game state only (MLP)
2. **Supervised Multimodal**: Game state + dialogue (Transformer embeddings)
3. **RL Baseline (PPO)**: Game state only with profit optimization
4. **RL Multimodal (PPO)**: Game state + dialogue with profit optimization

### Key Results

Evaluation against rule-based agent (1000 games):

| Model                 | Total Profit (BB) | Avg/Hand   | Win Rate  |
| --------------------- | ----------------- | ---------- | --------- |
| Supervised Baseline   | -1,858.0          | -1.86      | 40.0%     |
| Supervised Multimodal | +33,974.5         | +33.97     | 46.4%     |
| **RL Baseline**       | +23,826.5         | +23.83     | 52.6%     |
| **RL Multimodal** 🏆  | **+76,669.0**     | **+76.67** | **52.6%** |

**Key Findings:**

- ✅ **Dialogue improves performance** in both learning paradigms
- ✅ **Reinforcement learning outperforms supervised learning** for profit maximization
- ✅ **Best combination**: RL + Multimodal (3.2× better than RL Baseline)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. HuggingFace Token Setup (Optional)

For using gated models like Llama:

**Option A: Environment Variable (Recommended)**

```bash
export HF_TOKEN="your_huggingface_token"
```

**Option B: Login via CLI**

```bash
huggingface-cli login
# Enter your token when prompted
```

**Option C: Set in Config**
Edit `3_generate_dialogues.py`:

```python
CONFIG = {
    'hf_token': 'your_token_here',
    ...
}
```

**Getting a Token:**

1. Go to https://huggingface.co/settings/tokens
2. Create a new token (Read permission is enough)
3. For Llama models, request access at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

**Using Open Models (No Token Required):**
The default config uses `microsoft/Phi-3-mini-4k-instruct` which requires no authentication.

## Project Structure

```
poker_project/
├── data/
│   ├── raw/              # Pluribus dataset cache
│   ├── processed/        # Processed features/labels
│   └── text/             # Generated dialogues
├── src/
│   ├── dataset.py        # Data loading and preprocessing
│   ├── models.py         # Model architectures (Supervised + RL)
│   ├── train.py          # Training loops
│   ├── evaluate.py       # Evaluation utilities
│   └── generate_text.py  # Dialogue generation
├── checkpoints/          # Saved model checkpoints
├── outputs/              # Evaluation reports and plots
├── 1_preprocess_data.py          # Step 1: Data preprocessing
├── 2_train_baseline.py           # Step 2: Supervised baseline
├── 3_generate_dialogues.py       # Step 3: Dialogue generation
├── 4_train_multimodal.py         # Step 4: Supervised multimodal
├── 5_train_rl_baseline.py        # Step 5: RL baseline (PPO)
├── 6_train_rl_multimodal.py      # Step 6: RL multimodal (PPO)
├── 7_evaluate_vs_rule_based.py   # Step 7: Evaluate all models
├── run_all.ipynb         # Jupyter notebook pipeline
└── requirements.txt
```

## Quick Start

### Option 1: Jupyter Notebook (Recommended)

```bash
jupyter notebook run_all.ipynb
```

Interactive notebook with step-by-step execution and visualization.

### Option 2: Run Individual Steps

#### Step 1: Preprocess Data

```bash
python 1_preprocess_data.py
```

Downloads Pluribus dataset and extracts 377-dim features.

#### Step 2: Train Supervised Baseline

```bash
python 2_train_baseline.py
```

Trains MLP on game state features only (supervised learning).

#### Step 3: Generate Dialogues

```bash
python 3_generate_dialogues.py
```

Generates poker dialogues using LLM (vLLM or rule-based).

#### Step 4: Train Supervised Multimodal

```bash
python 4_train_multimodal.py
```

Trains model combining game state + text embeddings (supervised learning).

#### Step 5: Train RL Baseline (PPO)

```bash
python 5_train_rl_baseline.py
```

Trains simple Actor-Critic using PPO on game state only.

#### Step 6: Train RL Multimodal (PPO)

```bash
python 6_train_rl_multimodal.py
```

Trains Actor-Critic using PPO on game state + dialogue.

#### Step 7: Evaluate All Models

```bash
python 7_evaluate_vs_rule_based.py --model_type rl_multimodal --model_path checkpoints/rl_multimodal_best.pt --n_games 1000
```

Evaluates models against rule-based agent.

## Configuration

Each script has a `CONFIG` dict at the top for easy customization:

```python
CONFIG = {
    'batch_size': 64,
    'learning_rate': 0.0005,
    'n_epochs': 10,
    ...
}
```

## Model Architectures

### Supervised Learning Models

#### Baseline (Game Only)

```
Input (377) -> PCA (256) -> [2048, 1024] -> ReLU + Dropout -> Output (6)
```

#### Multimodal (Game + Text)

```
Game (377) -> PCA (256) -> [2048, 1024] -> [512]
                                                   -> Concat [768] -> [384, 192] -> Output (6)
Text -> DistilBERT -> PCA (256)
```

### Reinforcement Learning Models (PPO)

#### RL Baseline (Game Only)

```
Input (256) -> Shared [256] -> ReLU + Dropout
                                    ├─> Actor (Policy): [6 actions]
                                    └─> Critic (Value): [1 value]
```

**Simple architecture to prevent overfitting**

#### RL Multimodal (Game + Text)

```
Game (256) -> Encoder [128]
                              -> Concat [256] -> Fusion [256]
Text (256) -> Encoder [128]                          ├─> Actor: [6 actions]
                                                     └─> Critic: [1 value]
```

**Key Features:**

- PPO (Proximal Policy Optimization) with GAE (Generalized Advantage Estimation)
- Reward function: Profit maximization + expert action matching
- Action space clipping and entropy regularization for stable training

## Output Files

### Checkpoints

- `checkpoints/baseline_best.pt` - Best supervised baseline model
- `checkpoints/multimodal_best.pt` - Best supervised multimodal model
- `checkpoints/rl_baseline_best.pt` - Best RL baseline model
- `checkpoints/rl_multimodal_best.pt` - Best RL multimodal model
- `checkpoints/*_pca.pkl` - PCA models for feature reduction

### Evaluation Results

- `outputs/baseline_vs_rule_based.json` - Supervised baseline results
- `outputs/multimodal_vs_rule_based.json` - Supervised multimodal results
- `outputs/rl_baseline_vs_rule_based.json` - RL baseline results
- `outputs/rl_multimodal_vs_rule_based.json` - RL multimodal results
- `outputs/all_models_comparison.png` - Comparative visualization

### Training History

- `outputs/*_training_history.png` - Supervised learning curves
- `outputs/rl_*_training.png` - RL training curves (rewards, profits, losses)
- `outputs/*_confusion_matrix.png` - Confusion matrices (supervised only)

## Evaluation Metrics

### Supervised Learning

- **Accuracy**: Overall classification accuracy
- **Macro F1**: Unweighted average F1 across classes
- **Weighted F1**: Support-weighted average F1
- **Per-class Precision/Recall/F1**: Detailed per-action metrics

### Reinforcement Learning

- **Total Profit**: Sum of big blinds won/lost over all games
- **Average Profit per Hand**: Efficiency metric
- **Win Rate**: Percentage of hands won
- **Expert Match Rate**: How often RL policy matches expert actions (imitation component)

## Action Classes

0. Fold
1. Check/Call
2. Raise Small (< 0.5× pot)
3. Raise Medium (0.5-1.5× pot)
4. Raise Large (> 1.5× pot)
5. All-in

## GPU Requirements

- **Supervised Baseline Training**: ~4GB VRAM
- **Supervised Multimodal Training**: ~8GB VRAM
- **RL Baseline Training**: ~4GB VRAM
- **RL Multimodal Training**: ~10GB VRAM
- **Dialogue Generation (vLLM)**: ~12-16GB VRAM
  - For 3B parameter model in FP16

Tested on:

- NVIDIA A10G 24GB GPU (AWS)
- NVIDIA RTX 3090 24GB (Local)

CPU-only mode supported but significantly slower.

## Notes

- Data is cached after first download
- Dialogue generation can use rule-based fallback if LLM unavailable
- All scripts support resuming from checkpoints
- Class imbalance handled via weighted loss (supervised) or reward shaping (RL)
- RL training uses PPO with careful stability measures:
  - Gradient clipping
  - Advantage normalization
  - Action probability clipping
  - NaN detection and recovery

## Key Hyperparameters

### Supervised Learning

- Learning rate: 0.0002
- Batch size: 256
- Epochs: 30
- Dropout: 0.2

### Reinforcement Learning (PPO)

- Learning rate: 0.0001
- Episodes: 1000
- PPO epochs: 4
- Clip epsilon: 0.2
- GAE lambda: 0.95
- Max gradient norm: 1.0

## Future Work

- [ ] Self-play training between RL agents
- [ ] Multi-agent poker environments
- [ ] Advanced dialogue generation (GPT-4, Claude)
- [ ] Real-time inference optimization
- [ ] Integration with poker simulators

## Citation

Dataset: Pluribus Poker Hand Histories
https://github.com/uoftcprg/phh-dataset
