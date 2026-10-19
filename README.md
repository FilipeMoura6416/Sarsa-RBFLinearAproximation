# SARSA Linear Function Approximation for Mountain Car

This project aims to implement a **SARSA (State-Action-Reward-State-Action) agent** with **linear function approximation** for solving the Mountain Car environment from Gymnasium. The agent uses various feature extraction methods to convert continuous state spaces into feature vectors suitable for linear Q-value approximation.

## 🎯 **Key Features**

- **Modular Architecture**: Clean separation with abstract base classes
- **Model Persistence**: Save and load trained models

## 🏗️ **Architecture**

### **Core Components**

1. **`sarsa_linear_agent.py`** - Main SARSA agent implementation
2. **`feature_extractors.py`** - Abstract and concrete feature extraction classes
4. **`example.py`** - Usage examples and demonstrations

### **Feature Extractors**

#### **Radial Basis Functions** (`RBFFeatureExtractor`) 
- Uses Gaussian-like RBF centers distributed across state space
- **Best for**: Localized learning, non-linear patterns
- **Parameters**: `n_centers` (default: 25), `sigma` (default: 0.1)

## 🚀 **Quick Start**

### **Basic Usage**

```bash
python example.py
```


## 📊 **Analysis & Comparison**


## 🛠️ **Dependencies**

```bash
pip install gymnasium numpy matplotlib pytest
```

## 📁 **File Structure**

```
linear_mountain_car/
├── sarsa_linear_agent.py      # Core SARSA agent
├── feature_extractors.py      # Feature extraction implementations  
├── example.py                 # Usage examples
├── README.md                  # This documentation
└── requirements.txt           # Dependencies
```

## 🎯 **Algorithm Details**

### **SARSA Update Rule**
```
Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
```

Where:
- **α**: Learning rate
- **γ**: Discount factor  
- **r**: Immediate reward
- **s,a**: Current state-action pair
- **s',a'**: Next state-action pair

### **Linear Function Approximation**
```
Q(s,a) = θ^T φ(s,a)
```

Where:
- **θ**: Weight vector
- **φ(s,a)**: Feature vector for state-action pair

### **ε-Greedy Exploration**
```
π(s) = argmax_a Q(s,a) with probability (1-ε)
       random action    with probability ε
```


## 📖 **References**

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
- Gymnasium Mountain Car Environment Documentation


## 🧪 Assignment — Hyperparameter comparison and boxplots

Your task: systematically evaluate different hyperparameter configurations (for example, learning rate and number of RBF centers), run multiple independent trials for each configuration, and produce a boxplot showing the distribution of "steps to success" for the 5 most interesting configurations.

Guidelines:
- Each run should be executed for 50,000 environment steps.
- Choose a sensible grid of hyperparameters to explore. Typical choices to vary:
  - learning_rate (α): e.g. [0.005, 0.01, 0.02, 0.05, 0.1]
  - n_centers (RBF): e.g. [9, 16, 25, 36, 49]
  - keep other parameters fixed (epsilon schedule, discount_factor)
- For each hyperparameter configuration, run 10 independent seeds and record the number of environment steps required to reach 10 episodes reaching the goal.
- After collecting results, pick the 5 most interesting configurations (by median or mean steps-to-success or by visual inspection) and create a matplotlib boxplot that compares their step-to-success distributions.

Deliverables
- A run.py script  that runs the experiments and saves raw results (CSV/JSON).
- A short report (Markdown) showing the boxplot for the 5 selected configurations and a brief discussion of why they are interesting.

Tips
- Use a fixed RNG seed per trial for reproducibility, but vary the seed across trials.
- Save intermediate results frequently so long runs can be resumed.