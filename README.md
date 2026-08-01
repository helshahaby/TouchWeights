# Touch Weights - Continual Learning Hack

## Execution-Grounded LoRA: Synthetic Reinforcement Learning for Self-Improving LLMs

**Hackathon Project**

**Author:** Hossam Elshahaby

---

# 1. Project Overview

Large Language Models (LLMs) are powerful, but adapting them to new tasks is still expensive and slow.

Current approaches usually depend on:

- Massive retraining
- Human feedback labeling
- Retrieval-Augmented Generation (RAG)
- Prompt engineering

These approaches do not allow the model to directly learn from its own mistakes.

**Touch Weights introduces a lightweight continual learning loop where an AI model improves its own parameters through interaction with a synthetic environment.**

The model:

1. Generates a solution.
2. Executes the solution inside a controlled environment.
3. Receives automatic feedback.
4. Learns from the reward signal.
5. Updates its weights using LoRA-based reinforcement learning.

---

# 2. Key Idea

## From Static AI to Self-Improving AI

Traditional LLM:

User Question
|
v
LLM
|
v
Answer


The model does not learn after making mistakes.

---

Touch Weights:


          User Task
              |
              v
      Language Model
              |
      Generate Solution
              |
              v
   Synthetic Environment
              |
    Execute / Validate
              |
              v
          Reward
      (+1 Success / -1 Failure)
              |
              v
      GRPO Reinforcement Learning
              |
              v
      LoRA Weight Update
              |
              v
      Improved Model

      
---

# 3. Problem Statement

## The Challenge

Modern LLM systems have three major limitations:

## 1. No Real-Time Learning

Most deployed models cannot update their knowledge or behavior after deployment.

A model that fails today will usually fail tomorrow.

---

## 2. Expensive Feedback

Human preference labeling requires:

- Experts
- Time
- Money

Scaling human feedback is difficult.

---

## 3. Prompt-Based Improvements Are Limited

Techniques like:

- Prompt engineering
- Few-shot examples
- RAG

change the input context but do not modify the model itself.

The model remains unchanged.

---

# 4. Our Solution

## Execution-Grounded Continual Learning

Touch Weights creates a synthetic reinforcement learning environment.

Instead of humans providing feedback:

**The environment becomes the teacher.**

The system uses:

- Qwen2.5-Coder as the base model
- Python execution environment as the evaluator
- Automatic reward generation
- GRPO optimization
- LoRA adapter updates

---

# 5. Architecture

## High-Level Architecture
                Problem
                   |
                   v

          Qwen2.5-Coder Model

                   |
                   |
         Generate Python Solution

                   |
                   v

      Synthetic Execution Environment

          +----------------+
          | Python Sandbox |
          +----------------+

                   |
                   v

          Execution Result

                   |
                   v

          Reward Function

         Correct     Wrong
            |          |
           +1         -1

                   |
                   v

              GRPO Trainer

                   |
                   v

          LoRA Adapter Update

                   |
                   v

          Improved AI Model


---

# 6. Technology Stack

| Component | Technology |
|---|---|
| Base Model | Qwen2.5-Coder-1.5B |
| Reinforcement Learning | TRL GRPO |
| Parameter Update | LoRA / QLoRA |
| Environment | Python Execution Sandbox |
| Dataset | GSM8K Mathematics Dataset |
| Framework | Hugging Face Transformers |
| Optimization | PEFT |
| Demo Interface | Python / Streamlit |
| Hardware | NVIDIA GPU |

---

# 7. How It Works

## Step 1 - Generate Solution

The model receives:


Calculate 25 × 12 using Python



The model generates:


print(25*10)

## Step 2 - Execute

The synthetic environment runs the code.

Output:

250

## Step 3 - Evaluate

Expected:

300

Actual:

250

Reward:

-1

## Step 4 - Learn

GRPO compares multiple generated solutions.

Example:

Solution A
Reward: -1


Solution B
Reward: +1


Solution C
Reward: -1

The model increases the probability of generating solutions similar to Solution B.


## Step 5 - Update Weights

Instead of retraining billions of parameters:

LoRA updates only small adapter weights.

Original Model

+

LoRA Adapter

=

Improved Model

# 8 Project Structure

Execution-Grounded-LoRA/

│
├── README.md
│
├── config.py
│
├── model.py
│
├── environment.py
│
├── reward.py
│
├── download_data.py
│
├── train_grpo.py
│
├── demo.py
│
├── demo_after_training.py
│
├── evaluate.py
│
└── data/
    |
    └── gsm8k.json


# 9. Installation
Requirements
Python 3.11+
NVIDIA GPU recommended
CUDA installed    

Create environment:

python -m venv venv


Activate in:

Windows:

venv\Scripts\activate

Linux:

source venv/bin/activate

Install dependencies:

pip install -r requirements.txt


# 10. Download Dataset

Run:

python download_data.py

This downloads GSM8K samples:

data/gsm8k.json


# 11. Run Baseline Model

Before learning:

python demo.py

Example:

Question:
Calculate 25*12


Model Output:
print(25*10)


Execution:
250


Reward:
-1

The model fails.



# 12. Run Continual Learning

Start GRPO training:

python train_grpo.py

The training loop:

Generate Rollouts

        |

Execute Solutions

        |

Calculate Rewards

        |

GRPO Optimization

        |

Save LoRA Adapter

Output:

adapter/

adapter_model.safetensors

adapter_config.json


# 13. Test Improved Model

Run:

python demo_after_training.py

Expected:

Before:

Reward: -1

After:

Generated Code:

print(25*12)


Execution:

300


Reward:

+1



# 14. Why This Is Different
Traditional Fine-Tuning
Dataset
   |
   v
Training
   |
   v
New Model
Touch Weights
Model

 |

Interaction

 |

Environment Feedback

 |

Reward

 |

Weight Update

 |

Better Model

The model learns through experience.