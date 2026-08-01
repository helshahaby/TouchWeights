@echo off
echo ========================================
echo Creating Execution-Grounded LoRA Project
echo ========================================

REM Root folders
mkdir dataset
mkdir environment
mkdir models
mkdir rollout
mkdir trainer
mkdir evaluation
mkdir app

REM Root files
type nul > demo.py
type nul > requirements.txt
type nul > README.md

REM Dataset
type nul > dataset\download_dataset.py
type nul > dataset\loader.py
type nul > dataset\gsm8k.json
type nul > dataset\__init__.py

REM Environment
type nul > environment\executor.py
type nul > environment\reward.py
type nul > environment\sandbox.py
type nul > environment\verifier.py
type nul > environment\__init__.py

REM Models
type nul > models\model.py
type nul > models\tokenizer.py
type nul > models\lora.py
type nul > models\__init__.py

REM Rollout
type nul > rollout\generator.py
type nul > rollout\trajectory.py
type nul > rollout\__init__.py

REM Trainer
type nul > trainer\grpo_train.py
type nul > trainer\config.py
type nul > trainer\callbacks.py
type nul > trainer\__init__.py

REM Evaluation
type nul > evaluation\benchmark.py
type nul > evaluation\compare.py
type nul > evaluation\metrics.py
type nul > evaluation\__init__.py

REM App
type nul > app\streamlit_app.py
type nul > app\__init__.py

echo.
echo ========================================
echo Project structure created successfully!
echo ========================================
pause