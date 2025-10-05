#!/bin/bash
# FlavorGraph LLaMA Training - Full Pipeline Runner
# This script runs the complete training pipeline from data validation to model training

set -e  # Exit on error

echo "======================================================================"
echo "FlavorGraph LLaMA Training Pipeline"
echo "======================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Validate Data
echo -e "${BLUE}Step 1/4: Validating data...${NC}"
python training/preprocess_data.py
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Warning: Data validation issues detected${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo -e "${GREEN}✓ Data validation complete${NC}"
echo ""

# Step 2: Generate Training Data
echo -e "${BLUE}Step 2/4: Generating training data...${NC}"
if [ -f "training/data/flavorgraph_training_data.jsonl" ]; then
    echo "Training data already exists."
    read -p "Regenerate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python training/generate_llama_training_data.py
    fi
else
    python training/generate_llama_training_data.py
fi
echo -e "${GREEN}✓ Training data ready${NC}"
echo ""

# Step 3: Check GPU availability
echo -e "${BLUE}Step 3/4: Checking GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}Warning: No GPU detected. Training will be very slow on CPU.${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# Step 4: Train Model
echo -e "${BLUE}Step 4/4: Starting model training...${NC}"
echo "This may take several hours depending on your hardware."
echo ""

CONFIG_FILE="training/config_llama_training.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if W&B should be disabled
read -p "Use Weights & Biases for logging? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    export WANDB_MODE=disabled
    echo "W&B logging disabled"
fi

# Start training
python training/train_llama.py --config "$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo -e "${GREEN}✓ Training pipeline completed successfully!${NC}"
    echo "======================================================================"
    echo ""
    echo "Model saved to: training/output/flavorgraph_llama_v1"
    echo ""
    echo "Next steps:"
    echo "1. Evaluate: python training/evaluate_model.py --model training/output/flavorgraph_llama_v1"
    echo "2. Test inference with the example in training/README.md"
    echo ""
else
    echo ""
    echo "======================================================================"
    echo -e "${YELLOW}Training failed or was interrupted${NC}"
    echo "======================================================================"
    exit 1
fi
