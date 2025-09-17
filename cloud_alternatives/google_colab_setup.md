# FlavorGraph AI Training on Google Colab

## 🆓 Free Option (with limitations)

### 1. Colab Pro vs Free
```
Colab Free:
- T4 GPU (16GB) - FREE but limited time
- 12-hour max sessions
- May be interrupted

Colab Pro ($10/month):
- V100/A100 access
- Longer sessions (24+ hours)
- Priority access
```

### 2. Setup Notebook
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone FlavorGraph
!git clone <your-repo> /content/FlavorGraph
%cd /content/FlavorGraph

# Install requirements
!pip install -r paperspace/requirements.txt

# Generate training data
!python3 paperspace/prepare_training_data.py
```

### 3. Training Configuration
```python
# For T4 (16GB) - Use Mistral QLoRA
!python3 paperspace/train_flavor_model.py \
  --config paperspace/configs/mistral7b_qlora.yaml

# For V100/A100 (Colab Pro) - Use Llama
!python3 paperspace/train_flavor_model.py \
  --config paperspace/configs/llama7b_lora.yaml
```

### 4. Save to Drive
```python
# Save trained model to Google Drive
import shutil
shutil.copytree('/content/FlavorGraph/output', 
                '/content/drive/MyDrive/flavorgraph_model')
```

## 💰 Cost Estimate
- **Colab Free**: $0 (but may not complete)
- **Colab Pro**: $10/month (can complete training)

## 🎯 Best For
- ✅ Testing the pipeline first
- ✅ Learning/experimentation
- ⚠️ May timeout on full training
- ⚠️ Slower than dedicated GPUs
