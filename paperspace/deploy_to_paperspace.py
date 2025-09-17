#!/usr/bin/env python3
"""
FlavorGraph AI Paperspace Deployment Helper
Guides through the Paperspace deployment process
"""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} failed")
            print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def check_paperspace_cli():
    """Check if Paperspace CLI is installed"""
    return run_command("paperspace --version", "Checking Paperspace CLI")

def main():
    print("🚀 FlavorGraph AI Paperspace Deployment")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("paperspace").exists():
        print("❌ Please run this script from the FlavorGraph root directory")
        sys.exit(1)
    
    print("📋 Pre-deployment Checklist:")
    print("   1. Paperspace account created ✓")
    print("   2. FlavorGraph data prepared ✓")
    print("   3. Training scripts ready ✓")
    
    # Check Paperspace CLI
    print("\n🔧 Checking Paperspace CLI...")
    if not check_paperspace_cli():
        print("\n📦 Installing Paperspace CLI...")
        print("Run: pip install paperspace")
        print("Then: paperspace login")
        return
    
    # Machine recommendations
    print("\n💻 Recommended Paperspace Machines:")
    print("   🔥 A100-40GB: Best performance for Llama 7B LoRA")
    print("   ⚡ RTX4000:   Budget option for Mistral 7B QLoRA")
    print("   🚀 A100-80GB: Fastest training for large datasets")
    
    # Create deployment commands
    print("\n📝 Paperspace Deployment Commands:")
    print("=" * 40)
    
    print("1️⃣  Create and start machine:")
    print("   paperspace machines create \\")
    print("     --machineType A100-40GB \\")
    print("     --size 50 \\")
    print("     --templateId tpl_paperspace_ml \\")
    print("     --name flavorgraph-training")
    
    print("\n2️⃣  Upload FlavorGraph project:")
    print("   # Zip the project first")
    print("   tar -czf flavorgraph.tar.gz --exclude='.git' --exclude='__pycache__' .")
    print("   # Upload via Paperspace interface or:")
    print("   paperspace machines uploadFile --machineId <machine-id> --file flavorgraph.tar.gz")
    
    print("\n3️⃣  SSH into machine and setup:")
    print("   paperspace machines ssh --machineId <machine-id>")
    print("   # Then on the machine:")
    print("   tar -xzf flavorgraph.tar.gz")
    print("   cd FlavorGraph")
    print("   chmod +x paperspace/setup_paperspace.sh")
    print("   ./paperspace/setup_paperspace.sh")
    
    print("\n4️⃣  Start training:")
    print("   # For Llama 7B (A100 recommended):")
    print("   python3 paperspace/train_flavor_model.py --config paperspace/configs/llama7b_lora.yaml")
    print("   # For Mistral 7B (RTX4000+ compatible):")
    print("   python3 paperspace/train_flavor_model.py --config paperspace/configs/mistral7b_qlora.yaml")
    
    print("\n5️⃣  Monitor training:")
    print("   # Check logs in real-time")
    print("   tail -f output/<run_name>/training_logs.txt")
    print("   # Or use Weights & Biases dashboard")
    
    print("\n6️⃣  Download trained model:")
    print("   # Zip the output directory")
    print("   tar -czf trained_model.tar.gz output/")
    print("   # Download via Paperspace interface")
    
    # Alternative: Docker approach
    print("\n🐳 Alternative: Docker Deployment")
    print("=" * 35)
    print("Create a Dockerfile for easier deployment:")
    
    dockerfile_content = '''FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /workspace
COPY . /workspace/

RUN pip install -r paperspace/requirements.txt
RUN chmod +x paperspace/setup_paperspace.sh

CMD ["python3", "paperspace/train_flavor_model.py", "--config", "paperspace/configs/llama7b_lora.yaml"]'''
    
    print(dockerfile_content)
    
    # Final tips
    print("\n💡 Pro Tips:")
    print("   • Use tmux/screen for long-running training sessions")
    print("   • Monitor GPU usage with: nvidia-smi")
    print("   • Save checkpoints frequently in case of interruption")
    print("   • Use Weights & Biases for experiment tracking")
    
    print("\n🎉 Ready for deployment!")
    print("Follow the commands above to deploy FlavorGraph AI on Paperspace.")

if __name__ == "__main__":
    main()
