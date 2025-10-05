#!/usr/bin/env python3
"""
FlavorGraph Training Setup Checker
Validates environment, data, and dependencies before training
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


class SetupChecker:
    """Validates training setup"""

    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.checks_passed = []
        self.checks_failed = []
        self.warnings = []

        print("=" * 70)
        print("🔍 FlavorGraph Training Setup Checker")
        print("=" * 70 + "\n")

    def check_python_version(self) -> bool:
        """Check Python version"""
        print("🐍 Checking Python version...")

        version = sys.version_info
        required = (3, 8)

        if version >= required:
            print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            print(f"   ❌ Python {version.major}.{version.minor} (requires {required[0]}.{required[1]}+)")
            return False

    def check_gpu(self) -> bool:
        """Check GPU availability"""
        print("\n🎮 Checking GPU...")

        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"   ✅ {gpu_name} ({memory_gb:.1f} GB)")

                # Check memory
                if memory_gb < 8:
                    self.warnings.append(
                        f"GPU memory ({memory_gb:.1f} GB) is low. Training may be slow or fail."
                    )
                    print(f"   ⚠️  Low memory: {memory_gb:.1f} GB (recommend 24GB+)")

                return True
            else:
                print("   ⚠️  No GPU detected - training will be very slow on CPU")
                self.warnings.append("No GPU detected. Consider using cloud GPUs (Colab, Paperspace, RunPod)")
                return False

        except ImportError:
            print("   ❌ PyTorch not installed")
            return False

    def check_dependencies(self) -> Tuple[bool, List[str]]:
        """Check required Python packages"""
        print("\n📦 Checking dependencies...")

        required_packages = {
            'torch': 'PyTorch',
            'transformers': 'Hugging Face Transformers',
            'datasets': 'Hugging Face Datasets',
            'peft': 'PEFT (LoRA)',
            'bitsandbytes': 'BitsAndBytes',
            'pandas': 'Pandas',
            'numpy': 'NumPy',
            'yaml': 'PyYAML',
        }

        missing = []
        installed = []

        for package, name in required_packages.items():
            try:
                __import__(package)
                installed.append(name)
            except ImportError:
                missing.append(name)

        if missing:
            print(f"   ❌ Missing packages: {', '.join(missing)}")
            print(f"   ✅ Installed: {', '.join(installed)}")
            print("\n   Install with: pip install -r training/requirements.txt")
            return False, missing
        else:
            print(f"   ✅ All required packages installed ({len(installed)})")
            return True, []

    def check_data_files(self) -> Tuple[bool, List[str]]:
        """Check required data files"""
        print("\n📊 Checking data files...")

        required_files = {
            'input/nodes_191120.csv': 'Nodes (ingredients + compounds)',
            'input/edges_191120.csv': 'Edges (relationships)',
            'input/recipes/extracted_recipes.json': 'Recipe data',
            'input/compound_flavors/compound_flavor_mappings.json': 'Compound flavors',
        }

        missing = []
        found = []

        for filepath, description in required_files.items():
            full_path = self.base_dir / filepath
            if full_path.exists():
                size_mb = full_path.stat().st_size / 1e6
                found.append(f"{description} ({size_mb:.1f} MB)")
            else:
                missing.append(filepath)

        if missing:
            print(f"   ❌ Missing files:")
            for f in missing:
                print(f"      - {f}")
            return False, missing
        else:
            print(f"   ✅ All data files found:")
            for f in found:
                print(f"      - {f}")
            return True, []

    def check_training_files(self) -> bool:
        """Check training script files"""
        print("\n📝 Checking training scripts...")

        training_files = [
            'training/generate_llama_training_data.py',
            'training/train_llama.py',
            'training/evaluate_model.py',
            'training/config_llama_training.yaml',
        ]

        all_exist = True
        for filepath in training_files:
            full_path = self.base_dir / filepath
            if not full_path.exists():
                print(f"   ❌ Missing: {filepath}")
                all_exist = False

        if all_exist:
            print(f"   ✅ All training scripts present ({len(training_files)} files)")
        return all_exist

    def check_disk_space(self) -> bool:
        """Check available disk space"""
        print("\n💾 Checking disk space...")

        try:
            import shutil
            stat = shutil.disk_usage(self.base_dir)
            free_gb = stat.free / 1e9

            print(f"   💿 Free space: {free_gb:.1f} GB")

            if free_gb < 50:
                print(f"   ⚠️  Low disk space (recommend 100GB+)")
                self.warnings.append(f"Low disk space: {free_gb:.1f} GB")
                return False
            else:
                print(f"   ✅ Sufficient space")
                return True

        except Exception as e:
            print(f"   ⚠️  Could not check disk space: {e}")
            return True

    def check_huggingface_token(self) -> bool:
        """Check if HuggingFace token is configured"""
        print("\n🤗 Checking HuggingFace access...")

        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()

            if token:
                print("   ✅ HuggingFace token found")
                return True
            else:
                print("   ⚠️  No HuggingFace token (optional)")
                print("   ℹ️  Login with: huggingface-cli login")
                self.warnings.append(
                    "No HuggingFace token. Some models may require authentication."
                )
                return True  # Not critical

        except ImportError:
            print("   ⚠️  huggingface_hub not installed (optional)")
            return True

    def estimate_training_time(self):
        """Estimate training time based on hardware"""
        print("\n⏱️  Training time estimates...")

        try:
            import torch
            if torch.cuda.is_available():
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

                # Rough estimates for LLaMA 2 7B, 3 epochs
                if memory_gb >= 40:
                    time_est = "2-3 hours (A100/H100)"
                elif memory_gb >= 24:
                    time_est = "4-5 hours (RTX 4090/A5000)"
                elif memory_gb >= 16:
                    time_est = "6-8 hours (RTX 4060 Ti 16GB)"
                else:
                    time_est = "8-12+ hours (limited VRAM)"

                print(f"   ⏱️  Estimated time: {time_est}")
            else:
                print("   ⏱️  CPU training: Several days (not recommended)")

        except ImportError:
            pass

    def generate_report(self) -> bool:
        """Generate final report"""
        print("\n" + "=" * 70)
        print("📊 SETUP REPORT")
        print("=" * 70 + "\n")

        all_checks = [
            ("Python version", self.check_python_version()),
            ("GPU availability", self.check_gpu()),
            ("Dependencies", self.check_dependencies()[0]),
            ("Data files", self.check_data_files()[0]),
            ("Training scripts", self.check_training_files()),
            ("Disk space", self.check_disk_space()),
            ("HuggingFace access", self.check_huggingface_token()),
        ]

        passed = sum(1 for _, status in all_checks if status)
        total = len(all_checks)

        print("Check Results:")
        for name, status in all_checks:
            icon = "✅" if status else "❌"
            print(f"  {icon} {name}")

        print(f"\nPassed: {passed}/{total}")

        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")

        print("\n" + "=" * 70)

        if passed == total and not self.warnings:
            print("✅ READY TO TRAIN!")
            print("=" * 70)
            print("\nNext steps:")
            print("1. Generate training data:")
            print("   python training/generate_llama_training_data.py")
            print("\n2. Start training:")
            print("   python training/train_llama.py --config training/config_llama_training.yaml")
            print("\nOr run full pipeline:")
            print("   bash training/run_full_pipeline.sh")
            return True

        elif passed >= 5:  # Core requirements met
            print("⚠️  READY WITH WARNINGS")
            print("=" * 70)
            print("\nYou can proceed, but review warnings above.")
            print("\nTo install missing dependencies:")
            print("   pip install -r training/requirements.txt")
            return True

        else:
            print("❌ NOT READY - Please fix errors above")
            print("=" * 70)
            print("\nCommon fixes:")
            print("1. Install dependencies: pip install -r training/requirements.txt")
            print("2. Check data files exist in input/ directory")
            print("3. Ensure GPU drivers are installed (nvidia-smi)")
            return False

    def run(self) -> bool:
        """Run all checks"""
        try:
            self.estimate_training_time()
            return self.generate_report()
        except KeyboardInterrupt:
            print("\n\n⚠️  Check interrupted by user")
            return False
        except Exception as e:
            print(f"\n\n❌ Error during checks: {e}")
            return False


def main():
    checker = SetupChecker()
    is_ready = checker.run()
    sys.exit(0 if is_ready else 1)


if __name__ == "__main__":
    main()
