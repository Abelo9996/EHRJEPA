#!/usr/bin/env python3
"""
Verify JEPA-EHR installation and file structure
"""

import os
import sys

def check_file(filepath, description):
    """Check if a file exists and print status"""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"  {status} {description}")
    return exists

def main():
    print("=" * 70)
    print("JEPA-EHR Installation Verification")
    print("=" * 70)
    print()
    
    base_dir = "/Users/abelyagubyan/Downloads/ijepa"
    os.chdir(base_dir)
    
    all_good = True
    
    # Core model files
    print("📦 Core Model Files:")
    all_good &= check_file("src/models/temporal_transformer.py", "Temporal Transformer (encoder/predictor)")
    all_good &= check_file("src/datasets/mimic_ehr.py", "EHR Dataset Loader")
    all_good &= check_file("src/masks/temporal.py", "Temporal Masking Strategies")
    all_good &= check_file("src/helper_ehr.py", "EHR Helper Functions")
    print()
    
    # Training scripts
    print("🚀 Training Scripts:")
    all_good &= check_file("train_ehr.py", "Main Training Loop")
    all_good &= check_file("main_ehr.py", "Training Entry Point")
    print()
    
    # Configuration files
    print("⚙️  Configuration Files:")
    all_good &= check_file("configs/mimic_ehr_base.yaml", "Base Model Config")
    all_good &= check_file("configs/mimic_ehr_small.yaml", "Small Model Config")
    print()
    
    # Utility scripts
    print("🛠️  Utility Scripts:")
    all_good &= check_file("generate_sample_data.py", "Sample Data Generator")
    all_good &= check_file("quickstart.sh", "Quick Start Script")
    all_good &= check_file("test_components.py", "Component Test Script")
    all_good &= check_file("verify_installation.py", "This Script")
    print()
    
    # Documentation
    print("📚 Documentation:")
    all_good &= check_file("README_JEPA_EHR.md", "Comprehensive README")
    all_good &= check_file("IMPLEMENTATION_SUMMARY.md", "Technical Details")
    all_good &= check_file("GETTING_STARTED.md", "Quick Start Guide")
    all_good &= check_file("JEPA_EHR_OVERVIEW.md", "Complete Overview")
    all_good &= check_file("requirements_ehr.txt", "Python Dependencies")
    print()
    
    # Check directory structure
    print("📁 Directory Structure:")
    required_dirs = [
        "src/models",
        "src/datasets",
        "src/masks",
        "src/utils",
        "configs",
    ]
    
    for dir_path in required_dirs:
        exists = os.path.isdir(dir_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_path}/")
        all_good &= exists
    print()
    
    # Summary
    print("=" * 70)
    if all_good:
        print("✅ All files present! Installation verified.")
        print()
        print("Quick Start Options:")
        print("  1. Automatic: bash quickstart.sh")
        print("  2. Manual:    python generate_sample_data.py --num_patients 1000")
        print("                python main_ehr.py --fname configs/mimic_ehr_small.yaml")
        print()
        print("Documentation:")
        print("  - Quick start: JEPA_EHR_OVERVIEW.md")
        print("  - Full guide:  README_JEPA_EHR.md")
        print("  - Tech details: IMPLEMENTATION_SUMMARY.md")
    else:
        print("❌ Some files are missing. Please check the installation.")
        return 1
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
