import sys
import os

print("="*60)
print("SETUP VERIFICATION")
print("="*60)

# Check Python version
print(f"\nPython version: {sys.version.split()[0]}")

# Check packages
packages = ['numpy', 'pandas', 'sklearn', 'xgboost', 'lightgbm',
            'matplotlib', 'seaborn', 'shap', 'streamlit', 'jupyter']

print("\nPackage Check:")
missing = []
for package in packages:
    try:
        __import__(package)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} - MISSING!")
        missing.append(package)

# Check folders
print("\nFolder Structure Check:")
folders = ['data/raw', 'data/processed', 'notebooks', 'src', 
           'models', 'reports/figures', 'dashboard']
for folder in folders:
    if os.path.exists(folder):
        print(f"  ✓ {folder}")
    else:
        print(f"  ✗ {folder} - MISSING!")

print("\n" + "="*60)
if missing:
    print("⚠️  Some packages are missing!")
else:
    print("✅ ALL CHECKS PASSED! Setup complete!")
print("="*60)