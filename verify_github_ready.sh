#!/bin/bash
# Quick verification script before pushing to GitHub

echo "🔍 JEPA-EHR GitHub Pre-Push Verification"
echo "========================================"
echo ""

# Check 1: Private data directories
echo "1️⃣  Checking private data directories..."
if git status --porcelain | grep -qE "(mimic-iv-2.1|data/)"; then
    echo "   ❌ ERROR: Private data directories detected in staging!"
    echo "   Please review .gitignore"
    exit 1
else
    echo "   ✅ Private directories properly ignored"
fi

# Check 2: Model checkpoints
echo ""
echo "2️⃣  Checking for model checkpoints..."
if git status --porcelain | grep -qE "\.pth|\.pth\.tar"; then
    echo "   ❌ ERROR: Model checkpoint files detected!"
    echo "   These files are too large for GitHub"
    exit 1
else
    echo "   ✅ No model checkpoints staged"
fi

# Check 3: CSV data files
echo ""
echo "3️⃣  Checking for CSV data files..."
if git status --porcelain | grep -qE "\.csv"; then
    echo "   ⚠️  WARNING: CSV files detected (may contain patient data)"
    echo "   Please review manually"
else
    echo "   ✅ No CSV files staged"
fi

# Check 4: Virtual environment
echo ""
echo "4️⃣  Checking virtual environment..."
if git status --porcelain | grep -qE "venv/|env/"; then
    echo "   ❌ ERROR: Virtual environment detected in staging!"
    exit 1
else
    echo "   ✅ Virtual environment properly ignored"
fi

# Check 5: Python cache
echo ""
echo "5️⃣  Checking for Python cache..."
if git status --porcelain | grep -qE "__pycache__|\.pyc"; then
    echo "   ❌ ERROR: Python cache files detected!"
    exit 1
else
    echo "   ✅ No Python cache files"
fi

# Check 6: Repository size
echo ""
echo "6️⃣  Checking repository size..."
REPO_SIZE=$(du -sm .git | cut -f1)
if [ "$REPO_SIZE" -gt 100 ]; then
    echo "   ⚠️  WARNING: Repository size is ${REPO_SIZE}MB (large)"
    echo "   Consider reviewing what's included"
else
    echo "   ✅ Repository size: ${REPO_SIZE}MB (acceptable)"
fi

# Summary
echo ""
echo "========================================"
echo "✅ All checks passed! Ready to push to GitHub"
echo ""
echo "Next steps:"
echo "  1. git add ."
echo "  2. git commit -m 'Initial commit: JEPA-EHR implementation'"
echo "  3. git remote add origin https://github.com/yourusername/EHRJEPA.git"
echo "  4. git push -u origin main"
echo ""
