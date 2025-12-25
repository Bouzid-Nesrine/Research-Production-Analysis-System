#!/bin/bash

# Getting Started - Setup Verification Script
# Run this first to check if everything is ready

echo "=================================================="
echo "Research Article Classification - Setup Check"
echo "=================================================="
echo ""

# Function to check command
check_command() {
    if command -v $1 &> /dev/null; then
        echo "✅ $1 is installed"
        return 0
    else
        echo "❌ $1 is NOT installed"
        return 1
    fi
}

# Function to check Python package
check_python_package() {
    if python -c "import $1" 2>/dev/null; then
        VERSION=$(python -c "import $1; print($1.__version__)" 2>/dev/null)
        echo "✅ $1 is installed (version: $VERSION)"
        return 0
    else
        echo "❌ $1 is NOT installed"
        return 1
    fi
}

echo "[1/5] Checking Python..."
check_command python || check_command python3

echo ""
echo "[2/5] Checking key Python packages..."
check_python_package torch
check_python_package transformers
check_python_package peft
check_python_package datasets

echo ""
echo "[3/5] Checking CUDA/GPU..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA is available')
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   CUDA Version: {torch.version.cuda}')
    print(f'   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('⚠️  CUDA is NOT available (CPU training will be very slow)')
" 2>/dev/null || echo "⚠️  Could not check CUDA (PyTorch not installed?)"

echo ""
echo "[4/5] Checking data..."
if [ -d "./Abderrahmane_final_data/Abderrahmane_final_data" ]; then
    NUM_FILES=$(find "./Abderrahmane_final_data/Abderrahmane_final_data" -name "*.json" | wc -l)
    echo "✅ Raw data directory found ($NUM_FILES JSON files)"
else
    echo "❌ Raw data directory NOT found"
    echo "   Expected: ./Abderrahmane_final_data/Abderrahmane_final_data"
fi

if [ -d "./processed_data" ]; then
    echo "✅ Processed data directory found"
    ls -lh ./processed_data/*.json 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'
else
    echo "⚠️  Processed data NOT found (run prepare_data.py to create)"
fi

echo ""
echo "[5/5] Checking disk space..."
AVAILABLE=$(df -h . | awk 'NR==2 {print $4}')
echo "   Available space: $AVAILABLE"

echo ""
echo "=================================================="
echo "SETUP SUMMARY"
echo "=================================================="

# Determine readiness
if check_python_package torch > /dev/null 2>&1 && \
   check_python_package transformers > /dev/null 2>&1 && \
   [ -d "./Abderrahmane_final_data/Abderrahmane_final_data" ]; then
    echo "✅ Setup is READY!"
    echo ""
    echo "Next steps:"
    echo "  1. Prepare data:    python prepare_data.py"
    echo "  2. Train model:     ./run_pipeline.sh phi-2 3 4"
    echo "  3. Or step-by-step: See QUICKSTART.md"
else
    echo "❌ Setup is INCOMPLETE"
    echo ""
    echo "Fix the issues above and try again."
    echo ""
    echo "To install dependencies:"
    echo "  pip install -r requirements.txt"
fi

echo "=================================================="
