#!/bin/bash

# List all files in the fine_tuning directory with descriptions

echo "======================================================================"
echo "FINE-TUNING PIPELINE - FILE STRUCTURE"
echo "======================================================================"
echo ""

cd "/home/zahra/Documents/4rth Year/NLP/Project/Research-Production-Analysis-System/fine_tuning"

echo "📁 fine_tuning/"
echo ""

echo "🚀 MAIN SCRIPTS"
echo "├── prepare_data.py           📊 Convert JSON to training format"
echo "├── train_model.py            🎓 Fine-tune LLM on your data"
echo "├── inference.py              🔮 Classify & evaluate articles"
echo "├── run_pipeline.sh           ⚡ Automated end-to-end pipeline"
echo "└── utils.py                  🛠️  Utility commands"
echo ""

echo "📖 DOCUMENTATION"
echo "├── README.md                 📚 Complete documentation"
echo "├── QUICKSTART.md             ⚡ 5-minute quick start"
echo "├── SUMMARY.md                📋 Overview & summary"
echo "└── FILE_INDEX.md             🗂️  This file index"
echo ""

echo "⚙️  CONFIGURATION"
echo "├── requirements.txt          📦 Python dependencies"
echo "├── config.template.sh        ⚙️  Configuration template"
echo "└── check_setup.sh            ✅ Setup verification"
echo ""

echo "💡 EXAMPLES"
echo "└── example_usage.py          💡 Code examples"
echo ""

echo "📂 DATA DIRECTORIES"
echo "├── Abderrahmane_final_data/  📥 Raw data (235 JSON files)"
echo "├── processed_data/           📤 Formatted training data"
echo "└── output/                   💾 Trained models"
echo ""

echo "======================================================================"
echo "QUICK COMMANDS"
echo "======================================================================"
echo ""
echo "Setup & Validation:"
echo "  ./check_setup.sh                    # Check if setup is ready"
echo "  pip install -r requirements.txt     # Install dependencies"
echo "  python utils.py validate            # Validate setup"
echo ""
echo "Data Preparation:"
echo "  python prepare_data.py              # Prepare training data"
echo "  python utils.py count               # Count articles"
echo "  python utils.py distribution        # Show data distribution"
echo ""
echo "Training:"
echo "  ./run_pipeline.sh phi-2 3 4         # Complete pipeline (recommended)"
echo "  python train_model.py ...           # Manual training"
echo ""
echo "Evaluation & Inference:"
echo "  python inference.py --mode evaluate         # Evaluate model"
echo "  python inference.py --mode interactive      # Interactive mode"
echo "  python inference.py --mode classify ...     # Classify single article"
echo ""
echo "Maintenance:"
echo "  python utils.py check               # Check processed data"
echo "  python utils.py models              # List trained models"
echo "  python utils.py clean               # Clean outputs"
echo ""

echo "======================================================================"
echo "FILE SIZES"
echo "======================================================================"
echo ""

if [ -d "Abderrahmane_final_data" ]; then
    SIZE=$(du -sh Abderrahmane_final_data 2>/dev/null | awk '{print $1}')
    COUNT=$(find Abderrahmane_final_data -name "*.json" 2>/dev/null | wc -l)
    echo "Raw Data:       $SIZE ($COUNT JSON files)"
fi

if [ -d "processed_data" ]; then
    SIZE=$(du -sh processed_data 2>/dev/null | awk '{print $1}')
    echo "Processed Data: $SIZE"
fi

if [ -d "output" ]; then
    SIZE=$(du -sh output 2>/dev/null | awk '{print $1}')
    COUNT=$(find output -name "final_model" -type d 2>/dev/null | wc -l)
    echo "Models:         $SIZE ($COUNT trained models)"
fi

echo ""
echo "======================================================================"
echo "DOCUMENTATION QUICK LINKS"
echo "======================================================================"
echo ""
echo "1. Start here:        cat QUICKSTART.md"
echo "2. Full docs:         cat README.md"
echo "3. Overview:          cat SUMMARY.md"
echo "4. File guide:        cat FILE_INDEX.md"
echo "5. Example code:      cat example_usage.py"
echo ""
echo "======================================================================"
