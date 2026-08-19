#!/bin/bash
# GLM-OCR PDF to JSON Converter - ROCm 7.2 Setup and Run Script

set -e

echo "========================================"
echo "GLM-OCR PDF to JSON Converter for ROCm 7.2"
echo "========================================"
echo ""

# Configuration
MODEL_NAME="${MODEL_NAME:-THUDM/glm-4v-9b}"
INPUT_DIR="${INPUT_DIR:-./pdfs}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
MAX_PAGES="${MAX_PAGES:-0}"
VERBOSE="${VERBOSE:-}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --max-pages)
            MAX_PAGES="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -i, --input DIR      Input directory containing PDFs (default: ./pdfs)"
            echo "  -o, --output DIR     Output directory for JSON files (default: ./output)"
            echo "  --model NAME         Model name (default: THUDM/glm-4v-9b)"
            echo "  --max-pages N        Max pages per PDF, 0 for unlimited (default: 0)"
            echo "  -v, --verbose        Enable verbose output"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set ROCm environment variables for optimal performance
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512"

# Optional: Override GPU architecture if needed
# Uncomment and adjust based on your GPU:
# export HSA_OVERRIDE_GFX_VERSION="9.0.0"   # Vega
# export HSA_OVERRIDE_GFX_VERSION="9.4.0"   # RDNA2
# export HSA_OVERRIDE_GFX_VERSION="11.0.0"  # RDNA3

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Input: $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Max Pages: $MAX_PAGES"
echo "  ROCm Device: ${HIP_VISIBLE_DEVICES}"
echo ""

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check for PDF files
PDF_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -type f \( -name "*.pdf" -o -name "*.PDF" \) | wc -l)
if [ "$PDF_COUNT" -eq 0 ]; then
    echo "Error: No PDF files found in $INPUT_DIR"
    exit 1
fi

echo "Found $PDF_COUNT PDF file(s) to process"
echo ""

# Run the Python application
echo "Starting OCR processing..."
python3 main.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --model "$MODEL_NAME" \
    --max-pages "$MAX_PAGES" \
    $VERBOSE

echo ""
echo "Processing complete!"
echo "Output files are in: $(realpath "$OUTPUT_DIR")"
