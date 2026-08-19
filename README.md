# GLM-OCR PDF to JSON Converter for ROCm 7.2

This application processes PDF files using GLM-OCR (based on Zhipu AI's GLM vision model) and outputs structured JSON for each PDF. It's optimized for AMD ROCm 7.2 GPUs.

## Features

- Batch PDF processing
- High-quality OCR using GLM vision models
- Structured JSON output with text content and layout information
- ROCm 7.2 optimization for AMD GPUs
- Progress tracking and error handling

## Requirements

- AMD GPU with ROCm 7.2 support
- Python 3.10+
- Poppler (for pdf2image)

## Installation

### 1. Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y poppler-utils

# Install ROCm 7.2 (if not already installed)
# Follow official AMD ROCm installation guide for your distribution
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. ROCm Configuration

Ensure ROCm 7.2 is properly installed and configured:

```bash
# Verify ROCm installation
rocm-smi --version

# Set HIP_VISIBLE_DEVICES if needed
export HIP_VISIBLE_DEVICES=0
```

## Usage

### Basic Usage

```bash
python main.py --input /path/to/pdfs --output /path/to/output
```

### Command Line Options

```
--input, -i      Input directory containing PDF files (required)
--output, -o     Output directory for JSON files (required)
--model          Model name or path (default: THUDM/glm-4v-9b)
--batch-size     Number of pages to process at once (default: 1)
--device         Device to use: cuda, mps, cpu (default: auto-detect ROCm)
--max-pages      Maximum pages per PDF (0 = unlimited, default: 0)
--verbose        Enable verbose output
```

### Examples

Process all PDFs in a directory:
```bash
python main.py -i ./pdfs -o ./output
```

Specify a custom model:
```bash
python main.py -i ./pdfs -o ./output --model THUDM/glm-4v-9b
```

Limit processing to first 5 pages per PDF:
```bash
python main.py -i ./pdfs -o ./output --max-pages 5
```

## Output Format

Each PDF generates a JSON file with the following structure:

```json
{
  "filename": "example.pdf",
  "total_pages": 10,
  "processed_pages": 10,
  "pages": [
    {
      "page_number": 1,
      "text": "Extracted text content...",
      "confidence": 0.95,
      "layout": {
        "blocks": [...],
        "tables": [...]
      },
      "metadata": {
        "width": 612,
        "height": 792,
        "dpi": 300
      }
    }
  ],
  "processing_time_seconds": 45.2,
  "model_used": "THUDM/glm-4v-9b"
}
```

## ROCm 7.2 Specific Configuration

For optimal performance on ROCm 7.2:

```bash
# Set environment variables before running
export HSA_OVERRIDE_GFX_VERSION=9.0.0  # Adjust for your GPU architecture
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512

# Run the application
python main.py -i ./pdfs -o ./output
```

## Troubleshooting

### Common Issues

1. **ROCm not detected**: Ensure ROCm 7.2 is properly installed and `rocm-smi` is in PATH
2. **Out of memory**: Reduce batch size or limit max pages
3. **Slow processing**: Ensure GPU is being utilized (check with `rocm-smi`)

### Getting Help

- Check ROCm documentation: https://rocm.docs.amd.com/
- GLM model documentation: https://github.com/THUDM/GLM-EdgeV

## License

MIT License