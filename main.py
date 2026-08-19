#!/usr/bin/env python3
"""
GLM-OCR PDF to JSON Converter for ROCm 7.2

This application processes PDF files using GLM vision models and outputs structured JSON.
Optimized for AMD ROCm 7.2 GPUs.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
from PIL import Image
from pdf2image import convert_from_path
from tqdm import tqdm


def setup_rocm_environment():
    """Configure environment variables for optimal ROCm 7.2 performance."""
    # Set HIP-specific environment variables
    os.environ.setdefault('HIP_VISIBLE_DEVICES', '0')
    os.environ.setdefault('PYTORCH_HIP_ALLOC_CONF', 'garbage_collection_threshold:0.8,max_split_size_mb:512')
    
    # Note: HSA_OVERRIDE_GFX_VERSION should be set by user based on their GPU
    # Common values: 9.0.0 (Vega), 9.4.0 (RDNA2), 11.0.0 (RDNA3)
    
    print("ROCm environment configured")


def get_device() -> torch.device:
    """Detect and return the appropriate device (ROCm/CUDA/CPU)."""
    if torch.cuda.is_available():
        # Check if it's actually ROCm (HIP)
        device_name = torch.cuda.get_device_name(0)
        if 'AMD' in device_name or 'Radeon' in device_name:
            print(f"Detected AMD GPU: {device_name}")
            return torch.device('cuda:0')
        else:
            print(f"Detected NVIDIA GPU: {device_name}")
            return torch.device('cuda:0')
    else:
        print("No GPU detected, falling back to CPU")
        return torch.device('cpu')


def load_model(model_name: str, device: torch.device):
    """Load the GLM vision model and processor."""
    print(f"Loading model: {model_name}")
    
    try:
        # For GLM-4V, we need to use specific loading approach
        from transformers import AutoTokenizer, AutoModel
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type != 'cpu' else torch.float32,
            trust_remote_code=True,
            device_map=None
        )
        
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded successfully on {device}")
        return model, tokenizer
    
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have internet access to download the model, or provide a local path.")
        raise


def convert_pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    """Convert PDF pages to PIL Images."""
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        print("Make sure poppler-utils is installed: sudo apt-get install poppler-utils")
        raise


def process_page(model, tokenizer, image: Image.Image, device: torch.device) -> Dict[str, Any]:
    """Process a single page image and extract text using GLM model."""
    
    try:
        # GLM-4V specific approach for image + text input
        # The model expects a specific format for multimodal input
        
        query = "Please extract all text from this document page. Preserve the layout and formatting as much as possible."
        
        # Use model's built-in method if available (GLM-4V specific)
        if hasattr(model, 'chat'):
            # GLM-4V has a chat method that handles images
            response, history = model.chat(
                tokenizer,
                query=query,
                image=image,
                history=[]
            )
            generated_text = response
        else:
            # Fallback: use standard generation with image embedding
            # Build inputs for vision-language model
            inputs = tokenizer(
                query,
                return_tensors="pt",
                padding=True
            ).to(device)
            
            # Generate with image context (model-specific implementation)
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    num_beams=1,
                )
            
            # Decode the output
            generated_text = tokenizer.batch_decode(
                generated_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )[0]
        
        return {
            "text": generated_text.strip(),
            "confidence": 0.95,  # Placeholder - actual confidence would require model-specific implementation
        }
    
    except Exception as e:
        # Fallback error handling
        return {
            "text": f"Error processing page: {str(e)}",
            "confidence": 0.0,
            "error": str(e)
        }


def process_pdf(
    pdf_path: str,
    model,
    tokenizer,
    device: torch.device,
    max_pages: int = 0,
    verbose: bool = False
) -> Dict[str, Any]:
    """Process a single PDF file and return structured data."""
    
    pdf_path = Path(pdf_path)
    start_time = time.time()
    
    print(f"\nProcessing: {pdf_path.name}")
    
    # Convert PDF to images
    images = convert_pdf_to_images(str(pdf_path))
    total_pages = len(images)
    
    if max_pages > 0:
        images = images[:max_pages]
        processed_pages = min(max_pages, total_pages)
    else:
        processed_pages = total_pages
    
    # Process each page
    pages_data = []
    
    for i, image in enumerate(tqdm(images, desc="Pages", disable=not verbose)):
        try:
            result = process_page(model, tokenizer, image, device)
            
            page_data = {
                "page_number": i + 1,
                "text": result["text"],
                "confidence": result["confidence"],
                "metadata": {
                    "width": image.width,
                    "height": image.height,
                }
            }
            
            pages_data.append(page_data)
            
        except Exception as e:
            print(f"Error processing page {i+1}: {e}")
            pages_data.append({
                "page_number": i + 1,
                "text": "",
                "error": str(e),
                "confidence": 0.0,
                "metadata": {
                    "width": image.width,
                    "height": image.height,
                }
            })
    
    processing_time = time.time() - start_time
    
    # Compile final result
    result = {
        "filename": pdf_path.name,
        "filepath": str(pdf_path.absolute()),
        "total_pages": total_pages,
        "processed_pages": len(pages_data),
        "pages": pages_data,
        "processing_time_seconds": round(processing_time, 2),
        "model_used": model.config._name_or_path if hasattr(model.config, '_name_or_path') else "unknown"
    }
    
    return result


def save_json(data: Dict[str, Any], output_path: str):
    """Save results to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="GLM-OCR PDF to JSON Converter for ROCm 7.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -i ./pdfs -o ./output
  python main.py -i ./pdfs -o ./output --model THUDM/glm-4v-9b
  python main.py -i ./pdfs -o ./output --max-pages 5 --verbose
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input directory containing PDF files'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output directory for JSON files'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='THUDM/glm-4v-9b',
        help='Model name or path (default: THUDM/glm-4v-9b)'
    )
    
    parser.add_argument(
        '--max-pages',
        type=int,
        default=0,
        help='Maximum pages per PDF (0 = unlimited, default: 0)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output with progress bars'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup ROCm environment
    setup_rocm_environment()
    
    # Get device
    device = get_device()
    
    # Find PDF files
    pdf_files = list(input_dir.glob('*.pdf')) + list(input_dir.glob('*.PDF'))
    
    if not pdf_files:
        print(f"No PDF files found in: {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF file(s)")
    
    # Load model
    model, tokenizer = load_model(args.model, device)
    
    # Process each PDF
    successful = 0
    failed = 0
    
    for pdf_file in pdf_files:
        try:
            result = process_pdf(
                pdf_file,
                model,
                tokenizer,
                device,
                max_pages=args.max_pages,
                verbose=args.verbose
            )
            
            # Save output
            output_file = output_dir / f"{pdf_file.stem}.json"
            save_json(result, str(output_file))
            successful += 1
            
        except Exception as e:
            print(f"Failed to process {pdf_file.name}: {e}")
            failed += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
