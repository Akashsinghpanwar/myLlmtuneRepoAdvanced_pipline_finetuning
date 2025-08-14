#!/usr/bin/env python3
# /// script
# dependencies = [
#   "marker-pdf",
#   "requests",
#   "python-dotenv",
#   "PyPDF2",
#   "tqdm",
#   "markdown",
#   "aiohttp",
#   "asyncio",
# ]
# ///
"""
PDF to Markdown Converter

This script converts a PDF file to markdown using three different approaches:
1. marker-pdf
2. markitdown (simulated with PyPDF2 and markdown)
3. Gemini Flash via OpenRouter (with parallel processing)

Usage:
    uv run pdf_to_markdown.py --pdf_path <path_to_pdf> --output_dir <output_directory>

Requirements:
    - marker-pdf
    - requests
    - python-dotenv
    - PyPDF2
    - tqdm
    - markdown
    - aiohttp
    - asyncio
"""

import argparse
import json
import os
import re
import sys
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests
import aiohttp
from dotenv import load_dotenv
import tempfile
from tqdm import tqdm

# Try to import marker for PDF conversion
try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    marker_available = True
except ImportError:
    marker_available = False

# Try to import PyPDF2 for PDF parsing
try:
    import PyPDF2
    pypdf2_available = True
except ImportError:
    pypdf2_available = False


def ensure_dependencies():
    """Ensure all required dependencies are installed."""
    missing_deps = []
    
    if not marker_available:
        missing_deps.append("marker-pdf")
    
    if not pypdf2_available:
        missing_deps.append("pypdf2")
    
    try:
        import requests
        import dotenv
        from tqdm import tqdm
        import markdown
        import aiohttp
        import asyncio
    except ImportError as e:
        missing_deps.append(str(e).split("'")[1])
    
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        print("Installing missing dependencies...")
        for dep in missing_deps:
            os.system(f"uv add {dep}")
        print("Please restart the script after installation.")
        sys.exit(1)


def convert_pdf_marker(pdf_path: Path, output_path: Path) -> float:
    """
    Convert PDF to markdown using marker-pdf.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Path to save the markdown output
        
    Returns:
        float: Time taken for conversion in seconds
    """
    if output_path.exists():
        print(f"Skipping marker-pdf conversion - output file already exists: {output_path}")
        return 0.0
        
    if not marker_available:
        raise ImportError("marker-pdf is not installed. Install with: uv add marker-pdf")
    
    print(f"Converting {pdf_path} using marker-pdf...")
    
    start_time = time.time()
    
    # Use marker's PdfConverter
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )
    rendered = converter(str(pdf_path))
    text, _, _ = text_from_rendered(rendered)
    
    # Save the markdown output
    output_path.write_text(text, encoding="utf-8")
    
    elapsed_time = time.time() - start_time
    print(f"Saved marker-pdf output to {output_path} (took {elapsed_time:.2f} seconds)")
    
    return elapsed_time


def convert_pdf_markitdown(pdf_path: Path, output_path: Path) -> float:
    """
    Convert PDF to markdown using a PyPDF2 and markdown library approach
    (since the actual markitdown command-line tool might not be available).
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Path to save the markdown output
        
    Returns:
        float: Time taken for conversion in seconds
    """
    if output_path.exists():
        print(f"Skipping markitdown conversion - output file already exists: {output_path}")
        return 0.0
        
    print(f"Converting {pdf_path} using markitdown approach...")
    
    start_time = time.time()
    
    try:
        import PyPDF2
        import markdown
        
        # Extract text from PDF
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            all_text = []
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                
                # Skip empty pages
                if not page_text.strip():
                    continue
                
                # Add page header
                all_text.append(f"## Page {page_num + 1}\n\n")
                
                # Process text for markdown
                # Split into paragraphs
                paragraphs = page_text.split('\n\n')
                for para in paragraphs:
                    # Clean up the paragraph
                    para = para.strip()
                    if para:
                        # Detect if it might be a heading
                        if len(para) < 100 and para.isupper():
                            all_text.append(f"### {para}\n\n")
                        else:
                            all_text.append(f"{para}\n\n")
                
                # Add page separator
                all_text.append("---\n\n")
            
            # Combine all text and write to file
            full_text = "".join(all_text)
            output_path.write_text(full_text, encoding="utf-8")
            
    except Exception as e:
        raise Exception(f"Error in markitdown conversion: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"Saved markitdown output to {output_path} (took {elapsed_time:.2f} seconds)")
    
    return elapsed_time


async def process_page_with_gemini(page_num: int, page_text: str, num_pages: int, api_key: str, session: aiohttp.ClientSession) -> str:
    """
    Process a single page with Gemini Flash API asynchronously.
    
    Args:
        page_num: Page number
        page_text: Text content of the page
        num_pages: Total number of pages
        api_key: OpenRouter API key
        session: aiohttp ClientSession
        
    Returns:
        str: Markdown content for the page
    """
    # Skip empty pages
    if not page_text.strip():
        return f"\n\n## Page {page_num + 1} (Empty)\n\n---\n\n"
    
    # Create prompt for Gemini Flash
    system_message = """
You are an expert at converting text to clean, well-formatted markdown. 
Your task is to convert the provided text (extracted from a PDF) into proper markdown format.

Follow these guidelines:
1. Preserve the document structure (headings, paragraphs, lists)
2. Use appropriate markdown syntax for headings, lists, tables, code blocks, etc.
3. Clean up any artifacts from the PDF extraction process
4. Do not add any commentary or explanations - just return the converted markdown
5. Preserve the original content as much as possible

IMPORTANT: Your response should ONLY contain the markdown conversion, nothing else.
"""
    
    user_message = f"""
# PDF Page {page_num + 1} of {num_pages}

{page_text}

Convert the above text to clean, well-formatted markdown.
"""
    
    # Prepare request to OpenRouter API
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "google/gemini-2.5-flash-preview",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.2,
        "max_tokens": 4000
    }
    
    # Make API request
    try:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        ) as response:
            response_data = await response.json()
            
            # Extract markdown content
            markdown_content = response_data["choices"][0]["message"]["content"]
            
            # Use regex to extract only the markdown part if needed
            markdown_pattern = r'```markdown\s*(.*?)\s*```'
            markdown_match = re.search(markdown_pattern, markdown_content, re.DOTALL)
            
            if markdown_match:
                clean_markdown = markdown_match.group(1).strip()
            else:
                clean_markdown = markdown_content.strip()
            
            return f"\n\n## Page {page_num + 1}\n\n{clean_markdown}\n\n---\n\n"
            
    except Exception as e:
        return f"\n\n[Error processing page {page_num + 1}: {str(e)}]\n\n---\n\n"


async def process_pages_batch(batch: List[Tuple[int, str]], num_pages: int, api_key: str) -> List[Tuple[int, str]]:
    """
    Process a batch of pages asynchronously.
    
    Args:
        batch: List of (page_num, page_text) tuples
        num_pages: Total number of pages
        api_key: OpenRouter API key
        
    Returns:
        List[Tuple[int, str]]: List of (page_num, markdown_content) tuples
    """
    async with aiohttp.ClientSession() as session:
        tasks = []
        for page_num, page_text in batch:
            task = asyncio.ensure_future(process_page_with_gemini(
                page_num, page_text, num_pages, api_key, session
            ))
            tasks.append((page_num, task))
        
        # Wait for a small delay between creating tasks to avoid rate limiting
        await asyncio.sleep(0.5)
        
        # Wait for all tasks to complete
        results = []
        for page_num, task in tasks:
            markdown_content = await task
            results.append((page_num, markdown_content))
        
        return results


def convert_pdf_gemini_flash(pdf_path: Path, output_path: Path, batch_size: int = 32) -> float:
    """
    Convert PDF to markdown using Gemini Flash via OpenRouter with parallel processing.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Path to save the markdown output
        batch_size: Number of pages to process in parallel (default: 32)
        
    Returns:
        float: Time taken for conversion in seconds
    """
    if output_path.exists():
        print(f"Skipping Gemini Flash conversion - output file already exists: {output_path}")
        return 0.0
        
    if not pypdf2_available:
        raise ImportError("PyPDF2 is not installed. Install with: uv add pypdf2")
    
    # Load API key from .env file
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")
    
    print(f"Converting {pdf_path} using Gemini Flash (parallel processing, batch size: {batch_size})...")
    
    start_time = time.time()
    
    # Extract text from PDF
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        
        # Extract text from all pages first
        page_texts = []
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            page_texts.append((page_num, page_text))
        
        # Process pages in batches
        all_results = []
        
        # Create batches
        batches = [page_texts[i:i + batch_size] for i in range(0, len(page_texts), batch_size)]
        
        # Process each batch
        for batch_idx, batch in enumerate(batches):
            print(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} pages)...")
            
            # Run the async batch processing
            batch_results = asyncio.run(process_pages_batch(batch, num_pages, api_key))
            all_results.extend(batch_results)
            
            # Sleep between batches to avoid rate limiting
            if batch_idx < len(batches) - 1:
                print(f"Waiting 2 seconds before next batch...")
                time.sleep(2)
        
        # Sort results by page number
        all_results.sort(key=lambda x: x[0])
        
        # Combine all markdown content
        full_markdown = "".join([content for _, content in all_results])
        
        # Write to file
        output_path.write_text(full_markdown, encoding="utf-8")
    
    elapsed_time = time.time() - start_time
    print(f"Saved Gemini Flash output to {output_path} (took {elapsed_time:.2f} seconds)")
    
    return elapsed_time


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Convert PDF to markdown using different approaches.")
    parser.add_argument("--pdf_path", type=str, required=True, help="Path to the PDF file")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the markdown output")
    parser.add_argument("--force", action="store_true", help="Force reconversion even if output files exist")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for parallel processing (default: 32)")
    args = parser.parse_args()
    
    # Ensure dependencies are installed
    ensure_dependencies()
    
    # Validate PDF path
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists() or not pdf_path.is_file():
        print(f"PDF file not found: {pdf_path}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = pdf_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output paths
    base_name = pdf_path.stem
    marker_output_path = output_dir / f"{base_name}_marker.md"
    markitdown_output_path = output_dir / f"{base_name}_markitdown.md"
    gemini_output_path = output_dir / f"{base_name}_gemini.md"
    
    # If force flag is set, remove existing output files
    if args.force:
        for path in [marker_output_path, markitdown_output_path, gemini_output_path]:
            if path.exists():
                path.unlink()
                print(f"Removed existing file: {path}")
    
    # Track conversion times
    conversion_times = {
        "marker-pdf": 0.0,
        "markitdown": 0.0,
        "gemini-flash": 0.0
    }
    
    # Convert using marker-pdf
    try:
        conversion_times["marker-pdf"] = convert_pdf_marker(pdf_path, marker_output_path)
    except Exception as e:
        print(f"Error converting with marker-pdf: {e}")
    
    # Convert using markitdown
    try:
        conversion_times["markitdown"] = convert_pdf_markitdown(pdf_path, markitdown_output_path)
    except Exception as e:
        print(f"Error converting with markitdown: {e}")
    
    # Convert using Gemini Flash with parallel processing
    try:
        conversion_times["gemini-flash"] = convert_pdf_gemini_flash(pdf_path, gemini_output_path, args.batch_size)
    except Exception as e:
        print(f"Error converting with Gemini Flash: {e}")
    
    # Print summary
    print("\nConversion complete!")
    print(f"Marker output: {marker_output_path}")
    print(f"Markitdown output: {markitdown_output_path}")
    print(f"Gemini Flash output: {gemini_output_path}")
    
    # Print timing results
    print("\nPerformance comparison:")
    print("-" * 50)
    print(f"{'Method':<15} | {'Time (seconds)':<15} | {'Status':<15}")
    print("-" * 50)
    
    for method, elapsed_time in conversion_times.items():
        status = "Completed" if elapsed_time > 0 else "Skipped"
        print(f"{method:<15} | {elapsed_time:15.2f} | {status:<15}")
    
    # Determine the fastest method (only among completed ones)
    completed_conversions = {k: v for k, v in conversion_times.items() if v > 0}
    if completed_conversions:
        fastest_method = min(completed_conversions.items(), key=lambda x: x[1])
        print("-" * 50)
        print(f"Fastest method: {fastest_method[0]} ({fastest_method[1]:.2f} seconds)")


if __name__ == "__main__":
    main()
