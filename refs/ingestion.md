from pathlib import Path
import subprocess
from typing import Literal, Union
from mcp.server.fastmcp import FastMCP
import sys
import pypandoc

# Create FastMCP server instance
mcp = FastMCP("doc-convert")

# Supported formats
InputFormat = Literal["pdf", "docx", "md"]
OutputFormat = Literal["pdf", "docx", "md"]

def validate_file_exists(file_path: str) -> Union[None, dict]:
    """Validate that a file exists and is readable."""
    path = Path(file_path)
    if not path.exists():
        return {
            "content": [{"type": "text", "text": f"File does not exist: {file_path}"}],
            "isError": True
        }
    if not path.is_file():
        return {
            "content": [{"type": "text", "text": f"Path is not a file: {file_path}"}],
            "isError": True
        }
    if not path.is_absolute():
        return {
            "content": [{"type": "text", "text": f"File path must be absolute: {file_path}"}],
            "isError": True
        }
    return None

def validate_format(format: str, valid_formats: list[str], format_type: str) -> Union[None, dict]:
    """Validate that a format is in the list of valid formats."""
    if format.lower() not in valid_formats:
        return {
            "content": [{"type": "text", "text": f"Invalid {format_type} format: {format}. Must be one of {valid_formats}"}],
            "isError": True
        }
    return None

def run_command(cmd: list[str], cwd: str | None = None) -> None:
    """Run a command and suppress its output."""
    try:
        # Redirect stdout and stderr to devnull to suppress output
        with open('/dev/null', 'w') as devnull:
            if cmd[0] == "marker_single":
                # Use uv run to execute marker_single, no need for manual quoting
                cmd = ["uv", "run", "marker_single"] + cmd[1:]
            subprocess.run(cmd, cwd=cwd, stdout=devnull, stderr=devnull, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nError: {e}")

@mcp.tool()
def convert_document(input_path: str, output_format: OutputFormat) -> str:
    """Convert a document between different formats.
    
    Args:
        input_path: Absolute path to the input document
        output_format: Target format to convert to. One of: "pdf", "docx", "md"
    
    Returns:
        A message indicating success or failure of the conversion
        
    Notes:
        This tool is especially useful as a conversion step to make PDF and DOCX files 
        readable in plain text format by converting them to Markdown. This enables easier
        processing, analysis, and interaction with the content of these document formats.
        
        For conversions to PDF or DOCX, the file is saved in the same directory as the input file,
        with the same filename but with the extension changed to match the output format.
        
        For conversions to Markdown, the markdown file and any images extracted are placed in a folder
        that has the same name as the original file (without extension). The markdown file itself
        will have the same name as the original file but with the .md extension.
        
        For example:
        - Converting /path/to/document.pdf to docx will save as /path/to/document.docx
        - Converting /path/to/document.pdf to md will create a folder /path/to/document/
          containing document.md and any extracted images
    """
    try:
        # Validate input file
        if error := validate_file_exists(input_path):
            return error["content"][0]["text"]
        
        # Get input format from file extension
        input_format = Path(input_path).suffix[1:].lower()
        if input_format not in ["pdf", "docx", "md"]:
            return f"Unsupported input format: {input_format}"
        
        # Validate output format
        if error := validate_format(output_format, ["pdf", "docx", "md"], "output"):
            return error["content"][0]["text"]
        
        # Create output path
        output_path = str(Path(input_path).with_suffix(f".{output_format}"))
        
        # Convert based on input/output format combination
        if input_format == "pdf" and output_format == "md":
            try:
                # For PDF to MD, marker_single already creates the output folder structure
                # Just pass the parent directory where marker_single should create its output
                output_dir = str(Path(input_path).parent)
                run_command(["marker_single", input_path, "--output_dir", output_dir])
                expected_output_folder = str(Path(input_path).parent / Path(input_path).stem)
                return f"Successfully converted {input_path} to markdown. Files saved in folder: {expected_output_folder}"
            except RuntimeError as e:
                return f"Failed to convert PDF to markdown: {e}"
                
        elif input_format == "docx" and output_format == "md":
            try:
                # For DOCX to MD, files are saved in a folder with the same name as the original file
                # The markdown file will have the same name as the original file with .md extension
                output_dir = Path(input_path).parent / Path(input_path).stem
                output_dir.mkdir(exist_ok=True)
                md_path = output_dir / f"{Path(input_path).stem}.md"
                pypandoc.convert_file(input_path, 'markdown', outputfile=str(md_path))
                return f"Successfully converted {input_path} to markdown. Files saved in folder: {output_dir}"
            except Exception as e:
                return f"Failed to convert DOCX to markdown: {str(e)}"
                
        elif input_format == "md" and output_format in ["docx", "pdf"]:
            try:
                # For MD to DOCX/PDF, the file is saved at the same location as the input file but with .docx/.pdf extension
                print(f"Attempting pandoc conversion from {input_path} to {output_path}", file=sys.stderr)
                pypandoc.convert_file(input_path, output_format, outputfile=output_path)
                print("Pandoc conversion completed successfully", file=sys.stderr)
                return f"Successfully converted {input_path} to {output_path} (saved in the same directory as the input file)"
            except Exception as e:
                print(f"Pandoc conversion failed with error: {e}", file=sys.stderr)
                return f"Failed to convert markdown to {output_format}: {str(e)}"
                
        else:
            return f"Unsupported conversion: {input_format} to {output_format}"
        
    except Exception as e:
        return f"Error: {str(e)}" 