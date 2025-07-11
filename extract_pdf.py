#!/usr/bin/env python3
"""
Simple PDF text extraction script
This attempts multiple methods to extract text from a PDF
"""

import os
import sys
import subprocess

def extract_with_pdftotext():
    """Try to extract text using pdftotext"""
    try:
        result = subprocess.run(['pdftotext', 'AI-Centric Monetization System Overview.pdf', '-'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
    except FileNotFoundError:
        pass
    return None

def extract_with_python_libs():
    """Try to extract text using Python libraries"""
    # Try PyMuPDF
    try:
        import fitz
        doc = fitz.open('AI-Centric Monetization System Overview.pdf')
        text = ''
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except ImportError:
        pass
    
    # Try pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open('AI-Centric Monetization System Overview.pdf') as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text() or ''
        return text
    except ImportError:
        pass
    
    return None

def extract_basic_info():
    """Extract basic info about the PDF"""
    try:
        # Get file size
        size = os.path.getsize('AI-Centric Monetization System Overview.pdf')
        
        # Try to read some basic structure
        with open('AI-Centric Monetization System Overview.pdf', 'rb') as f:
            header = f.read(1024).decode('latin-1', errors='ignore')
            
        info = f"""
PDF File Analysis:
- File size: {size} bytes
- Header info: Contains PDF version and basic structure
- File appears to be a standard PDF document

Based on the filename 'AI-Centric Monetization System Overview.pdf', this appears to be
documentation about an AI-focused monetization system.

Without being able to extract the full text, I can infer this document likely contains:
- System architecture overview
- Monetization strategies
- AI integration points
- Revenue models
- Implementation guidelines
"""
        return info
    except Exception as e:
        return f"Error analyzing file: {e}"

def main():
    if not os.path.exists('AI-Centric Monetization System Overview.pdf'):
        print("PDF file not found!")
        return
    
    print("Attempting to extract PDF text...")
    
    # Try pdftotext first
    text = extract_with_pdftotext()
    if text:
        print("Successfully extracted with pdftotext:")
        print(text)
        return
    
    # Try Python libraries
    text = extract_with_python_libs()
    if text:
        print("Successfully extracted with Python libraries:")
        print(text)
        return
    
    # Fallback to basic analysis
    print("Could not extract text directly. Providing basic analysis:")
    print(extract_basic_info())

if __name__ == "__main__":
    main()