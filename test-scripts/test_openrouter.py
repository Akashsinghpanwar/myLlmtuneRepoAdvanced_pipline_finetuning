#!/usr/bin/env python3
"""
Test script for OpenRouter API integration.
This script tests the OpenRouter API with a simple question rephrasing request.
"""

import os
import requests
import json
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Get API key from environment
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("Error: OPENROUTER_API_KEY not found in environment")
    print("Make sure to export OPENROUTER_API_KEY or set it in the .env file")
    exit(1)

print(f"API key found: {api_key[:8]}...{api_key[-4:]}")

# Test question
test_question = "What is the specific rule regarding players concealing the Touch Rugby ball under their attire?"

# Set up headers with all required fields for OpenRouter
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/TrelisResearch/ADVANCED-fine-tuning",  # Required by OpenRouter
    "X-Title": "ADVANCED-fine-tuning Question Rephraser"  # Helpful for OpenRouter to identify your app
}

# System and user messages
system_message = """
You are an expert at rephrasing questions while maintaining their exact meaning.
Your task is to rephrase the given question in a different way, but ensure that:
1. The rephrased question asks for exactly the same information
2. The rephrased question has the same level of specificity
3. The rephrased question maintains the same context and domain knowledge requirements

Provide ONLY the rephrased question, with no additional text, explanations, or formatting.
"""

user_message = f"""
Original question: {test_question}

Rephrase this question while maintaining its exact meaning.
"""

# Payload for the API request
payload = {
    "model": "google/gemini-2.0-flash-001",
    "messages": [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ],
    "temperature": 0.7,
    "max_tokens": 2000
}

print("\nSending request to OpenRouter API...")
try:
    # Make the API request
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    # Check if the request was successful
    response.raise_for_status()
    
    # Parse the response
    response_data = response.json()
    print("\nAPI Response Status:", response.status_code)
    print("Response Headers:", json.dumps(dict(response.headers), indent=2))
    
    # Extract and print the rephrased question
    rephrased_question = response_data["choices"][0]["message"]["content"].strip()
    print("\nOriginal Question:", test_question)
    print("Rephrased Question:", rephrased_question)
    
    # Print full response for debugging
    print("\nFull Response Data:")
    print(json.dumps(response_data, indent=2))
    
    print("\nTest completed successfully!")
    
except requests.exceptions.HTTPError as e:
    print(f"\nHTTP Error: {e}")
    print(f"Response Status Code: {e.response.status_code}")
    print(f"Response Text: {e.response.text}")
except Exception as e:
    print(f"\nError: {str(e)}")
