"""
Configuration settings for the Financial Transcript Analyzer.
This file handles API configurations and client initialization.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Debug mode
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

# API Keys (set as environment variables)
YAHOO_FINANCE_API_KEY: Optional[str] = os.environ.get("YAHOO_FINANCE_API_KEY")

# OpenAI Configuration (standard OpenAI, not Azure)
OPENAI_API_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

# Mock client for development/testing
class MockOpenAIClient:
    """
    Mock OpenAI client for development/testing when no actual API key is available.
    In production, replace this with actual API client initialization.
    """
    class ChatCompletions:
        @staticmethod
        def create(**kwargs):
            class MockResponse:
                class Choice:
                    class Message:
                        def __init__(self, content):
                            self.content = content
                    
                    def __init__(self, content):
                        self.message = self.Message(content)
                
                def __init__(self, content):
                    self.choices = [self.Choice(content)]
            
            # Simple rule-based response for demonstration
            prompt = kwargs.get("messages", [{}])[-1].get("content", "")
            
            if "positive" in prompt.lower():
                return MockResponse("Positive sentiment detected in this text. The language suggests optimism and confidence about future performance.")
            elif "negative" in prompt.lower():
                return MockResponse("Negative sentiment detected in this text. The language indicates concerns and cautious outlook about future performance.")
            else:
                return MockResponse("Neutral sentiment detected in this text. The language is factual and balanced without strong positive or negative indicators.")
    
    def __init__(self):
        self.chat = self.ChatCompletions()

# Initialize OpenAI client
try:
    if OPENAI_API_KEY:
        # In production, use actual OpenAI client
        from openai import OpenAI
        
        openai_client = OpenAI(
            api_key=OPENAI_API_KEY
        )
        if DEBUG:
            print(f"[CONFIG] Initialized OpenAI client with model: {OPENAI_MODEL}")
    else:
        openai_client = MockOpenAIClient()
        if DEBUG:
            print("[CONFIG] Using MockOpenAIClient (no API key provided)")
except ImportError:
    openai_client = MockOpenAIClient()
    if DEBUG:
        print("[CONFIG] Using MockOpenAIClient (OpenAI package not installed)")
except Exception as e:
    openai_client = MockOpenAIClient()
    if DEBUG:
        print(f"[CONFIG] Error initializing OpenAI client: {e}")
        print("[CONFIG] Using MockOpenAIClient as fallback")

def get_sentiment_analysis(text):
    """
    Analyze sentiment of financial text using OpenAI API
    
    Args:
        text (str): The text to analyze
        
    Returns:
        str: Sentiment analysis result
    """
    try:
        if isinstance(openai_client, MockOpenAIClient):
            # Use mock client's simplified response
            response = openai_client.chat.create(
                messages=[{"role": "user", "content": text}]
            )
        else:
            # Use actual OpenAI API
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in earnings call transcript analysis. Provide a sentiment analysis of the following text from a financial earnings call."},
                    {"role": "user", "content": text}
                ],
                max_tokens=150
            )
        
        # Extract the content from the response
        result = response.choices[0].message.content
        return result
    
    except Exception as e:
        if DEBUG:
            print(f"[ERROR] Sentiment analysis failed: {e}")
        return "Error performing sentiment analysis"