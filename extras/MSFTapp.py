# config.py
# Configuration settings for the Financial Transcript Analyzer

import os
from typing import Optional

# Debug mode
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

# API Keys (set as environment variables or replace with your actual keys)
YAHOO_FINANCE_API_KEY: Optional[str] = os.environ.get("YAHOO_FINANCE_API_KEY")

# Azure OpenAI Configuration (for ChatGPT)
AZURE_OPENAI_API_KEY: Optional[str] = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT: Optional[str] = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME: str = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-35-turbo")

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
                return MockResponse("Positive")
            elif "negative" in prompt.lower():
                return MockResponse("Negative")
            else:
                return MockResponse("Neutral")
    
    def __init__(self):
        self.chat = self.ChatCompletions()

# Initialize clients
try:
    if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
        # In production, use actual OpenAI client
        from openai import AzureOpenAI
        
        openai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2023-12-01-preview",
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
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