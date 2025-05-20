"""
Script to set up the environment with API keys securely.
Run this script once to configure your environment.
"""

import os
import getpass
from pathlib import Path
from dotenv import set_key

def setup_environment():
    """Create or update .env file with user-provided API keys"""
    
    # Determine the location of the .env file
    env_path = Path(".env")
    
    # Check if .env file exists, create it if not
    if not env_path.exists():
        env_path.touch()
        print(f"Created new .env file at {env_path.absolute()}")
    else:
        print(f"Updating existing .env file at {env_path.absolute()}")
    
    # Configure debug mode
    debug_mode = input("Enable debug mode? (yes/no, default: no): ").lower().strip()
    debug_value = "True" if debug_mode in ("yes", "y", "true") else "False"
    set_key(env_path, "DEBUG", debug_value)
    
    # Configure OpenAI API
    print("\n--- OpenAI API Configuration ---")
    print("Your OpenAI API key will be stored in the .env file.")
    openai_key = getpass.getpass("Enter your OpenAI API key (input will be hidden): ")
    
    if openai_key:
        set_key(env_path, "OPENAI_API_KEY", openai_key)
        print("OpenAI API key saved successfully.")
    else:
        print("No OpenAI API key provided. The application will use mock responses.")
    
    # Configure OpenAI model
    openai_model = input("OpenAI model to use (default: gpt-3.5-turbo): ").strip()
    if not openai_model:
        openai_model = "gpt-3.5-turbo"
    set_key(env_path, "OPENAI_MODEL", openai_model)
    
    # Configure Yahoo Finance API (optional)
    print("\n--- Yahoo Finance API Configuration (Optional) ---")
    yahoo_key = getpass.getpass("Enter your Yahoo Finance API key (optional, input will be hidden): ")
    
    if yahoo_key:
        set_key(env_path, "YAHOO_FINANCE_API_KEY", yahoo_key)
        print("Yahoo Finance API key saved successfully.")
    else:
        print("No Yahoo Finance API key provided.")
    
    print("\nEnvironment setup complete!")
    print("You can edit the .env file directly at any time to update these values.")
    print("IMPORTANT: Never commit the .env file to version control!")

if __name__ == "__main__":
    setup_environment()