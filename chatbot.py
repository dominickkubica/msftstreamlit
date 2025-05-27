def clean_text(text):
    """
    Clean text by removing non-ASCII characters and fixing formatting issues.
    """
    # Normalize Unicode characters to their ASCII equivalents where possible
    text = unicodedata.normalize("NFKD", text)
    
    # Replace common Unicode punctuation with ASCII equivalents
    replacements = {
        "—": "-",  # EM dash
        "–": "-",  # EN dash
        """: '"',  # Left double quote
        """: '"',  # Right double quote
        "'": "'",  # Left single quote
        "'": "'",  # Right single quote
        "…": "...",  # Ellipsis
        " ": " ",  # Non-breaking space
        "\u2013": "-",  # EN dash (by code)
        "\u2014": "-",  # EM dash (by code)
        "\u2018": "'",  # Left single quote (by code)
        "\u2019": "'",  # Right single quote (by code)
        "\u201C": '"',  # Left double quote (by code)
        "\u201D": '"',  # Right double quote (by code)
        "\u2026": "...",  # Ellipsis (by code)
        "\u00A0": " ",  # Non-breaking space (by code)
    }
    
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    
    # Clean up any remaining non-ASCII characters
    # PRESERVE newlines, tabs, and carriage returns for formatting
    def clean_char(char):
        if ord(char) < 128:
            return char
        elif char in ['\n', '\r', '\t']:  # Preserve important whitespace
            return char
        else:
            return ' '
    
    text = ''.join(clean_char(char) for char in text)
    
    # Critical: Insert spaces between digits and letters that are stuck together
    # This fixes issues like "442.33ontheearningsannouncementdayto447.20"
    # Using lookahead/lookbehind to only insert spaces at digit-letter boundaries
    text = re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', text)  # 123Word → 123 Word
    text = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', text)  # Word123 → Word 123
    
    # Ensure spaces around punctuation aren't lost
    text = re.sub(r'(\w)([,.])', r'\1 \2', text)
    text = re.sub(r'([,.])(\w)', r'\1 \2', text)
    
    # Final cleanup of multiple spaces (but preserve single newlines)
    text = re.sub(r' {2,}', ' ', text)  # Multiple spaces to single space
    text = re.sub(r'[ \t]+\n', '\n', text)  # Remove trailing spaces before newlines
    text = re.sub(r'\n[ \t]+', '\n', text)  # Remove leading spaces after newlines
    
    return text.strip()
