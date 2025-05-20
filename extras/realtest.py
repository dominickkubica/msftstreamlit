# agentic_sentiment_streamlit.py
"""
A minimal Streamlit pipeline demonstrating an **agentic** approach:
1. **Uploader** – drag‑and‑drop or browse to add an earnings‑call transcript (TXT or PDF).
2. **Agent #1** – LLM segments the transcript by *line of business*.
3. **Agent #2** – LLM scores each sentence (Positive / Neutral / Negative).
4. **Analytics UI** – interactive tables + charts + CSV export.

🔑 **Update 05‑08‑25**: 
- Fixed OpenAI API client usage to support v1.0+ of the library
- Added robust JSON extraction to handle malformed LLM responses
- Implemented better error handling and fallback mechanisms
- Improved prompting to get more reliable JSON responses

Required libs: `streamlit`, `pandas`, `openai`, `PyPDF2` (optional for PDFs).
"""

from __future__ import annotations
import os
import json
import re
import time
import nltk
from typing import List, Dict, Any, Optional, Union

import streamlit as st
import pandas as pd
from openai import OpenAI
from openai.types.chat import ChatCompletion

# Optional PDF support ---------------------------------------------------------
try:
    import PyPDF2
except ModuleNotFoundError:
    PyPDF2 = None

# Try to download NLTK resources (for sentence tokenization)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass  # Will fall back to simple sentence splitting if this fails

###############################################################################
# Page config & constants
###############################################################################

st.set_page_config(
    page_title="Agentic Sentiment Explorer",
    page_icon="📊",
    layout="wide",
)

SENTIMENT_TO_SCORE = {"positive": 1, "neutral": 0, "negative": -1}
MODEL_NAME = "gpt-4o-mini"  # adjust as desired
MAX_RETRIES = 3  # Number of retries for API calls
SYSTEM_PROMPT = "You are a helpful assistant that always responds with valid JSON. Never include explanations or text outside the JSON object."

###############################################################################
# Secure API‑key handling (sidebar)
###############################################################################


def get_api_key() -> str | None:
    """Return an OpenAI key from (1) session, (2) secrets, (3) env var, or user input."""

    # 1) Streamlit session — already validated in this run
    if st.session_state.get("OPENAI_API_KEY"):
        return st.session_state["OPENAI_API_KEY"]

    # 2) Streamlit *secrets* (safe even if file missing)
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", None)
        if secret_key:
            return secret_key
    except Exception:
        # Handle any potential exceptions when accessing secrets
        pass

    # 3) Environment variable
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    # 4) Prompt user (masked input)
    with st.sidebar:
        st.header("🔑 API Key Required")
        api_key_input = st.text_input("Enter your OpenAI API key", type="password", placeholder="sk-…")
        if api_key_input:
            st.session_state["OPENAI_API_KEY"] = api_key_input.strip()
            return api_key_input.strip()

    return None


# Get API key and initialize OpenAI client
openai_api_key = get_api_key()
if not openai_api_key:
    st.info("Add your OpenAI API key in the sidebar to begin.")
    st.stop()

# Initialize the OpenAI client
client = OpenAI(api_key=openai_api_key)

###############################################################################
# Helper functions for JSON extraction and text processing
###############################################################################

def extract_json(text: str) -> Union[Dict, List, None]:
    """
    Extract JSON from text, even if it's surrounded by other text.
    Handles common LLM formatting issues with JSON.
    """
    # First try to parse the text directly as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON patterns with regex
    json_pattern = r'```(?:json)?\s*({[\s\S]*?}|\[[\s\S]*?\])\s*```|({[\s\S]*?}|\[[\s\S]*?\])'
    matches = re.findall(json_pattern, text)
    
    # Check each potential match
    for match in matches:
        # Take the first non-empty group from each match
        json_str = next((m for m in match if m), "")
        
        # Clean the string - handle common LLM JSON errors
        json_str = json_str.strip()
        
        # Replace single quotes with double quotes
        json_str = re.sub(r"(?<!\w)'([^']*?)'(?!\w)", r'"\1"', json_str)
        
        # Fix trailing commas in arrays/objects
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            continue
    
    return None

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK if available, otherwise use a simple splitter."""
    if not text:
        return []
    
    try:
        # Try NLTK sentence tokenization
        return nltk.sent_tokenize(text)
    except:
        # Fallback to simple splitting on common sentence endings
        sentences = []
        for paragraph in text.split('\n'):
            if not paragraph.strip():
                continue
                
            # Split on sentence endings followed by space and capital letter
            for sent in re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph):
                if sent.strip():
                    sentences.append(sent.strip())
        
        return sentences if sentences else [text]  # Return original text as one sentence if all else fails


###############################################################################
# Helper – improved wrapper around ChatCompletion with retry logic
###############################################################################


def call_llm(messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
    """Call OpenAI API with retry logic and error handling."""
    for attempt in range(MAX_RETRIES):
        try:
            # Always use the JSON system prompt
            if messages[0]["role"] == "system":
                messages[0]["content"] = SYSTEM_PROMPT + " " + messages[0]["content"]
            else:
                messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
                
            response: ChatCompletion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"}  # Force JSON output
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                st.warning(f"API call failed, retrying ({attempt+1}/{MAX_RETRIES}): {str(e)}")
                time.sleep(2 * (attempt + 1))  # Simple exponential backoff
            else:
                st.error(f"Failed to get response after {MAX_RETRIES} attempts: {str(e)}")
                return "{}"  # Return empty JSON as fallback


###############################################################################
# Agent #1 – split transcript by business line
###############################################################################


@st.cache_data(show_spinner=False)
def agent_split_by_business(transcript: str) -> Dict[str, List[str]]:
    """Segment a transcript into distinct business lines."""
    # Handle empty transcript
    if not transcript or len(transcript.strip()) < 50:
        st.warning("Transcript is too short to analyze.")
        return {"General": [transcript]}
        
    # Split transcript into sentences for better analysis
    all_sentences = split_into_sentences(transcript)
    if not all_sentences:
        return {"General": [transcript]}
        
    # Limit transcript length for API call
    transcript_preview = "\n".join(all_sentences[:200])  # Limit to first 200 sentences
    
    prompt = (
        "You are a financial analyst. Analyze the earnings call transcript below.\n"
        "Identify the distinct business lines discussed (e.g. Cloud, Gaming, Advertising).\n\n"
        "IMPORTANT: Respond with ONLY a JSON object. Format: { \"Business Line Name\": [\"Sentence 1\", \"Sentence 2\", ...] }\n\n"
        "For transcripts where business lines are unclear, use \"General\" as the business line.\n"
        "Include only sentences that clearly belong to each business line."
    )
    messages = [
        {"role": "system", "content": "You are a financial analyst who responds with valid JSON."},
        {"role": "user", "content": prompt + "\n\nTRANSCRIPT:\n" + transcript_preview},
    ]
    
    with st.spinner("Identifying business lines..."):
        content = call_llm(messages)
    
    # Extract and parse JSON using our robust function
    result = extract_json(content)
    
    # Validate result is a dictionary with business lines
    if not result or not isinstance(result, dict):
        st.warning("Failed to identify business lines. Using 'General' as fallback.")
        return {"General": all_sentences}
    
    # Ensure values are lists of strings
    valid_segments = {}
    for business, sentences in result.items():
        if not sentences:
            continue
            
        if not isinstance(sentences, list):
            # Convert to a list if it's not already
            if isinstance(sentences, str):
                valid_segments[business] = [sentences]
            continue
            
        # Keep only string values in the list
        valid_sentences = [s for s in sentences if isinstance(s, str) and s.strip()]
        if valid_sentences:
            valid_segments[business] = valid_sentences
    
    # If no valid segments, use all sentences under "General"
    if not valid_segments:
        return {"General": all_sentences}
    
    return valid_segments


###############################################################################
# Agent #2 – sentence‑level sentiment scoring
###############################################################################


@st.cache_data(show_spinner=False)
def agent_sentiment(lines: List[str], business: str) -> List[Dict[str, Any]]:
    """Score sentiment for each sentence in a business segment."""
    # Handle empty input
    if not lines:
        return []
    
    # If too many lines, process in chunks
    chunk_size = 20  # Process 20 sentences at a time
    if len(lines) > chunk_size:
        all_results = []
        chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            chunk_results = agent_sentiment(chunk, business)
            all_results.extend(chunk_results)
            
        return all_results
    
    # Join lines with a separator
    joined = "\n".join(lines)
    
    prompt = (
        f"Analyze the sentiment of these sentences from the '{business}' business segment.\n"
        "Classify each as 'positive', 'neutral', or 'negative'.\n\n"
        "RESPONSE FORMAT: Return ONLY a JSON array of objects with 'text' and 'sentiment' keys.\n"
        "Example: [{\"text\": \"Revenue increased by 20%\", \"sentiment\": \"positive\"}, ...]\n\n"
        "SENTENCES TO ANALYZE:"
    )
    
    messages = [
        {"role": "system", "content": "You analyze sentiment and respond with valid JSON."},
        {"role": "user", "content": prompt + "\n" + joined},
    ]
    
    with st.spinner(f"Scoring sentiment for {business} segment..."):
        content = call_llm(messages)
    
    # Extract JSON from response
    result = extract_json(content)
    
    # Validate structure and create fallback
    if not result or not isinstance(result, list):
        st.warning(f"{business}: Invalid response format. Using neutral fallback.")
        return [{"text": t, "sentiment": "neutral", "business_line": business} for t in lines]
    
    # Validate and fix each row
    fixed_rows = []
    for i, r in enumerate(result):
        if not isinstance(r, dict) or "text" not in r or "sentiment" not in r:
            # Use original text if available, otherwise use placeholder
            text = lines[i] if i < len(lines) else f"Text {i}"
            fixed_rows.append({
                "text": text, 
                "sentiment": "neutral",
                "business_line": business
            })
        else:
            # Normalize sentiment to expected values
            sentiment = r.get("sentiment", "").lower()
            if sentiment not in SENTIMENT_TO_SCORE:
                sentiment = "neutral"  # Default fallback
                
            fixed_rows.append({
                "text": r.get("text", lines[i] if i < len(lines) else f"Text {i}"),
                "sentiment": sentiment,
                "business_line": business
            })
    
    # If somehow we got more rows than input lines, truncate
    if len(fixed_rows) > len(lines):
        fixed_rows = fixed_rows[:len(lines)]
    
    # If we got fewer rows than input lines, add the missing ones with neutral sentiment
    if len(fixed_rows) < len(lines):
        processed_texts = [r["text"] for r in fixed_rows]
        for i, line in enumerate(lines):
            if line not in processed_texts:
                fixed_rows.append({
                    "text": line,
                    "sentiment": "neutral",
                    "business_line": business
                })
    
    return fixed_rows


###############################################################################
# Text preprocessing helper
###############################################################################

def preprocess_text(text: str) -> str:
    """Clean up text for better analysis."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Replace special characters that might interfere with JSON
    text = text.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
    
    return text.strip()


###############################################################################
# PDF and text extraction
###############################################################################

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF with better error handling."""
    if PyPDF2 is None:
        st.error("PyPDF2 not installed – upload .txt or run: pip install pypdf2")
        return ""
        
    try:
        with st.spinner("Extracting text from PDF..."):
            reader = PyPDF2.PdfReader(pdf_file)
            text = []
            for page in reader.pages:
                try:
                    text.append(page.extract_text())
                except Exception as e:
                    st.warning(f"Error extracting text from page: {str(e)}")
            
            return "\n".join([t for t in text if t])
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""


###############################################################################
# Main UI – uploader ➜ agents ➜ charts
###############################################################################

st.title("📈 Agentic Sentiment Explorer")

st.markdown("""
Upload an earnings call transcript to analyze sentiment by business line.
The app uses OpenAI's models to segment the transcript and score each sentence.
""")

uploaded = st.file_uploader(
    "Drag‑and‑drop or browse for a transcript (TXT or PDF)…",
    type=["txt", "pdf"],
)

if uploaded:
    # 1️⃣ Load transcript text --------------------------------------------------
    try:
        if uploaded.type == "application/pdf":
            transcript = extract_text_from_pdf(uploaded)
        else:
            transcript = uploaded.read().decode("utf-8", errors="ignore")
        
        # Preprocess text
        transcript = preprocess_text(transcript)
        
        if len(transcript) < 50:
            st.warning("The uploaded file contains very little text. Results may not be useful.")
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.stop()

    # Show a collapsible section with the transcript preview
    with st.expander("Raw Transcript Preview"):
        st.code(transcript[:1000] + ("…" if len(transcript) > 1000 else ""))
        st.text(f"Total characters: {len(transcript)}")

    # 2️⃣ Agent #1 – segment ----------------------------------------------------
    segments = agent_split_by_business(transcript)
    
    st.success(f"Identified {len(segments)} business line(s).")
    
    # Show segments in an expander
    with st.expander("View Identified Business Segments"):
        for biz, sentences in segments.items():
            st.subheader(biz)
            st.text(f"{len(sentences)} sentences")
            if sentences:
                st.markdown("**Sample sentences:**")
                for s in sentences[:3]:  # Show just a few sample sentences
                    st.markdown(f"- {s}")

    # 3️⃣ Agent #2 – sentiment scoring -----------------------------------------
    progress_bar = st.progress(0)
    all_rows: List[Dict] = []
    
    for i, (biz, sentences) in enumerate(segments.items()):
        if not sentences:
            continue
        
        segment_rows = agent_sentiment(sentences, biz)
        all_rows.extend(segment_rows)
        
        # Update progress
        progress_bar.progress((i + 1) / len(segments))
    
    # 4️⃣ DataFrame & numeric score -------------------------------------------
    if not all_rows:
        st.warning("No sentiment data was generated. The transcript might be too short or unclear.")
        st.stop()
    
    df = pd.DataFrame(all_rows)
    
    # Ensure sentiment column exists and is lowercase
    if "sentiment" not in df.columns:
        df["sentiment"] = "neutral"  # Default fallback
    else:
        df["sentiment"] = df["sentiment"].str.lower()
    
    # Map sentiment to numeric score
    df["score"] = df["sentiment"].map(SENTIMENT_TO_SCORE)
    
    # Fill missing scores with neutral (0)
    df["score"] = df["score"].fillna(0)

    # 5️⃣ Visualizations --------------------------------------------------------
    st.subheader("Sentiment Analysis Results")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Detail", "Summary Charts", "Export"])
    
    with tab1:
        # Filter controls
        st.subheader("Filter by Sentiment")
        sentiment_filter = st.multiselect(
            "Select sentiment types to display:",
            options=["positive", "neutral", "negative"],
            default=["positive", "neutral", "negative"]
        )
        
        if sentiment_filter:
            filtered_df = df[df["sentiment"].isin(sentiment_filter)]
            st.dataframe(filtered_df, use_container_width=True)
        else:
            st.info("Select at least one sentiment type to display results.")
    
    with tab2:
        # Business line sentiment chart
        st.subheader("Average Sentiment by Business Line (-1 → Neg, +1 → Pos)")
        business_sentiment = df.groupby("business_line")["score"].mean().sort_values()
        st.bar_chart(business_sentiment)
        
        # Sentiment distribution counts
        st.subheader("Sentiment Distribution by Business Line")
        counts = df.groupby(["business_line", "sentiment"]).size().unstack(fill_value=0)
        st.bar_chart(counts)
        
        # Display summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            positive_count = len(df[df["sentiment"] == "positive"])
            st.metric("Positive Sentences", positive_count)
        
        with col2:
            neutral_count = len(df[df["sentiment"] == "neutral"])
            st.metric("Neutral Sentences", neutral_count)
        
        with col3:
            negative_count = len(df[df["sentiment"] == "negative"])
            st.metric("Negative Sentences", negative_count)
    
    with tab3:
        # CSV download ----------------------------------------------------------
        st.subheader("Export Results")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV results",
            csv,
            file_name="sentiment_by_business.csv",
            mime="text/csv",
        )
        
        # Show data summary
        st.subheader("Data Summary")
        st.write(df.describe(include="all"))
else:
    st.info("⬆️ Upload a transcript to get started.")