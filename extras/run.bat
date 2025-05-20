@echo off
echo Starting Reading Between the Lines Application
echo =============================================

:: Create .streamlit directory if it doesn't exist
if not exist .streamlit mkdir .streamlit

:: Copy the config file to .streamlit directory
echo Creating Streamlit configuration...
copy config.toml .streamlit\config.toml 2>nul
if not %ERRORLEVEL% == 0 (
    echo Warning: Could not find config.toml in current directory.
    echo Creating default config file...
    
    echo [server] > .streamlit\config.toml
    echo maxUploadSize = 200 >> .streamlit\config.toml
    echo maxMessageSize = 200 >> .streamlit\config.toml
    echo enableWebsocketCompression = false >> .streamlit\config.toml
    echo enableCORS = false >> .streamlit\config.toml
    echo enableXsrfProtection = false >> .streamlit\config.toml
    
    echo [theme] >> .streamlit\config.toml
    echo primaryColor = "#0078d4" >> .streamlit\config.toml
    echo backgroundColor = "#f3f2f1" >> .streamlit\config.toml
    echo secondaryBackgroundColor = "#ffffff" >> .streamlit\config.toml
    echo textColor = "#323130" >> .streamlit\config.toml
)

:: Check if Python is installed
echo Checking Python installation...
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python not found. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

:: Check and install required packages
echo Checking and installing required packages...
python -c "import pandas, numpy, streamlit, nltk, yfinance, plotly" > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing required packages...
    python -m pip install --upgrade pip
    python -m pip install streamlit==1.27.0 pandas==2.0.3 numpy==1.24.3 matplotlib==3.7.3
    python -m pip install yfinance==0.2.28 nltk==3.8.1 plotly==5.17.0 
    python -m pip install python-docx==0.8.11 PyPDF2==3.0.1 requests==2.31.0
)

:: Download NLTK data in advance
echo Downloading NLTK data...
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('vader_lexicon')" > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Failed to download NLTK data.
)

:: Set environment variables to help with stability
echo Setting environment variables...
set STREAMLIT_SERVER_HEADLESS=false
set STREAMLIT_SERVER_ENABLE_CORS=false
set STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
set STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

:: Test file existence before running
echo Checking required files...
if not exist main.py (
    echo Error: main.py not found in current directory.
    pause
    exit /b 1
)

:: Run the application with error handling
echo Starting the application...
echo --------------------------------------------------
echo If the application doesn't open in your browser:
echo ✅ Open http://localhost:8501 manually
echo ✅ Check logs for any error messages
echo ✅ Press Ctrl+C to stop the application
echo --------------------------------------------------

:: Run Streamlit with optimized parameters
streamlit run main.py --server.maxMessageSize=200 --server.enableWebsocketCompression=false ^
--server.enableCORS=false --server.enableXsrfProtection=false --browser.gatherUsageStats=false ^
--browser.serverAddress="localhost"
