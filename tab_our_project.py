import streamlit as st

def render_our_project_tab(microsoft_colors):
    # Create a container for the project content
    project_container = st.container()
    
    with project_container:
        # First, create columns for layout
        sidebar_col, content_col = st.columns([1, 3])
        
        # Fill the sidebar column
        with sidebar_col:
            st.markdown("<div class='sidebar-nav'>", unsafe_allow_html=True)
            st.markdown("### Project Contents")
            
            sections = [
                "Executive Summary", 
                "Problem Statement", 
                "Benchmarking", 
                "Real World Testing",
                "Results & Findings",
                "Publications"
                
            ]
            
            # Make this a session state to track which section is active
            if 'active_section' not in st.session_state:
                st.session_state.active_section = sections[0]
                
            for section in sections:
                if st.button(section, key=f"btn_{section}"):
                    st.session_state.active_section = section
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Fill the content column
        with content_col:
            st.markdown("<div class='ms-card'>", unsafe_allow_html=True)
            
            if st.session_state.active_section == "Executive Summary":
                st.markdown("## Executive Summary")
                st.markdown("""
                The Santa Clara University - Microsoft Practicum Project was established to compare various LLM's 
                and show their potential for financial sentiment analysis on earnings transcripts. We developed this tool 
                in conjunction with our findings, to showcase our methods of breaking down financial transcripts for sentiment 
                analysis. 
                
                The project addresses a crucial gap in financial analysis by automating the extraction of sentiment 
                signals from earnings calls and quantifying their relationship with market movements. Using natural 
                language processing and machine learning techniques, we've built a system that can:
                
                1. Process raw earnings call transcripts
                2. Identify sentiment patterns across different segments of the call
                3. Correlate these patterns with short-term stock price changes
                4. Visualize the results for intuitive understanding
                
                Our analysis provides unique insights into important market sectors that may drive stock movement. Although markets
                are tumultuous and unpredictable, our process highlights individual business lines that may be impacting investor sentiment
                with greater impact than others. We hope this tool can provide Microsoft with more information to better navigate the delicate 
                process of presenting quarterly earnings.
                """)
                st.markdown("</div>", unsafe_allow_html=True)

                
            elif st.session_state.active_section == "Problem Statement":
                st.markdown("## Problem Statement")
                st.markdown("""
                Our project set out to benchmark large language models (LLMs) in their ability to perform financial sentiment analysis, comparing 
                their effectiveness against traditional machine learning methods using Python-based libraries.

                Our findings demonstrate that LLMs significantly outperform conventional models, particularly in capturing financial nuance and 
                contextual meaning within earnings transcripts.

                To validate these insights, we conducted a case study on Microsoft’s last eight quarterly earnings calls, analyzing the sentiment 
                across business segments and correlating it with same-day stock performance. Our results suggest that certain divisions, such as Search tools and Reporter Q&A
                ,have a disproportionate impact on investor sentiment and price movement.

                To make this research actionable, we developed a user-friendly tool that automates this analysis. Users can upload an earnings transcript, 
                pair it with a relevant stock ticker, and receive targeted sentiment insights across business segments—highlighting which areas 
                may have driven investor reaction.
                """)
                st.markdown("</div>", unsafe_allow_html=True)

                
            elif st.session_state.active_section == "Benchmarking":
                st.markdown("## Benchmarking")

                # two‑column layout: text on the left, figure on the right
                text_col, fig_col = st.columns([2, 3])

                with text_col:
                    st.markdown("""
                    Before diving into the Microsoft transcripts, we first benchmarked a variety of models  
                    on the Financial Phrasebank dataset (Aalto University) to gauge baseline performance.

                    **Our approach combines several components:**

                    1. **Data Collection**: Financial Phrasebank from Kaggle, with 3‑label sentiment (Positive, Negative, Neutral).  
                    2. **Preprocessing**: NLTK & SpaCy for traditional models; skipped for LLMs.  
                    3. **Sentiment Analysis**: Fin‑BERT, Vader (NLTK), TextBlob, Copilot 365, Copilot Chat & App, ChatGPT‑4o (± prompts), and Gemini 2.0 Flash.  
                    4. **Metrics**: Accuracy, Lift, plus Pearson correlation for business‑line sentiment vs. stock returns.
                    """)

                st.markdown("</div>", unsafe_allow_html=True)


            elif st.session_state.active_section == "Real World Testing":
                st.markdown("## Real World Testing")

                # two‐column layout for text
                text_col, img_col = st.columns([2, 3])

                with text_col:
                    st.markdown("**Key Findings**")
                    st.write("""
                    - **Segmented sentiment drives insight** – Business‐line sentiment (e.g. Devices, Search & Advertising) often correlates more closely to next‐day stock moves than overall tone.
                    - **Inverse signals** – Positive spikes in “Search & News Advertising” preceded sell‐offs, suggesting over‑optimism can trigger defensive selling.
                    - **Model choice matters** – ChatGPT‑4o handled batch transcript processing more cleanly than Copilot for large‐scale analysis.
                    """)

                    st.markdown("**Optimization Recommendations**")
                    st.write("""
                    1. **Automate segmentation** – Build an LLM‑powered pipeline that tags transcripts by business unit before sentiment extraction.  
                    2. **Overlay price data** – Pull stock price changes via an API (e.g. yfinance) and dynamically plot movements next to sentiment.
                    3. **Interactive visuals** – Use Streamlit’s Plotly or Altair integration to let users toggle segments and date ranges.
                    """)

                st.markdown("</div>", unsafe_allow_html=True)


            elif st.session_state.active_section == "Results & Findings":
                st.markdown("## Results & Findings")
                st.markdown("""
                
                **Key Findings**
                - LLMs outpace traditional NLP: ChatGPT‑4o and Copilot App excel in accuracy and nuance. This is the future of Financial Sentiment Analysis
                - Copilot 365 offloads to TextBlob—limit exposed by reduced performance

                **Copilot Optimization Recommendations**
                1. **Transparency**: Notify users when analysis falls back to simpler libraries
                2. **API Access**: Enable enterprise-tier LLM API for Copilot App for batch workflows
                3. **Performance**: Expose token limits and allow prompt tailoring in Copilot 365
                """
                )
                
                st.markdown("</div>", unsafe_allow_html=True)
    
            
            elif st.session_state.active_section == "Publications":
                st.markdown("## Publications")

                text_col, link_col = st.columns([3, 1])

                with text_col:
                    st.markdown("""
                    We’ve been actively sharing our findings through a variety of channels:

                    1. **Microsoft Internal Report** (April 2025)  
                    – Detailed technical write‑up of Copilot 365 vs. LLM benchmarking, circulated within Microsoft Research.

                    2. **Microsoft AI Blog Submission** (May 2025)  
                    – Draft blog post titled *“Benchmarking Copilot & ChatGPT‑4o for Financial Sentiment Analysis”* under review by the Microsoft AI content team.

                    3. **MIT Technology Review Manuscript** (Under Review)  
                    – Paper **“Can AI Read Between the Lines? Benchmarking LLMs on Financial Transcripts”** submitted for external publication.
                    """)
                st.markdown("</div>", unsafe_allow_html=True)

    
