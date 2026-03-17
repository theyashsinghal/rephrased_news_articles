import os
import json
import time
import logging
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from llama_cpp import Llama

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
SOURCE_SHEET_NAME = 'News Scrapper AI'
SOURCE_WORKSHEET_NAME = 'Sheet1'

DEST_SHEET_NAME = 'News Scrapper AI Processed'
DEST_WORKSHEET_NAME = 'Sheet1'

MODEL_PATH = "./models/gemma-2b-it-q6_k_m.gguf"
MAX_ARTICLES_TO_PROCESS = 200
MAX_RUNTIME_SECONDS = 5 * 3600  # 5.5 hours to prevent GitHub Actions timeout

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==============================================================================
# --- GOOGLE SHEETS SETUP ---
# ==============================================================================
def connect_to_sheets():
    logging.info("Connecting to Google Sheets...")
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    gcp_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    
    if not gcp_json:
        raise ValueError("GCP_SERVICE_ACCOUNT_JSON missing from environment variables!")
        
    creds_dict = json.loads(gcp_json)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    
    source_sheet = client.open(SOURCE_SHEET_NAME).worksheet(SOURCE_WORKSHEET_NAME)
    
    try:
        dest_sheet = client.open(DEST_SHEET_NAME).worksheet(DEST_WORKSHEET_NAME)
    except gspread.exceptions.SpreadsheetNotFound:
        logging.critical(f"Destination sheet '{DEST_SHEET_NAME}' not found. Please create it.")
        raise
        
    return source_sheet, dest_sheet

def get_existing_urls(dest_sheet):
    logging.info("Fetching existing URLs from destination sheet to build cache...")
    existing_urls = set()
    try:
        raw_data = dest_sheet.col_values(1)
        for cell in raw_data:
            if not cell:
                continue
            try:
                data = json.loads(cell)
                if 'url' in data:
                    existing_urls.add(data['url'])
            except json.JSONDecodeError:
                continue
    except Exception as e:
        logging.error(f"Error fetching destination sheet data: {e}")
        
    logging.info(f"Loaded {len(existing_urls)} existing URLs from destination.")
    return existing_urls

# ==============================================================================
# --- AI INFERENCE SETUP ---
# ==============================================================================
def load_llm():
    logging.info(f"Loading Gemma model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
    # n_ctx is the context window. 2048 is plenty for 100-word outputs.
    # n_threads=2 matches the standard GitHub Runner specs.
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_batch=512,
        n_threads=4,
        verbose=False 
    )
    logging.info("Model loaded successfully.")
    return llm

def rephrase_article(llm, content):
    prompt = f"""<start_of_turn>user
Rewrite the news article below as a concise news summary.

Rules:
1. In ONLY 50 to 60 words.
2. One paragraph only.
3. No headline or title.
4. Bold (**...**) important people or organizations on first mention.
5. Do NOT add information that is not present in the article.
6. Clear, factual, journalistic tone.
7. Always end with a complete sentence. NEVER stop mid-sentence.

Article:
{content}
<end_of_turn>
<start_of_turn>model
"""
    
    response = llm(
        prompt,
        max_tokens=200, 
        top_p = 0.9,
        stop=["<end_of_turn>", "Article:"], 
        temperature=0.25,
        repeat_penalty = 1.1,
        echo=False
    )
    
    rephrased_text = response['choices'][0].get('text', '').strip()

    if rephrased_text and rephrased_text[-1] not in '.!?"\'':
        last_sentence_end = max(
            rephrased_text.rfind('.'),
            rephrased_text.rfind('!'),
            rephrased_text.rfind('?')
        )
        if last_sentence_end != -1:
            rephrased_text = rephrased_text[:last_sentence_end + 1]

    return rephrased_text

# ==============================================================================
# --- MAIN PIPELINE ---
# ==============================================================================
def main():
    start_time = time.time()
    logging.info("--- Starting News Rephrasing Pipeline ---")
    
    source_sheet, dest_sheet = connect_to_sheets()
    existing_urls = get_existing_urls(dest_sheet)
    
    logging.info("Fetching latest records from source sheet...")
    raw_source_data = source_sheet.col_values(1)
    
    # Get the latest MAX_ARTICLES_TO_PROCESS
    articles_to_check = raw_source_data[-MAX_ARTICLES_TO_PROCESS:] if len(raw_source_data) > MAX_ARTICLES_TO_PROCESS else raw_source_data
    
    parsed_articles = []
    for cell in articles_to_check:
        if not cell:
            continue
        try:
            parsed_articles.append(json.loads(cell))
        except json.JSONDecodeError:
            continue
            
    logging.info(f"Evaluating {len(parsed_articles)} articles for processing...")
    
    llm = None
    processed_count = 0
    
    for article in parsed_articles:
        # 1. Global Timeout Check
        if time.time() - start_time > MAX_RUNTIME_SECONDS:
            logging.warning("Approaching maximum GitHub Actions runtime. Halting execution gracefully.")
            break
            
        url = article.get('url')
        if not url:
            continue
            
        # 2. Cache Check
        if url in existing_urls:
            continue
            
        # 3. Lazy Load LLM (Only load into RAM if we actually have work to do)
        if llm is None:
            llm = load_llm()
            
        content = article.get('content', '')
        title = article.get('title', 'Unknown Title')
        
        if len(content.split()) < 20:
            logging.warning(f"Skipping {title} - content too short.")
            continue
            
        logging.info(f"Processing: {title}")
        
        try:
            # 4. Generate Rephrase
            rephrased = rephrase_article(llm, content)
            
            # 5. Append new key
            article['rephrased_article'] = rephrased
            article['rephrased_at'] = str(datetime.now())
            
            # 6. Save to Destination Sheet
            safe_json = json.dumps(article)
            dest_sheet.append_row([safe_json])
            
            existing_urls.add(url)
            processed_count += 1
            logging.info(f"Successfully saved rephrased article. (Total this run: {processed_count})")
            
            # Rate limiting for Google Sheets API
            time.sleep(2.0)
            
        except Exception as e:
            logging.error(f"Failed to process article {url}: {e}")
            
    logging.info(f"--- Pipeline Finished. Processed {processed_count} new articles. ---")

if __name__ == '__main__':
    main()
