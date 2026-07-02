import os
import json
import time
import logging
import sqlite3
import zlib
from datetime import datetime

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
# Upgraded to Gemma 2 2B IT (Q6_K_L Quantization)
MODEL_PATH = "./models/gemma-2-2b-it-Q6_K_L.gguf"
MAX_ARTICLES_TO_PROCESS = 100
MAX_RUNTIME_SECONDS = 5 * 3600

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==============================================================================
# --- DATABASE CONFIGURATION ---
# ==============================================================================
def load_env():
    # Check both current directory and parent directory for .env
    env_paths = [
        os.path.join(os.path.dirname(__file__), '.env'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    ]
    for env_path in env_paths:
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, val = line.split('=', 1)
                        os.environ[key.strip()] = val.strip()

load_env()

# Self-healing default local path for environments like GHA
default_db_path = '/Users/mac/Downloads/Code/Satya/satya.db'
if not os.path.exists(os.path.dirname(default_db_path)):
    default_db_path = os.path.join(os.path.dirname(__file__), 'satya.db')

DB_PATH = os.environ.get('SATYA_DB_PATH', default_db_path)

def get_db_connection():
    db_url = os.environ.get('SATYA_DB_URL')
    db_token = os.environ.get('SATYA_DB_TOKEN')
    
    if db_url and (db_url.startswith('libsql://') or db_url.startswith('https://')):
        try:
            import libsql
            return libsql.connect(database=db_url, auth_token=db_token)
        except ImportError:
            logging.error("libsql package not installed. Falling back to local sqlite3.")
            
    import sqlite3
    return sqlite3.connect(DB_PATH)

# ==============================================================================
# --- AI INFERENCE SETUP ---
# ==============================================================================
def load_llm():
    from llama_cpp import Llama
    logging.info(f"Loading Gemma 2 2B model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
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
        top_p=0.9,
        stop=["<end_of_turn>", "Article:"], 
        temperature=0.25,
        repeat_penalty=1.1,
        echo=False
    )
    
    rephrased_text = response['choices'][0].get('text', '').strip()

    # Ensure clean full-sentence ending
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
# --- REPHRASED HEADLINE GENERATION & VALIDATION ---
# ==============================================================================

TITLE_PROMPT = """<start_of_turn>user
You write headlines for SatyaDheesh, an Indian news platform.
Rewrite the headline for the article below.

RULES:
1. ONE headline only. Maximum 12 words. No quotes around it.
2. Punchy and direct — strong active verbs, lead with the most striking fact.
3. Every name, number, and fact MUST come from the article below. Never invent or exaggerate.
4. If someone CLAIMS or PROMISES something, keep it as their claim:
   "Modi vows 2 crore jobs" — never "2 crore jobs coming".
5. No question headlines. No "this is why / here's what" teasers.
6. Numbers make headlines stronger — use them when the article has them.

GOOD: "Yogi vows to end mafia raj in UP"
GOOD: "11 dead in Bondi Beach mass shooting, gunman tackled by bystander"
GOOD: "Kerala polls: UDF storms back to power in Kochi"
BAD:  "You won't believe what happened in Kerala" (teaser)
BAD:  "2 crore jobs created every year" (drops attribution)

ORIGINAL TITLE: {title}
ARTICLE: {body_snippet}

HEADLINE:
<end_of_turn>
<start_of_turn>model
"""

def generate_rephrased_title(llm, original_title, body_snippet):
    prompt = TITLE_PROMPT.format(title=original_title, body_snippet=body_snippet)
    
    response = llm(
        prompt,
        max_tokens=50, 
        top_p=0.9,
        stop=["<end_of_turn>", "ORIGINAL TITLE:"], 
        temperature=0.2,
        repeat_penalty=1.1,
        echo=False
    )
    
    return response['choices'][0].get('text', '').strip()

import re
import string

def validate_rephrased_title(generated_title, original_title, body):
    gen_title = generated_title.strip()
    
    # 1. Check length
    words = gen_title.split()
    if len(words) < 3 or len(words) > 12:
        return False, f"length {len(words)} out of bounds [3, 12]"
        
    # 2. Check leading/trailing quotes
    if gen_title.startswith('"') or gen_title.endswith('"') or gen_title.startswith("'") or gen_title.endswith("'"):
        return False, "contains leading or trailing quotes"
        
    # Helper to check if a word (proper noun or number) is in text
    def clean_word(w):
        return "".join(c for c in w if c.isalnum()).lower()
        
    # Normalize number strings for comparison
    def normalize_numbers(text):
        text_lower = text.lower()
        text_lower = re.sub(r'(\d+)\s+(crore|lakh|million|billion|percent|pct)', r'\1\2', text_lower)
        return text_lower

    norm_gen = normalize_numbers(gen_title)
    norm_orig = normalize_numbers(original_title)
    norm_body = normalize_numbers(body)

    # 3. Extract proper nouns (all capitalized words, including the first word)
    punctuation = string.punctuation
    proper_nouns = []
    for idx, w in enumerate(words):
        cleaned_w = w.strip(punctuation)
        if not cleaned_w:
            continue
        if cleaned_w[0].isupper():
            proper_nouns.append(cleaned_w)
            
    # 4. Extract all numbers (including normalized units)
    numbers = re.findall(r'\b\d+(?:crore|lakh|million|billion|percent|pct)?\b', norm_gen)
    
    # Verify proper nouns are in original title or body
    for pn in proper_nouns:
        pn_clean = clean_word(pn)
        if not pn_clean:
            continue
        if pn_clean not in norm_orig and pn_clean not in norm_body:
            return False, f"proper noun '{pn}' not found in source"

    # Verify numbers are in original title or body
    for num in numbers:
        pattern = r'\b' + re.escape(num) + r'\b'
        if not re.search(pattern, norm_orig) and not re.search(pattern, norm_body):
            return False, f"number '{num}' not found in source"
            
    return True, None

# ==============================================================================
# --- MAIN PIPELINE ---
# ==============================================================================
def main():
    start_time = time.time()
    logging.info("--- Starting News Rephrasing Pipeline ---")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
    except Exception as e:
        logging.critical(f"Failed to connect to database: {e}")
        return
        
    logging.info("Fetching un-rephrased articles from database...")
    try:
        cursor.execute("SELECT id, title, content FROM articles WHERE status = 'scraped' ORDER BY id DESC LIMIT ?", (MAX_ARTICLES_TO_PROCESS,))
        rows = cursor.fetchall()
    except Exception as e:
        logging.critical(f"Failed to query articles: {e}")
        conn.close()
        return
        
    logging.info(f"Evaluating {len(rows)} articles for rephrasing...")
    
    llm = None
    processed_count = 0
    
    for r in rows:
        if time.time() - start_time > MAX_RUNTIME_SECONDS:
            logging.warning("Approaching maximum GitHub Actions runtime. Halting execution gracefully.")
            break
            
        article_id = r[0]
        title = r[1]
        compressed_content = r[2]
        
        try:
            content = zlib.decompress(compressed_content).decode('utf-8') if compressed_content else ""
        except Exception as e:
            logging.error(f"Failed to decompress content for article {article_id}: {e}")
            continue
            
        if len(content.split()) < 20:
            logging.warning(f"Skipping {title} - content too short.")
            try:
                cursor.execute("UPDATE articles SET status = 'skipped_short' WHERE id = ?", (article_id,))
                conn.commit()
            except Exception as e_upd:
                logging.error(f"Failed to update skipped status: {e_upd}")
            continue
        logging.info(f"Processing: {title}")
        
        try:
            if llm is None:
                llm = load_llm()
                
            rephrased = None
            rephrased_title = None
            try:
                rephrased = rephrase_article(llm, content)
                if rephrased:
                    body_snippet = content[:1500]
                    rephrased_title = generate_rephrased_title(llm, title, body_snippet)
                    
                    is_valid, reject_reason = validate_rephrased_title(rephrased_title, title, content)
                    if not is_valid:
                        # Log validation failure: article_id, reason, generated text
                        logging.warning(f"[Validation Failed] Article ID {article_id} title rejected: {reject_reason}. Generated: '{rephrased_title}'")
                        rephrased_title = None
            except Exception as inner_e:
                err_msg = str(inner_e).lower()
                if "context window" in err_msg or "token" in err_msg or "exceed" in err_msg:
                    max_chars = 12000
                    if len(content) > max_chars:
                        logging.info(f"Article {article_id} exceeded context window. Retrying with smart truncation to {max_chars} chars...")
                        truncated_content = content[:max_chars] + "..."
                        rephrased = rephrase_article(llm, truncated_content)
                        if rephrased:
                            body_snippet = truncated_content[:1500]
                            rephrased_title = generate_rephrased_title(llm, title, body_snippet)
                            is_valid, reject_reason = validate_rephrased_title(rephrased_title, title, content)
                            if not is_valid:
                                logging.warning(f"[Validation Failed] Article ID {article_id} title rejected: {reject_reason}. Generated: '{rephrased_title}'")
                                rephrased_title = None
                    else:
                        raise inner_e
                else:
                    raise inner_e
            
            if not rephrased:
                logging.error(f"Rephraser returned empty text for {title}")
                continue
                
            compressed_rephrased = zlib.compress(rephrased.encode('utf-8'))
            
            cursor.execute("""
                UPDATE articles 
                SET rephrased_article = ?, rephrased_title = ?, status = 'rephrased' 
                WHERE id = ?
            """, (compressed_rephrased, rephrased_title, article_id))
            conn.commit()
            
            processed_count += 1
            logging.info(f"Successfully saved rephrased article {article_id}. (Total this run: {processed_count})")
            
            time.sleep(2.0)
            
        except Exception as e:
            logging.error(f"Failed to process article {article_id}: {e}")
            
    conn.close()
    logging.info(f"--- Pipeline Finished. Processed {processed_count} new articles. ---")

if __name__ == '__main__':
    main()
