import os
import json
import time
import logging
import sqlite3
import zlib
from datetime import datetime
import argparse

# Setup argument parser for parallel sharding
parser = argparse.ArgumentParser()
parser.add_argument('--shard', type=int, default=None, help='Shard ID to process (0 to num-shards - 1)')
parser.add_argument('--num-shards', type=int, default=1, help='Total number of shards')
args, unknown = parser.parse_known_args()

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
# Upgraded to Gemma 2 2B IT (Q6_K_L Quantization)
MODEL_PATH = "./models/Qwen2.5-14B-Instruct-Q5_K_M.gguf"
MAX_ARTICLES_TO_PROCESS = 50
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
if DB_PATH:
    DB_PATH = DB_PATH.strip()

def get_db_connection():
    db_url = os.environ.get('SATYA_DB_URL')
    db_token = os.environ.get('SATYA_DB_TOKEN')
    
    if db_url:
        db_url = db_url.strip()
    if db_token:
        db_token = db_token.strip()
        
    if db_url and (db_url.startswith('libsql://') or db_url.startswith('https://')):
        try:
            import libsql
            # Replace libsql:// with https:// to prevent InvalidUriChar error in libsql Rust wrapper
            normalized_url = db_url.replace("libsql://", "https://")
            return libsql.connect(database=normalized_url, auth_token=db_token)
        except ImportError:
            logging.error("libsql package not installed. Falling back to local sqlite3.")
            
    import sqlite3
    return sqlite3.connect(DB_PATH)

# ==============================================================================
# --- AI INFERENCE SETUP ---
# ==============================================================================
def load_llm():
    from llama_cpp import Llama
    logging.info(f"Loading Qwen model from {MODEL_PATH}...")
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
    prompt = f"""<|im_start|>system
You are a professional editor. Rewrite the news article as a concise news summary.<|im_end|>
<|im_start|>user
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
{content}<|im_end|>
<|im_start|>assistant
"""
    
    response = llm(
        prompt,
        max_tokens=200, 
        top_p=0.9,
        stop=["<|im_end|>", "Article:", "<|im_start|>"], 
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
# --- MAIN PIPELINE ---
# ==============================================================================
def main():
    start_time = time.time()
    logging.info("--- Starting News Rephrasing Pipeline ---")
    
    shard = args.shard if args.shard is not None else (int(os.environ.get('SHARD_ID')) if os.environ.get('SHARD_ID') is not None else None)
    num_shards = args.num_shards if args.num_shards != 1 else (int(os.environ.get('NUM_SHARDS')) if os.environ.get('NUM_SHARDS') is not None else 1)

    logging.info("Fetching un-rephrased articles from database...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        if shard is not None and num_shards > 1:
            logging.info(f"Running in shard mode: shard {shard} of {num_shards}")
            cursor.execute(
                "SELECT id, title, content FROM articles WHERE status = 'scraped' AND (id % ?) = ? ORDER BY id DESC LIMIT ?",
                (num_shards, shard, MAX_ARTICLES_TO_PROCESS)
            )
        else:
            cursor.execute(
                "SELECT id, title, content FROM articles WHERE status = 'scraped' ORDER BY id DESC LIMIT ?",
                (MAX_ARTICLES_TO_PROCESS,)
            )
        rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        logging.critical(f"Failed to query articles: {e}")
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
            max_db_retries = 3
            for db_attempt in range(max_db_retries):
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("UPDATE articles SET status = 'skipped_short' WHERE id = ?", (article_id,))
                    conn.commit()
                    conn.close()
                    break
                except Exception as e_upd:
                    err_msg = str(e_upd).lower()
                    if "stream not found" in err_msg or "404" in err_msg or "connection" in err_msg:
                        logging.warning(f"Failed to update skipped status due to timeout (attempt {db_attempt + 1}/{max_db_retries}): {e_upd}. Retrying in 2s...")
                        try:
                            conn.close()
                        except Exception:
                            pass
                        time.sleep(2.0)
                    else:
                        logging.error(f"Failed to update skipped status: {e_upd}")
                        break
            continue
        logging.info(f"Processing: {title}")
        
        try:
            if llm is None:
                llm = load_llm()
                
            rephrased = None
            try:
                rephrased = rephrase_article(llm, content)
            except Exception as inner_e:
                err_msg = str(inner_e).lower()
                if "context window" in err_msg or "token" in err_msg or "exceed" in err_msg:
                    max_chars = 12000
                    if len(content) > max_chars:
                        logging.info(f"Article {article_id} exceeded context window. Retrying with smart truncation to {max_chars} chars...")
                        truncated_content = content[:max_chars] + "..."
                        rephrased = rephrase_article(llm, truncated_content)
                    else:
                        raise inner_e
                else:
                    raise inner_e
            
            if not rephrased:
                logging.error(f"Rephraser returned empty text for {title}")
                continue
                
            compressed_rephrased = zlib.compress(rephrased.encode('utf-8'))
            
            max_db_retries = 3
            for db_attempt in range(max_db_retries):
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE articles 
                        SET rephrased_article = ?, status = 'rephrased' 
                        WHERE id = ?
                    """, (compressed_rephrased, article_id))
                    conn.commit()
                    conn.close()
                    break
                except Exception as db_e:
                    err_msg = str(db_e).lower()
                    if "stream not found" in err_msg or "404" in err_msg or "connection" in err_msg:
                        logging.warning(f"Database write failed due to connection/stream timeout (attempt {db_attempt + 1}/{max_db_retries}): {db_e}. Retrying with fresh connection in 2s...")
                        try:
                            conn.close()
                        except Exception:
                            pass
                        time.sleep(2.0)
                    else:
                        raise db_e
            
            processed_count += 1
            logging.info(f"Successfully saved rephrased article {article_id}. (Total this run: {processed_count})")
            
            time.sleep(2.0)
            
        except Exception as e:
            logging.error(f"Failed to process article {article_id}: {e}")
            
    logging.info(f"--- Pipeline Finished. Processed {processed_count} new articles. ---")

if __name__ == '__main__':
    main()
