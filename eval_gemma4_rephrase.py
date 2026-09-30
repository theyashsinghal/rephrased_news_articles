import argparse
import json
import logging
import os
import re
import sys
import time
from llama_cpp import Llama

# Mock unused cloud dependencies if needed
from unittest.mock import MagicMock
sys.modules.setdefault('gspread', MagicMock())
sys.modules.setdefault('oauth2client', MagicMock())
sys.modules.setdefault('oauth2client.service_account', MagicMock())
sys.modules.setdefault('libsql', MagicMock())

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_rephrased_text(raw_text):
    if not raw_text:
        return ""
    # Strip thought channel if present (Gemma 4 internal reasoning)
    if "<channel|>" in raw_text:
        raw_text = raw_text.split("<channel|>")[-1]
    raw_text = re.sub(r'<\|channel\|?>.*?<channel\|?>', '', raw_text, flags=re.DOTALL).strip()
    
    # Strip leading preambles like "Summary:", "Here is the summary:", etc.
    raw_text = re.sub(r'^(?:Summary|Here is (?:a|the) summary|Paragraph):\s*', '', raw_text, flags=re.IGNORECASE)

    # Join lines into single paragraph
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    cleaned = " ".join(lines)
    
    # Strip wrapping quotes
    while (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()

    # Ensure clean full-sentence ending
    if cleaned and cleaned[-1] not in '.!?"\'':
        last_sentence_end = max(
            cleaned.rfind('.'),
            cleaned.rfind('!'),
            cleaned.rfind('?')
        )
        if last_sentence_end != -1:
            cleaned = cleaned[:last_sentence_end + 1]

    return cleaned

def build_rephrase_prompt(content):
    return f"""<|turn>user
You are a news editor who writes concise, factual summaries. You follow formatting rules exactly.

Summarize the article below as a single paragraph of 50–60 words.

The summary must capture ALL key facts of the article: who, what, when, where, why, and the outcome or impact. Do not skip any important detail, number, or development mentioned in the article. Prefer dropping minor background details over dropping core facts.

Requirements:
- Plain paragraph only: no headline, no title, no bullet points, no preamble like "Here is a summary".
- Bold key people and organizations with **double asterisks** on first mention only.
- Use only facts stated in the article. Do not infer, speculate, or add outside context.
- Neutral, journalistic tone.
- Finish with a complete sentence.

<article>
{content}
</article><turn|>
<|turn>model
"""

def main():
    parser = argparse.ArgumentParser(description="Gemma 4 12B Sharded Rephraser Benchmark")
    parser.add_argument("--shard", type=int, default=0, help="Shard index (0 to num_shards - 1)")
    parser.add_argument("--num-shards", type=int, default=15, help="Total number of parallel shards")
    parser.add_argument("--model-repo", default="unsloth/gemma-4-12b-it-GGUF", help="HuggingFace model repo")
    parser.add_argument("--model-file", default="gemma-4-12b-it-Q4_K_M.gguf", help="GGUF model filename")
    parser.add_argument("--sample-file", default="eval/sample_60_rephrase.json", help="Path to sample articles")
    parser.add_argument("--output-file", default=None, help="Output JSON path")
    args = parser.parse_args()

    if not args.output_file:
        args.output_file = f"eval_rephrase_results_shard_{args.shard}.json"

    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, args.model_file)

    if not os.path.exists(model_path):
        logging.info(f"Downloading {args.model_file} from {args.model_repo} via HuggingFace...")
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=args.model_repo,
            filename=args.model_file,
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )

    logging.info(f"Loading Gemma 4 12B from {model_path} with 4 threads...")
    t_load = time.time()
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_batch=512,
        n_threads=4,
        verbose=False
    )
    logging.info(f"Model loaded in {time.time() - t_load:.2f}s")

    with open(args.sample_file, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    # Shard slicing: partition articles evenly across num_shards (15 shards x 4 samples = 60)
    samples = [art for i, art in enumerate(all_samples) if i % args.num_shards == args.shard]
    logging.info(f"Runner Shard {args.shard}/{args.num_shards}: processing {len(samples)} articles (IDs: {[s['id'] for s in samples]})")

    results = []
    total_latency = 0.0

    for idx, article in enumerate(samples):
        aid = article["id"]
        title = article["title"]
        category = article.get("category", "")
        content = article["original_content"]
        prev_rephrased = article.get("prev_rephrased", "")
        prev_words = article.get("prev_word_count", len(prev_rephrased.split()))

        t0 = time.time()
        try:
            prompt = build_rephrase_prompt(content)
            response = llm(
                prompt,
                max_tokens=220,
                top_p=0.9,
                stop=["<turn|>", "<|turn>", "<eos>"],
                temperature=0.25,
                repeat_penalty=1.1,
                echo=False
            )
            raw_text = response['choices'][0].get('text', '').strip()
            g4_rephrased = clean_rephrased_text(raw_text)
            g4_words = len(g4_rephrased.split())
            dur = time.time() - t0
            total_latency += dur

            # Extract bolded entities:
            bolded = re.findall(r'\*\*(.*?)\*\*', g4_rephrased)

            item = {
                "id": aid,
                "title": title,
                "category": category,
                "prev_rephrased": prev_rephrased,
                "prev_word_count": prev_words,
                "gemma4_rephrased": g4_rephrased,
                "gemma4_word_count": g4_words,
                "bolded_entities": bolded,
                "latency_seconds": round(dur, 2)
            }
            results.append(item)
            logging.info(f"[Shard {args.shard} | {idx+1}/{len(samples)}] #{aid} ({category}) | G4 Words: {g4_words} (Prev: {prev_words}) | {dur:.2f}s")
        except Exception as e:
            logging.error(f"Error on article #{aid}: {e}")
            results.append({
                "id": aid,
                "title": title,
                "category": category,
                "prev_rephrased": prev_rephrased,
                "prev_word_count": prev_words,
                "error": str(e)
            })

    avg_lat = (total_latency / len(samples)) if samples else 0.0

    summary = {
        "shard": args.shard,
        "num_shards": args.num_shards,
        "model_name": "gemma-4-12b",
        "total_articles": len(samples),
        "total_latency_seconds": round(total_latency, 2),
        "avg_latency_seconds": round(avg_lat, 2),
        "articles": results
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Shard {args.shard} finished: {len(samples)} articles processed in {total_latency:.1f}s")

if __name__ == "__main__":
    main()
