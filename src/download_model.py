"""Utility: download a HuggingFace sentence-transformers model and save locally.

Usage:
    python download_model.py --model sentence-transformers/all-MiniLM-L6-v2 --out ./models/all-MiniLM-L6-v2
"""
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id (sentence-transformers/...)")
    parser.add_argument("--out", required=True, help="Local directory to save model")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print("Please install sentence-transformers: pip install sentence-transformers")
        raise

    print(f"Loading model {args.model} (this will download weights)...")
    model = SentenceTransformer(args.model)
    print(f"Saving model to {args.out}...")
    model.save(args.out)
    print("Done.")

if __name__ == "__main__":
    main()
