#!/usr/bin/env python3
"""
Generate precomputed seed traces and embeddings for each embedding backend.

Saves:
  - certificates/seed_traces_{backend}.json  (raw traces)
  - certificates/seed_embeddings_{backend}.npz  (numpy .npz with arrays: embeddings, texts)
"""

from __future__ import annotations
import json
import os
import argparse
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assessor.kickoff import run_episode
from certificates.make_certificate import compute_certificate

# Local embedding helpers (mirror embed_trace_steps but explicit and canonical)
def embed_sentence_transformers(texts, model_name="all-MiniLM-L6-v2"):
    if os.environ.get("ESM_FAST_EMBEDDINGS"):
        rng = np.random.default_rng(0)
        arr = rng.standard_normal((len(texts), 384))
        arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
        return arr
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        print(f"sentence-transformers unavailable: {exc}. Falling back to TF-IDF.")
        return embed_tfidf(texts)
    model = SentenceTransformer(model_name)
    arr = model.encode(texts, normalize_embeddings=True)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(len(texts), -1)
    return arr

def embed_tfidf(texts, max_features=512):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vect = TfidfVectorizer(ngram_range=(1,2), max_features=max_features)
    X = vect.fit_transform(texts).toarray().astype(float)
    # normalize rows
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = X / norms
    return X

def embed_openai(texts, model="text-embedding-3-small"):
    try:
        import openai
    except Exception:
        raise RuntimeError("openai package not installed or OPENAI_API_KEY missing")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot make OpenAI embeddings")
    # Modern OpenAI client if available
    if hasattr(openai, "OpenAI"):
        client = openai.OpenAI()
        resp = client.embeddings.create(model=model, input=texts)
        arr = np.array([item.embedding for item in resp.data], dtype=float)
    else:
        resp = openai.Embedding.create(model=model, input=texts)
        arr = np.array([item["embedding"] for item in resp["data"]], dtype=float)
    # normalize
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
    return arr

def extract_texts_from_trace(trace):
    # trace is a list of dicts, we embed step.text or step["text"]
    texts = []
    for step in trace:
        t = step.get("text", "")
        texts.append(str(t))
    return texts

def main():
    parser = argparse.ArgumentParser(description="Generate seed traces & embeddings per backend")
    parser.add_argument("--episodes-dir", type=Path, default=Path("demo_swe/episodes"), help="episodes dir")
    parser.add_argument("--outdir", type=Path, default=Path("certificates"), help="output directory")
    parser.add_argument("--backends", nargs="+", default=["sentence-transformers","tfidf","openai"], help="backends to prepare")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Find episodes
    eps = sorted(Path(args.episodes_dir).glob("*.json"))
    if not eps:
        raise SystemExit("No episodes found in demo_swe/episodes; run tools/generate_refactor_episode.py or create demo episodes.")

    episodes = [json.loads(p.read_text(encoding="utf-8")) for p in eps]

    for backend in args.backends:
        seeds = []
        embeddings_list = []
        texts_list = []
        print(f"[seed] backend={backend} trials={args.trials}")
        for i in range(min(args.trials, len(episodes))):
            ep = episodes[i % len(episodes)]
            # run episodes to produce a trace. Use deterministic run with seed variation.
            result = run_episode(ep, seed=args.seed + i, max_steps=ep.get("max_steps", 12))
            trace = result.get("trace", [])
            texts = extract_texts_from_trace(trace)
            # embed using chosen backend locally
            if backend == "sentence-transformers":
                emb = embed_sentence_transformers(texts)
            elif backend == "tfidf":
                emb = embed_tfidf(texts)
            elif backend == "openai":
                try:
                    emb = embed_openai(texts)
                except Exception as exc:
                    print(f"[seed] skipping openai backend: {exc}")
                    continue
            else:
                raise SystemExit(f"Unknown backend: {backend}")

            seeds.append(trace)
            embeddings_list.append(emb)
            texts_list.append(texts)

        # Save seeds and embeddings
        raw_path = outdir / f"seed_traces_{backend}.json"
        npz_path = outdir / f"seed_embeddings_{backend}.npz"
        # raw traces JSON contains list of traces (each trace is list of dicts)
        raw_path.write_text(json.dumps(seeds, ensure_ascii=False, indent=2), encoding="utf-8")
        # Save embeddings as .npz with arrays of variable length; we save as list-of-arrays via np.savez
        # We'll store embeddings as object arrays: each element is a (T,d) array
        np.savez_compressed(str(npz_path), embeddings=embeddings_list, texts=texts_list)
        print(f"[seed] wrote {raw_path} and {npz_path}")

if __name__ == "__main__":
    main()
