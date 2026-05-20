"""
collect_responses.py — version basse mémoire (Transformers 4-bit, sans vLLM).

Compatible avec la même commande qu'avant. Sur Colab : redémarrer le runtime
avant de lancer (sinon la VRAM reste occupée par un ancien chargement vLLM).

    python scripts/collect_responses.py \
        --model_path mistralai/Mistral-7B-v0.3 \
        --data_path data/TriviaQA/rc.nocontext/train-00000-of-00001.parquet \
        --output_path results/Mistral-7B-v0.3/responses_test.jsonl \
        --sample_num 10 --max_samples 10000 --judge_type rule --gpu_util 0.85
"""

import os
import gc
import json
import re
import string
import argparse
import time
from typing import List, Set, Iterator, Dict, Any

import torch
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from openai import OpenAI

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BACKEND = "transformers-4bit"  # vérifiez ce message au démarrage


def parse_args():
    parser = argparse.ArgumentParser(description="Consistency Filtering with Rule or LLM Judge.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model for sampling")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the TriviaQA parquet file")
    parser.add_argument("--output_path", type=str, default="data/consistency_samples.jsonl", help="Output path")

    parser.add_argument("--sample_num", type=int, default=10, help="Samples per question")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of questions to process")
    parser.add_argument(
        "--gpu_util",
        type=float,
        default=0.75,
        help="Fraction max de VRAM pour le modèle (laisser de la marge pour la génération)",
    )
    parser.add_argument("--tp_size", type=int, default=None, help="(ignoré, conservé pour compatibilité)")

    parser.add_argument("--judge_type", type=str, choices=["rule", "llm"], default="rule", help="How to judge correctness")
    parser.add_argument("--api_key", type=str, default=None, help="API key for LLM Judge")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1", help="API base URL")
    parser.add_argument("--judge_model", type=str, default="gpt-4o", help="Model name for LLM Judge")

    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--max_input_tokens", type=int, default=512, help="Tronque les prompts trop longs")
    return parser.parse_args()


# ==========================================
# Utilities
# ==========================================

def normalize_answer(s: str) -> str:
    """Standardize answer strings for Rule Judge."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "''´`")
        return "".join(ch if ch not in exclude else " " for ch in text)

    if not s:
        return ""
    return white_space_fix(remove_articles(handle_punc(str(s).lower().replace("_", " ")))).strip()


def load_existing_qids(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    qids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                qids.update(data.keys())
            except json.JSONDecodeError:
                continue
    return qids


def iter_parquet_rows(data_path: str, max_samples: int | None) -> Iterator[Dict[str, Any]]:
    """Lit le parquet par petits lots (évite de charger 10k lignes en RAM)."""
    pf = pq.ParquetFile(data_path)
    count = 0
    for batch in pf.iter_batches(batch_size=32):
        records = batch.to_pydict()
        n = len(next(iter(records.values())))
        for i in range(n):
            if max_samples is not None and count >= max_samples:
                return
            yield {col: records[col][i] for col in records}
            count += 1


def _gpu_memory_cap_gib(gpu_util: float) -> str:
    """Réserve de la VRAM pour les pics pendant generate()."""
    if not torch.cuda.is_available():
        return "0GiB"
    total = torch.cuda.get_device_properties(0).total_memory
    # Ne pas dépasser 80 % ni la valeur demandée ; marge pour KV cache
    fraction = min(max(gpu_util, 0.5), 0.80)
    gib = int(total * fraction / (1024**3))
    return f"{max(gib - 1, 4)}GiB"


def _clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ==========================================
# Consistency Sampler (Transformers, basse mémoire)
# ==========================================

class ConsistencySampler:
    SUFFIX = "Respond with the answer only, without any explanation."
    UNCERTAIN_TERMS = ("don't know", "cannot", "not provided", "no information")

    def __init__(self, args):
        self.args = args
        print(f"[collect_responses] backend={BACKEND}  (si vous voyez vLLM, le mauvais fichier est exécuté)")

        _clear_cuda()

        self.use_cuda = torch.cuda.is_available()
        self.model_device = torch.device("cuda:0") if self.use_cuda else torch.device("cpu")

        print(f"Chargement 4-bit : {args.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: Dict[str, Any] = dict(
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        if self.use_cuda:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = "auto"
            load_kwargs["max_memory"] = {
                0: _gpu_memory_cap_gib(args.gpu_util),
                "cpu": "48GiB",
            }
        else:
            load_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
        self.model.eval()
        # Désactive le cache HF interne (économise VRAM pendant generate)
        self.model.config.use_cache = False

        self.judge_client = None
        if args.judge_type == "llm":
            if not args.api_key:
                raise ValueError("API Key is required for LLM Judge.")
            self.judge_client = OpenAI(api_key=args.api_key, base_url=args.base_url)

        _clear_cuda()
        if self.use_cuda:
            alloc = torch.cuda.memory_allocated() / 1024**3
            print(f"VRAM après chargement : {alloc:.2f} GiB alloués")

    def _model_input_device(self):
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def _generate_one(self, question: str) -> str:
        prompt = f"{question.strip()} {self.SUFFIX}"
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            truncation=True,
            max_length=self.args.max_input_tokens,
        )
        dev = self._model_input_device()
        input_ids = input_ids.to(dev)

        try:
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.args.max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                top_k=50,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=False,
            )
            new_tokens = output_ids[0, input_ids.shape[1] :]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        finally:
            del input_ids
            _clear_cuda()

    def rule_judge(self, response: str, norm_gts: List[str]) -> str:
        norm_res = normalize_answer(response)
        for gt in norm_gts:
            if gt and gt in norm_res:
                return "true"
        return "false"

    def llm_judge(self, question: str, response: str, answer_list: List[str]) -> str:
        prompt = (
            f"Question: {question}\n"
            f"Response: {response}\n"
            f"Correct Answers: {answer_list}\n"
            f"Please judge whether the response is correct or not. "
            f"Return 't' if the response is correct, and 'f' if the response is incorrect. "
            f"Don't add any additional information."
        )
        for attempt in range(5):
            try:
                completion = self.judge_client.chat.completions.create(
                    model=self.args.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                res = completion.choices[0].message.content.strip().lower()
                if "t" in res:
                    return "true"
                if "f" in res:
                    return "false"
            except Exception as e:
                print(f"Judge API failed (attempt {attempt+1}): {e}")
                time.sleep(1)
        return "error"

    @staticmethod
    def _extract_aliases(item: Dict[str, Any]) -> List[str]:
        raw_aliases = []
        answer = item.get("answer") or {}
        for col in ["aliases", "normalized_aliases"]:
            val = answer.get(col)
            if not val:
                continue
            if isinstance(val, list):
                raw_aliases.extend(val)
            else:
                raw_aliases.append(str(val))
        return list(set(a for a in raw_aliases if a))

    def process_data(self):
        os.makedirs(os.path.dirname(self.args.output_path) or ".", exist_ok=True)
        processed_qids = load_existing_qids(self.args.output_path)

        all_correct_count = 0
        all_incorrect_count = 0

        with open(self.args.output_path, "a", encoding="utf-8") as f:
            row_iter = iter_parquet_rows(self.args.data_path, self.args.max_samples)
            for item in tqdm(row_iter, desc=f"Sampling ({self.args.judge_type} judge)"):
                qid = str(item.get("question_id", ""))
                if qid in processed_qids:
                    continue

                question = item.get("question", "")
                if not question or "answer" not in item:
                    continue

                raw_aliases = self._extract_aliases(item)
                norm_gts = [normalize_answer(a) for a in raw_aliases]
                if not norm_gts:
                    continue

                suffix = self.SUFFIX
                responses = []
                judges = []
                judge_cache = {}

                for _ in range(self.args.sample_num):
                    try:
                        ans = self._generate_one(question)
                        responses.append(ans)

                        uncertain_terms = list(self.UNCERTAIN_TERMS)
                        if any(term in ans.lower() for term in uncertain_terms):
                            judges.append("uncertain")
                            continue

                        if self.args.judge_type == "rule":
                            judges.append(self.rule_judge(ans, norm_gts))
                        else:
                            if ans not in judge_cache:
                                judge_cache[ans] = self.llm_judge(question, ans, raw_aliases)
                            judges.append(judge_cache[ans])

                    except Exception as e:
                        print(f"Sampling error at {qid}: {e}")
                        break

                if len(responses) < self.args.sample_num:
                    continue

                true_count = judges.count("true")
                if true_count == self.args.sample_num:
                    all_correct_count += 1
                elif true_count == 0:
                    all_incorrect_count += 1

                result = {
                    qid: {
                        "question": f"{question.strip()} {suffix}",
                        "responses": responses,
                        "judges": judges,
                        "ground_truth": list(set(raw_aliases)),
                    }
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                processed_qids.add(qid)

                if len(processed_qids) % 10 == 0:
                    tqdm.write(f"Stats -> All-Correct: {all_correct_count}, All-Incorrect: {all_incorrect_count}")


if __name__ == "__main__":
    args = parse_args()
    sampler = ConsistencySampler(args)
    sampler.process_data()
