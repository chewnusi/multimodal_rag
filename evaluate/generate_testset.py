import os
from pathlib import Path
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset, Golden
import pandas as pd
import json
import random
from datetime import datetime

random.seed(datetime.now().timestamp())

synthesizer = Synthesizer(model="o3-mini")

docs_folder = Path("data/txt")
doc_paths = [str(p) for p in docs_folder.glob("*.txt")]

print(f"Found {len(doc_paths)} documents")
print("Generating test cases...")

num_docs_to_sample = 15
sampled_doc_paths = random.sample(doc_paths, min(num_docs_to_sample, len(doc_paths)))

print(f"Randomly selected {len(sampled_doc_paths)} documents for test generation")
print("\nGenerating test cases...")

goldens = synthesizer.generate_goldens_from_docs(
    document_paths=sampled_doc_paths,
    max_goldens_per_context=1,
)

print(f"Generated {len(goldens)} test cases")

test_cases = []
for golden in goldens:
    test_cases.append({
        "user_input": golden.input,
        "reference": golden.expected_output if hasattr(golden, 'expected_output') else "",
        "context": str(golden.context) if hasattr(golden, 'context') else ""
    })

output_folder = Path("evaluate")
output_folder.mkdir(exist_ok=True)

json_path = output_folder / "testset.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(test_cases, f, indent=2, ensure_ascii=False)
print(f"Saved testset to {json_path}")

csv_path = output_folder / "testset.csv"
df = pd.DataFrame(test_cases)
df.to_csv(csv_path, index=False)
print(f"Also saved as CSV: {csv_path}")

docs_list_path = output_folder / "sampled_documents.txt"
with open(docs_list_path, 'w') as f:
    for path in sampled_doc_paths:
        f.write(f"{Path(path).name}\n")
print(f"Saved list of sampled documents: {docs_list_path}")

print("\nReady to evaluate! Run:")
print("  python3 evaluate/evaluate_rag.py --max-samples 5")