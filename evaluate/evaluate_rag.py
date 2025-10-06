import os
import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

sys.path.append(str(Path(__file__).parent.parent / "src"))

from search_service import MultimodalSearchService
from answer_generation_service import AnswerGenerationService
from deepeval.models import GPTModel

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,   
    AnswerRelevancyMetric,  
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)


def run_evaluation(
    testset_path: str = "evaluate/testset.csv",
    max_samples: int = None,
    k_text: int = 5,
    n_articles: int = 3
):
    """Run RAG evaluation with DeepEval using OpenAI."""
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set in .env file")

    evaluator_model = GPTModel(model="o3-mini")
    
    print(f"Loading testset from {testset_path}")
    testset_path = Path(testset_path)
    
    if testset_path.suffix == '.csv':
        df = pd.read_csv(testset_path)
    elif testset_path.suffix == '.json':
        df = pd.read_json(testset_path)
    else:
        raise ValueError(f"Unsupported file format: {testset_path.suffix}. Use .csv or .json")
    
    if max_samples:
        df = df.head(max_samples)
        print(f"Evaluating on first {max_samples} samples")
    
    print(f"Loaded {len(df)} test cases\n")
    
    print("Initializing RAG services...")
    search_service = MultimodalSearchService()
    text_loaded, _ = search_service.load_all_indexes()
    
    if not text_loaded:
        raise ValueError("Text index not loaded. Run: python src/create_indexes.py")
    
    answer_service = AnswerGenerationService()
    print("RAG services ready\n")
    
    test_cases = []
    detailed_results = []
    
    print("="*80)
    print("RUNNING RAG ON TEST QUERIES")
    print("="*80)
    
    for idx, row in df.iterrows():
        query = row['user_input']
        print(f"\n[{idx+1}/{len(df)}] Query: {query[:100]}")
        
        try:
            search_results = search_service.search_multimodal(
                query=query,
                k_text=k_text,
                k_images=0
            )
            
            contexts = []
            for result in search_results.get('text', []):
                contexts.append(result.get('full_content', result.get('content', '')))
            
            answer_result = answer_service.generate_answer_with_summary(
                query=query,
                search_results=search_results,
                n_articles=n_articles,
                n_images=0
            )
            
            response = answer_result['answer']
            print(f"  Retrieved: {len(contexts)} contexts")
            print(f"  Answer: {response[:150]}...")
            
            expected_output = row.get('reference', '') if 'reference' in row else None
            
            test_case = LLMTestCase(
                input=query,
                actual_output=response,
                retrieval_context=contexts,
                expected_output=expected_output if expected_output and str(expected_output) != 'nan' else None
            )
            
            test_cases.append(test_case)
            
            detailed_results.append({
                'query': query,
                'response': response,
                'num_contexts': len(contexts),
                'sources_used': answer_result['sources_used']
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    print(f"\n{'='*80}")
    print(f"Successfully processed {len(test_cases)} queries")
    print("="*80)
    
    print("\nEVALUATING METRICS...")
    print("This will take several minutes...\n")
    
    metrics_to_evaluate = [
        ("Faithfulness", FaithfulnessMetric(threshold=0.5, include_reason=False, model=evaluator_model)),
        ("Answer Relevancy", AnswerRelevancyMetric(threshold=0.5, include_reason=False, model=evaluator_model)),
        ("Contextual Relevancy", ContextualRelevancyMetric(threshold=0.5, include_reason=False, model=evaluator_model)),
        ("Contextual Precision", ContextualPrecisionMetric(threshold=0.5, include_reason=False, model=evaluator_model)),
        ("Contextual Recall", ContextualRecallMetric(threshold=0.5, include_reason=False, model=evaluator_model)),
    ]
    
    all_scores = {name: [] for name, _ in metrics_to_evaluate}
    
    for i, test_case in enumerate(test_cases):
        print(f"Evaluating {i+1}/{len(test_cases)}...", end='\r')
        
        for metric_name, metric in metrics_to_evaluate:
            try:
                metric.measure(test_case)
                all_scores[metric_name].append(metric.score)
            except Exception as e:
                print(f"\nError evaluating {metric_name} for case {i+1}: {e}")
                all_scores[metric_name].append(0.0)
    
    metrics = {
        name.lower().replace(' ', '_'): 
        sum(scores) / len(scores) if scores else 0
        for name, scores in all_scores.items()
    }
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    for metric_name, score in metrics.items():
        print(f"{metric_name:.<50} {score:.4f}")
    print("="*80)
    
    print("\nINTERPRETATION:")
    print("-"*80)
    
    interpretations = {
        'faithfulness': [
            (0.8, "Excellent - answers are well-grounded in context"),
            (0.6, "Good - answers mostly use retrieved context"),
            (0.4, "Fair - some hallucination occurring"),
            (0.0, "Needs work - significant hallucination")
        ],
        'answer_relevancy': [
            (0.8, "Excellent - answers directly address questions"),
            (0.6, "Good - answers are generally relevant"),
            (0.4, "Fair - answers sometimes miss the point"),
            (0.0, "Needs work - answers often off-topic")
        ],
        'contextual_relevancy': [
            (0.8, "Excellent - retrieved docs are highly relevant"),
            (0.6, "Good - most retrieved docs are relevant"),
            (0.4, "Fair - retrieval needs improvement"),
            (0.0, "Needs work - poor retrieval quality")
        ],
        'contextual_precision': [
            (0.8, "Excellent - relevant docs ranked at top"),
            (0.6, "Good - ranking mostly accurate"),
            (0.4, "Fair - ranking could be better"),
            (0.0, "Needs work - poor ranking")
        ],
        'contextual_recall': [
            (0.8, "Excellent - all needed info retrieved"),
            (0.6, "Good - most needed info retrieved"),
            (0.4, "Fair - missing some information"),
            (0.0, "Needs work - missing key information")
        ],
    }
    
    for metric_name, score in metrics.items():
        if metric_name in interpretations:
            for threshold, interpretation in interpretations[metric_name]:
                if score >= threshold:
                    print(f"{metric_name} {score:.2f}: {interpretation}")
                    break
    
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = Path("evaluate")
    output_folder.mkdir(exist_ok=True)
    
    individual_scores = []
    for i in range(len(test_cases)):
        score_dict = {'query': detailed_results[i]['query']}
        for metric_name, scores in all_scores.items():
            score_dict[metric_name.lower().replace(' ', '_')] = scores[i]
        individual_scores.append(score_dict)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'num_samples': len(test_cases),
        'individual_scores': individual_scores
    }
    
    json_path = output_folder / f"eval_results_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    for i, result in enumerate(detailed_results):
        for metric_name, scores in all_scores.items():
            result[metric_name.lower().replace(' ', '_')] = scores[i]
    
    csv_path = output_folder / f"eval_results_{timestamp}.csv"
    pd.DataFrame(detailed_results).to_csv(csv_path, index=False)
    
    print(f"\nResults saved:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG with DeepEval")
    parser.add_argument("--testset", default="evaluate/testset.json", help="Path to testset (CSV or JSON)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--k-text", type=int, default=7, help="Number of docs to retrieve")
    parser.add_argument("--n-articles", type=int, default=4, help="Number of docs for answer generation")
    
    args = parser.parse_args()
    
    try:
        results = run_evaluation(
            testset_path=args.testset,
            max_samples=args.max_samples,
            k_text=args.k_text,
            n_articles=args.n_articles
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)