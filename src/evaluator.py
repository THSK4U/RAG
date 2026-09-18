"""Evaluation module for calculating Recall@k on retrieval results."""

import os
from typing import Dict, List, Optional

from .models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


def compute_iou(yours: MinimalSource, correct: MinimalSource) -> float:
    """Compute Intersection over Union (IoU) between two character spans.

    Args:
        yours: The retrieved source chunk.
        correct: The reference ground-truth source chunk.

    Returns:
        float: IoU score between 0.0 and 1.0.
    """
    start_overlap = max(yours.first_character_index, correct.first_character_index)
    end_overlap = min(yours.last_character_index, correct.last_character_index)

    intersection = max(0, end_overlap - start_overlap)
    if intersection == 0:
        return 0.0

    union = max(yours.last_character_index, correct.last_character_index) - min(
        yours.first_character_index, correct.first_character_index
    )
    if union == 0:
        return 0.0

    return intersection / union


def evaluate(
    student_search_results_path: str,
    dataset_path: str,
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Evaluate retrieval results by computing Recall@k against ground truth.

    Args:
        student_search_results_path: Path to StudentSearchResults JSON file.
        dataset_path: Path to ground-truth AnsweredQuestions JSON file.
        k_values: List of k values for recall (defaults to [1, 3, 5, 10]).

    Returns:
        Dict[str, float]: Dictionary containing Recall@k metrics.
    """

    if not os.path.isfile(student_search_results_path):
        print(f"Error: File not found: {student_search_results_path}")
        return {}

    if not os.path.isfile(dataset_path):
        print(f"Error: File not found: {dataset_path}")
        return {}

    try:
        with open(student_search_results_path, "r", encoding="utf-8") as f:
            student_data = StudentSearchResults.model_validate_json(f.read())
    except Exception as err:
        print(f"Error parsing student results: {err}")
        return {}

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            gt_data = RagDataset.model_validate_json(f.read())
    except Exception as err:
        print(f"Error parsing dataset: {err}")
        return {}

    student_map = {
        res.question_id: res.retrieved_sources for res in student_data.search_results
    }

    gt_questions = [q for q in gt_data.rag_questions if isinstance(q, AnsweredQuestion)]

    if not gt_questions:
        print("Warning: No answered questions found in dataset.")
        return {}

    k_vals = k_values or [1, 3, 5, 10]
    results: Dict[str, float] = {}
    for k in k_vals:
        scores = [
            sum(
                any(
                    r.file_path == c.file_path and compute_iou(r, c) >= 0.05
                    for r in student_map.get(q.question_id, [])[:k]
                )
                for c in q.sources
            )
            / len(q.sources)
            for q in gt_questions
        ]
        results[f"Recall@{k}"] = sum(scores) / len(scores) if scores else 0.0

    print("Student data is valid: True\nEvaluation Results\n" + "=" * 40)
    print(" ".join(f"{k}: {v:.3f}" for k, v in results.items()))
    print()
    return results
