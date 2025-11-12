"""Main runner for baseline and ACE-style evaluation pipelines."""

import time
from pathlib import Path
from typing import Dict, List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from slm_ace.ace_roles import (
    build_generator_prompt,
    build_reflector_prompt,
    parse_reflector_output_to_lessons,
    choose_lessons_for_playbook,
)
from slm_ace.config import ModelConfig
from slm_ace.metrics import compute_accuracy, compute_average_latency,semantic_answer_score,compute_token_metrics
from slm_ace.model_manager import generate
from slm_ace.playbook import Playbook


def run_dataset_baseline(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: List[Dict],
    domain: str,
    config: ModelConfig,
    model_id: str,
    task_name: str,
    mode: str = "baseline",
) -> tuple[List[Dict], Dict]:
    """
    Run baseline evaluation (no ACE) on a dataset.
    
    This function processes each example in the dataset by:
    1. Building a simple prompt (question + optional context)
    2. Generating an answer using the model
    3. Comparing answer to ground truth (exact match)
    4. Recording results with consistent schema
    
    Args:
        model: Loaded language model (from model_manager.load_model_and_tokenizer).
        tokenizer: Loaded tokenizer (from model_manager.load_model_and_tokenizer).
        dataset: List of example dicts, each with keys:
            - id (str): Example identifier
            - question (str): Question text
            - answer (str): Ground truth answer
            - context (str, optional): Additional context
            - domain (str, optional): Domain name
        domain: Domain name (e.g., "finance", "medical", "iot").
        config: ModelConfig with generation parameters (max_new_tokens, temperature, top_p).
        model_id: HuggingFace model ID (for result tracking).
        task_name: Task name (e.g., "tatqa_tiny", "medqa_tiny") for result tracking.
        mode: Evaluation mode (default: "baseline").
    
    Returns:
        Tuple of (results, summary):
        - results: List[Dict] with consistent schema. Each dict contains:
            * Core columns (always present):
              - model_id (str): Model identifier
              - task_name (str): Task identifier
              - domain (str): Domain name
              - mode (str): Evaluation mode ("baseline" or "ace")
              - sample_id (str): Example ID from dataset
              - gold (str): Ground truth answer
              - pred (str): Model prediction
              - latency_ms (float): Generation latency in milliseconds
            * Debug columns (always present):
              - question (str): Original question
              - context (str): Context if provided, else empty string
              - correct (int): 1 if exact match, 0 otherwise
        - summary: Dict with aggregate statistics:
            * accuracy (float): Overall accuracy (0.0 to 1.0)
            * avg_latency_ms (float): Average latency in milliseconds
            * num_examples (int): Number of examples processed
    """
    results = []
    predictions = []
    labels = []
    latencies = []
    
    for example in dataset:
        example_id = example.get("id", "unknown")
        question = example.get("question", "")
        context = example.get("context")
        ground_truth = example.get("answer", "")
        
        # Build simple prompt
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"
        
        # Generate answer
        start_time = time.time()
        answer = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        latency_ms = (time.time() - start_time) * 1000
        
        # Check correctness
        correct = answer.strip().lower() == ground_truth.strip().lower()
        score = semantic_answer_score(answer, ground_truth)
        prompt_tokens = len(tokenizer.encode(question, add_special_tokens=False))
        output_tokens = len(tokenizer.encode(answer, add_special_tokens=False))
        # Record result with consistent schema
        result = {
            "model_id": model_id,
            "task_name": task_name,
            "domain": domain,
            "mode": mode,
            "sample_id": example_id,
            "gold": ground_truth,
            "pred": answer,
            "latency_ms": latency_ms,
            # Additional fields for debugging
            "question": question,
            "context": context or "",
            "correct": 1 if correct else 0,
            "semantic_score":score,
            "prompt_tokens":prompt_tokens,
            "output_tokens":output_tokens
        }
        results.append(result)
        
        predictions.append(answer)
        labels.append(ground_truth)
        latencies.append(latency_ms)
    
    # Compute summary
    accuracy = compute_accuracy(predictions, labels)
    avg_latency = compute_average_latency(latencies)
    
    summary = {
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency,
        "num_examples": len(dataset),
    }
    
    return results, summary


def run_dataset_ace(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: List[Dict],
    domain: str,
    config: ModelConfig,
    playbook: Playbook,
    playbook_path: Path,
    model_id: str,
    task_name: str,
    mode: str = "ace",
    reflect_on_correct_every_n: int = 5,
    prune_every_n: int = 10,
) -> tuple[List[Dict], Dict]:
    """
    Run ACE-style adaptive evaluation on a dataset.
    
    This is ACE-inspired and not a full reproduction of the original ACE, SEAL, or SPICE algorithms.
    
    For each example, the ACE pipeline:
    1. Generator: Build prompt with playbook context → generate answer
    2. Check correctness against ground truth
    3. Reflector: If incorrect (or periodically if correct), generate lessons
    4. Curator: Add filtered lessons to playbook, record feedback
    5. Periodically prune playbook to keep top entries
    
    TODO(Sathwik): Improve ACE logic in ace_roles.py:
    - Refine generator prompts to better incorporate playbook strategies
    - Improve reflector prompts to generate more specific, actionable lessons
    - Enhance lesson filtering and deduplication
    - Consider multi-turn reflection for complex errors
    
    TODO(Archit): Extend metrics and logging:
    - Track playbook evolution over time (size, quality metrics)
    - Measure impact of playbook on accuracy improvement
    - Add per-step latency breakdown (generation vs reflection)
    - Log playbook entries added/removed during pruning
    
    Args:
        model: Loaded language model (used for both generation and reflection).
        tokenizer: Loaded tokenizer.
        dataset: List of example dicts, each with keys:
            - id (str): Example identifier
            - question (str): Question text
            - answer (str): Ground truth answer
            - context (str, optional): Additional context
            - domain (str, optional): Domain name
        domain: Domain name (used to filter playbook entries).
        config: ModelConfig with generation parameters.
        playbook: The ACE playbook (will be updated in-place).
        playbook_path: Path to save playbook after run.
        model_id: HuggingFace model ID (for result tracking).
        task_name: Task name (e.g., "tatqa_tiny") for result tracking.
        mode: Evaluation mode (default: "ace").
        reflect_on_correct_every_n: Reflect on correct answers every N examples (for learning).
        prune_every_n: Prune playbook every N examples to limit size.
        
    Returns:
        Tuple of (results, summary):
        - results: List[Dict] with consistent schema. Each dict contains:
            * Core columns (always present):
              - model_id (str): Model identifier
              - task_name (str): Task identifier
              - domain (str): Domain name
              - mode (str): Evaluation mode ("ace")
              - sample_id (str): Example ID from dataset
              - gold (str): Ground truth answer
              - pred (str): Model prediction
              - latency_ms (float): Generation latency in milliseconds
            * Debug columns (always present):
              - question (str): Original question
              - context (str): Context if provided, else empty string
              - correct (int): 1 if exact match, 0 otherwise
            * ACE-specific columns (always present in ACE mode):
              - reflection_latency_ms (float): Time spent on reflection (0.0 if no reflection)
              - reflected (bool): Whether reflection occurred for this example
        - summary: Dict with aggregate statistics:
            * accuracy (float): Overall accuracy (0.0 to 1.0)
            * avg_latency_ms (float): Average latency in milliseconds
            * num_examples (int): Number of examples processed
            * playbook_size (int): Final number of entries in playbook
    """
    results = []
    predictions = []
    labels = []
    latencies = []
    
    for step, example in enumerate(dataset, start=1):
        example_id = example.get("id", "unknown")
        question = example.get("question", "")
        context = example.get("context")
        ground_truth = example.get("answer", "")
        
        # Step 1: Generator - build prompt with playbook context
        generator_prompt = build_generator_prompt(
            domain=domain,
            playbook=playbook,
            question=question,
            context=context,
        )
        
        # Step 2: Generate answer
        start_time = time.time()
        answer = generate(
            model,
            tokenizer,
            generator_prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        latency_ms = (time.time() - start_time) * 1000
        
        # Step 3: Check correctness
        correct = answer.strip().lower() == ground_truth.strip().lower()
        
        # Step 4: Reflector - generate lessons
        should_reflect = not correct or (step % reflect_on_correct_every_n == 0)
        
        if should_reflect:
            reflector_prompt = build_reflector_prompt(
                domain=domain,
                question=question,
                context=context,
                model_answer=answer,
                ground_truth=ground_truth,
                reasoning=None,  # TODO: Extract reasoning if model provides it
            )
            
            # Generate reflection
            reflection_start = time.time()
            reflection_text = generate(
                model,
                tokenizer,
                reflector_prompt,
                max_new_tokens=config.max_new_tokens // 2,  # Shorter for reflections
                temperature=config.temperature,
                top_p=config.top_p,
            )
            reflection_latency_ms = (time.time() - reflection_start) * 1000
            
            # Parse lessons
            raw_lessons = parse_reflector_output_to_lessons(reflection_text)
            filtered_lessons = choose_lessons_for_playbook(
                domain=domain,
                lessons=raw_lessons,
                existing_playbook=playbook,
            )
            
            # Step 5: Curator - add lessons to playbook
            for lesson in filtered_lessons:
                entry = playbook.add_entry(
                    domain=domain,
                    text=lesson,
                    step=step,
                )
                # Record feedback based on correctness
                playbook.record_feedback(entry.id, helpful=correct)
        else:
            reflection_text = ""
            reflection_latency_ms = 0.0
        
        # Step 6: Occasional pruning
        if step % prune_every_n == 0:
            playbook.prune(max_entries_per_domain=32)
        
        # Record result with consistent schema
        result = {
            "model_id": model_id,
            "task_name": task_name,
            "domain": domain,
            "mode": mode,
            "sample_id": example_id,
            "gold": ground_truth,
            "pred": answer,
            "latency_ms": latency_ms,
            # Additional fields for debugging
            "question": question,
            "context": context or "",
            "correct": 1 if correct else 0,
            "reflection_latency_ms": reflection_latency_ms,
            "reflected": should_reflect,
        }
        results.append(result)
        
        predictions.append(answer)
        labels.append(ground_truth)
        latencies.append(latency_ms)
    
    # Save playbook after run
    playbook.save(playbook_path)
    
    # Compute summary
    accuracy = compute_accuracy(predictions, labels)
    avg_latency = compute_average_latency(latencies)
    
    summary = {
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency,
        "num_examples": len(dataset),
        "playbook_size": len(playbook.entries),
    }
    
    return results, summary

