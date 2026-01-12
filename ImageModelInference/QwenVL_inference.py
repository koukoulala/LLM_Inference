# -*- coding: utf-8 -*-
"""
Image and Landing Page Relevance Evaluation Script
This script processes batches of image-landing page pairs and evaluates their relevance.
"""
import argparse
import os
import re
import yaml
import pandas as pd
import time
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


def load_prompts(prompt_file: str) -> Dict[str, Any]:
    """
    Load prompts from YAML file.
    
    Args:
        prompt_file: Path to the prompts YAML file
        
    Returns:
        Dictionary containing prompt templates
    """
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = yaml.safe_load(f)
    return prompts


def prepare_inputs_for_vllm(messages: List[Dict], processor: Any) -> Optional[Dict[str, Any]]:
    """
    Prepare inputs for vLLM inference.
    
    Args:
        messages: List of message dictionaries with role and content
        processor: AutoProcessor for the model
        
    Returns:
        Dictionary with prompt and multi-modal data, or None if failed
    """
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Process vision information (requires qwen_vl_utils 0.0.14+)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True
        )

        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs

        return {
            'prompt': text,
            'multi_modal_data': mm_data,
            'mm_processor_kwargs': video_kwargs
        }
    except Exception as e:
        print(f"Warning: Failed to prepare inputs - {str(e)}")
        return None


def create_messages_from_row(row: pd.Series, prompt_template: str) -> List[Dict]:
    """
    Create message list from a data row using the prompt template.
    
    Args:
        row: DataFrame row containing FinalUrl, ImgUrl, and doc columns
        prompt_template: Prompt template string with placeholders
        
    Returns:
        List of message dictionaries for vLLM
    """
    # Format the prompt with row data
    text_prompt = prompt_template.format(
        FinalUrl=row['FinalUrl'],
        ImgUrl=row['ImgUrl'],
        doc=row['doc']
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": row['ImgUrl'],
                },
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    
    return messages


def parse_result(output_text: str) -> Dict[str, str]:
    """
    Parse the model output to extract Think and Result tags.
    If no tags are found, treat the entire output as the result (for finetuned models).
    
    Args:
        output_text: Raw output from the model
        
    Returns:
        Dictionary with 'think' and 'result' keys
    """
    result = {
        'think': '',
        'result': ''
    }
    
    # Extract content within <Think> tags
    think_match = re.search(r'<Think>(.*?)</Think>', output_text, re.DOTALL | re.IGNORECASE)
    if think_match:
        result['think'] = think_match.group(1).strip()
    
    # Extract content within <Result> tags
    result_match = re.search(r'<Result>(.*?)</Result>', output_text, re.DOTALL | re.IGNORECASE)
    if result_match:
        result['result'] = result_match.group(1).strip()
    else:
        # If no Result tag found, check if there are no tags at all
        # In that case, treat the entire output as the result (for finetuned models)
        if not think_match:
            result['result'] = output_text.strip()
    
    return result


def evaluate_predictions(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate predictions against ground truth labels.
    Calculate confusion matrix, precision, recall, and F1 for each class.
    
    Args:
        df: DataFrame with 'Label' and 'Prediction' columns
        
    Returns:
        Dictionary with evaluation metrics including confusion matrix
    """
    # Normalize labels for comparison (case-insensitive)
    df['Label_normalized'] = df['Label'].str.lower().str.strip()
    df['Prediction_normalized'] = df['Prediction'].str.lower().str.strip()
    
    # Calculate accuracy
    correct = (df['Label_normalized'] == df['Prediction_normalized']).sum()
    total = len(df)
    accuracy = correct / total if total > 0 else 0.0
    
    # Define class order for confusion matrix
    all_labels = ['good', 'fair', 'bad']
    present_labels = [label for label in all_labels if label in df['Label_normalized'].values or label in df['Prediction_normalized'].values]
    
    # Calculate confusion matrix
    confusion_matrix = {}
    for true_label in present_labels:
        confusion_matrix[true_label] = {}
        for pred_label in present_labels:
            count = ((df['Label_normalized'] == true_label) & 
                    (df['Prediction_normalized'] == pred_label)).sum()
            confusion_matrix[true_label][pred_label] = int(count)
    
    # Calculate per-class metrics (precision, recall, F1)
    class_metrics = {}
    
    for label in present_labels:
        # True Positives: predicted as label and actually is label
        tp = ((df['Label_normalized'] == label) & (df['Prediction_normalized'] == label)).sum()
        
        # False Positives: predicted as label but actually is not
        fp = ((df['Label_normalized'] != label) & (df['Prediction_normalized'] == label)).sum()
        
        # False Negatives: predicted as not label but actually is label
        fn = ((df['Label_normalized'] == label) & (df['Prediction_normalized'] != label)).sum()
        
        # True Negatives: predicted as not label and actually is not label
        tn = ((df['Label_normalized'] != label) & (df['Prediction_normalized'] != label)).sum()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Ground truth count
        label_total = (df['Label_normalized'] == label).sum()
        
        class_metrics[label] = {
            'total_in_ground_truth': int(label_total),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
    
    metrics = {
        'overall_accuracy': float(accuracy),
        'total_samples': int(total),
        'correct_predictions': int(correct),
        'confusion_matrix': confusion_matrix,
        'class_metrics': class_metrics
    }
    
    return metrics

# CUDA_VISIBLE_DEVICES=1 nohup python -u QwenVL_inference.py --model_path /data/xiaoyukou/ckpts/Qwen3-VL-2B-Instruct --prompt_name relevance > ../logs/inference_qwen3-vl-2b.out 2>&1 &
def main():
    parser = argparse.ArgumentParser(
        description='Image and Landing Page Relevance Evaluation using Vision-Language Model'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default='../data/Internal100WithSD.tsv',
        help='Path to input TSV file with columns: FinalUrl, ImgUrl, Label, doc'
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        default='../output/test/',
        help='Path to output folder for predictions and evaluation results'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='/data/xiaoyukou/ckpts/Qwen3-VL-2B-Instruct',
        help='Path to the model checkpoint'
    )
    parser.add_argument(
        '--prompt_file',
        type=str,
        default='prompts.yaml',
        help='Path to prompts YAML file'
    )
    parser.add_argument(
        '--prompt_name',
        type=str,
        default='relevance',
        help='Name of the prompt to use from the prompts file'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=40,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=2048,
        help='Maximum number of tokens to generate (out_seq_length)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.8,
        help='Top-p (nucleus) sampling parameter'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=20,
        help='Top-k sampling parameter'
    )
    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=1.0,
        help='Repetition penalty parameter'
    )
    parser.add_argument(
        '--presence_penalty',
        type=float,
        default=1.5,
        help='Presence penalty parameter'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=3407,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--tensor_parallel_size',
        type=int,
        default=None,
        help='Tensor parallel size (default: use all available GPUs)'
    )
    parser.add_argument(
        '--enable_expert_parallel',
        action='store_true',
        help='Enable expert parallelism (only for MoE models like Qwen3-VL-235B)'
    )
    parser.add_argument(
        '--max_model_len',
        type=int,
        default=20000,
        help='Maximum model context length (default: 20000, reduce if OOM)'
    )
    parser.add_argument(
        '--gpu_memory_utilization',
        type=float,
        default=0.95,
        help='GPU memory utilization (0.0-1.0, default: 0.95)'
    )
    parser.add_argument(
        '--output_file_name',
        type=str,
        default='',
        help='Custom output file name (without extension). If empty, will auto-generate based on input file, prompt, and model name'
    )
    
    args = parser.parse_args()
    
    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Generate output file names based on input file name, prompt name, and model name
    if args.output_file_name:
        # Use custom output file name
        base_name = args.output_file_name
    else:
        # Auto-generate based on input file, prompt name, and model name
        input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
        model_name = os.path.basename(args.model_path.rstrip('/'))  # Extract model name from path
        base_name = f"{input_basename}_{args.prompt_name}_{model_name}"
    
    output_file = os.path.join(args.output_folder, f"{base_name}_predictions.tsv")
    eval_file = os.path.join(args.output_folder, f"{base_name}_evaluation.json")
    
    # Load prompts
    print(f"Loading prompts from {args.prompt_file}...")
    prompts = load_prompts(args.prompt_file)
    
    if args.prompt_name not in prompts['prompts']:
        raise ValueError(f"Prompt '{args.prompt_name}' not found in {args.prompt_file}")
    
    prompt_config = prompts['prompts'][args.prompt_name]
    prompt_template = prompt_config['user']
    
    # Load input data
    print(f"Loading input data from {args.input_file}...")
    df = pd.read_csv(args.input_file, sep='\t')
    print(f"Loaded {len(df)} samples")
    
    # Validate required columns
    required_columns = ['FinalUrl', 'ImgUrl', 'Label', 'doc']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Filter out samples with "can't load" label from the beginning
    original_count = len(df)
    df = df[df['Label'].str.lower().str.strip() != "can't load"].reset_index(drop=True)
    filtered_count = original_count - len(df)
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} samples with 'can't load' label")
    
    # Calculate empty doc statistics
    empty_doc_count = df['doc'].isna().sum() + (df['doc'].astype(str).str.strip() == '').sum()
    empty_doc_ratio = empty_doc_count / len(df) if len(df) > 0 else 0.0
    print(f"Empty doc entries: {empty_doc_count}/{len(df)} ({empty_doc_ratio:.2%})")
    print(f"Processing {len(df)} valid samples")
    
    # Initialize model
    print(f"Loading model from {args.model_path}...")
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    tensor_parallel_size = args.tensor_parallel_size or torch.cuda.device_count()
    
    # Build LLM kwargs
    llm_kwargs = {
        'model': args.model_path,
        'mm_encoder_tp_mode': 'data',
        'tensor_parallel_size': tensor_parallel_size,
        'seed': args.seed,
        'max_model_len': args.max_model_len,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'disable_log_stats': True
    }
    
    # Only enable expert parallel for MoE models
    if args.enable_expert_parallel:
        llm_kwargs['enable_expert_parallel'] = True
        print("Expert parallelism enabled (MoE mode)")
    
    print(f"Initializing LLM with max_model_len={args.max_model_len}, gpu_memory_utilization={args.gpu_memory_utilization}")
    llm = LLM(**llm_kwargs)
    
    # Configure sampling parameters with optimal settings from Instruct models
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        seed=args.seed,
        stop_token_ids=[],
    )
    
    print(f"Sampling parameters: temperature={args.temperature}, top_p={args.top_p}, "
          f"top_k={args.top_k}, max_tokens={args.max_tokens}, seed={args.seed}")
    
    # Process data in batches
    print(f"Processing {len(df)} samples with batch size {args.batch_size}...")
    
    # Create lists to store results (will only include successful samples)
    successful_indices = []
    all_predictions = []
    all_thoughts = []
    all_raw_outputs = []
    failed_indices = []  # Track indices of failed samples
    
    total_start_time = time.time()
    total_inference_time = 0.0
    
    for batch_start in tqdm(range(0, len(df), args.batch_size)):
        batch_end = min(batch_start + args.batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        
        # Prepare batch inputs
        batch_messages = []
        batch_original_indices = []
        
        for idx, (df_idx, row) in enumerate(batch_df.iterrows()):
            messages = create_messages_from_row(row, prompt_template)
            batch_messages.append(messages)
            batch_original_indices.append(df_idx)
        
        # Prepare inputs for vLLM (with error handling)
        batch_inputs = []
        valid_indices = []
        for idx, (msg, original_idx) in enumerate(zip(batch_messages, batch_original_indices)):
            prepared_input = prepare_inputs_for_vllm(msg, processor)
            if prepared_input is not None:
                batch_inputs.append(prepared_input)
                valid_indices.append(original_idx)
            else:
                # Track failed sample but don't include in results
                failed_indices.append(original_idx)
        
        # Run inference only on valid inputs
        if batch_inputs:
            batch_start_time = time.time()
            outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
            batch_end_time = time.time()
            batch_inference_time = batch_end_time - batch_start_time
            total_inference_time += batch_inference_time
            
            # Parse results and store with original indices
            for output, original_idx in zip(outputs, valid_indices):
                raw_text = output.outputs[0].text
                parsed = parse_result(raw_text)
                
                successful_indices.append(original_idx)
                all_raw_outputs.append(raw_text)
                all_thoughts.append(parsed['think'])
                all_predictions.append(parsed['result'])
    
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    
    # Create results dataframe with only successful samples
    result_df = df.loc[successful_indices].copy()
    result_df['Prediction'] = all_predictions
    result_df['Think'] = all_thoughts
    result_df['RawOutput'] = all_raw_outputs
    
    # Select only required columns for output: FinalUrl, ImgUrl, Label, and prediction results
    output_columns = ['FinalUrl', 'ImgUrl', 'Label', 'Prediction', 'RawOutput']
    result_df_output = result_df[output_columns]
    
    # Save predictions
    print(f"Saving predictions to {output_file}...")
    result_df_output.to_csv(output_file, sep='\t', index=False)
    
    # Evaluate predictions
    print("Evaluating predictions...")
    successful_samples = len(result_df)
    if successful_samples > 0:
        metrics = evaluate_predictions(result_df)
    else:
        metrics = {
            'overall_accuracy': 0.0,
            'total_samples': 0,
            'correct_predictions': 0,
            'confusion_matrix': {},
            'class_metrics': {}
        }
    
    # Add latency and failure statistics to metrics
    avg_inference_time = total_inference_time / successful_samples if successful_samples > 0 else 0.0
    
    metrics['latency_stats'] = {
        'total_elapsed_time_seconds': round(total_elapsed_time, 2),
        'total_inference_time_seconds': round(total_inference_time, 2),
        'average_inference_time_per_sample_seconds': round(avg_inference_time, 4),
        'samples_per_second': round(successful_samples / total_inference_time, 2) if total_inference_time > 0 else 0.0
    }
    
    # Note: sample_stats.total_samples refers to valid samples (after filtering "can't load")
    # Accuracy is calculated only on successfully processed valid samples
    metrics['sample_stats'] = {
        'original_total_samples': original_count,  # Total from input file
        'cant_load_samples': filtered_count,  # Filtered out "can't load" labels
        'valid_samples': len(df),  # After filtering "can't load"
        'successful_samples': successful_samples,  # Successfully processed
        'failed_during_processing': len(failed_indices),  # Failed during image loading/processing
        'success_rate': round(successful_samples / len(df), 4) if len(df) > 0 else 0.0,
        'empty_doc_count': int(empty_doc_count),  # Number of samples with empty doc
        'empty_doc_ratio': round(empty_doc_ratio, 4)  # Ratio of empty doc entries
    }
    
    if failed_indices:
        metrics['failed_indices'] = failed_indices
    
    # Save evaluation results
    print(f"Saving evaluation results to {eval_file}...")
    import json
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Samples in Input: {len(df)}")
    print(f"Successfully Processed: {successful_samples} | Failed to Load: {len(failed_indices)}")
    if len(failed_indices) > 0:
        print(f"Failed Indices: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}")
    print(f"Success Rate: {metrics['sample_stats']['success_rate']:.2%}")
    print(f"Empty Doc Entries: {metrics['sample_stats']['empty_doc_count']} ({metrics['sample_stats']['empty_doc_ratio']:.2%})")
    
    print(f"\nLatency Statistics:")
    print(f"  Total Elapsed Time: {metrics['latency_stats']['total_elapsed_time_seconds']:.2f}s")
    print(f"  Total Inference Time: {metrics['latency_stats']['total_inference_time_seconds']:.2f}s")
    print(f"  Average Time per Sample: {metrics['latency_stats']['average_inference_time_per_sample_seconds']:.4f}s")
    print(f"  Throughput: {metrics['latency_stats']['samples_per_second']:.2f} samples/s")
    
    print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['correct_predictions']}/{metrics['total_samples']})")
    
    # Print confusion matrix
    if 'confusion_matrix' in metrics and metrics['confusion_matrix']:
        print(f"\nConfusion Matrix:")
        print(f"{'':>10} ", end='')
        labels = sorted(metrics['confusion_matrix'].keys())
        for label in labels:
            print(f"{label:>8}", end='')
        print()
        for true_label in labels:
            print(f"{true_label:>10} ", end='')
            for pred_label in labels:
                count = metrics['confusion_matrix'][true_label].get(pred_label, 0)
                print(f"{count:>8}", end='')
            print()
    
    # Print per-class metrics
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':>8} {'Total':>8} {'TP':>8} {'FP':>8} {'FN':>8} {'Precision':>12} {'Recall':>12} {'F1':>12}")
    print("-" * 80)
    for label in sorted(metrics['class_metrics'].keys()):
        cm = metrics['class_metrics'][label]
        print(f"{label.upper():>8} {cm['total_in_ground_truth']:>8} {cm['true_positives']:>8} "
              f"{cm['false_positives']:>8} {cm['false_negatives']:>8} "
              f"{cm['precision']:>12.4f} {cm['recall']:>12.4f} {cm['f1_score']:>12.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()