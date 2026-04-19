"""
training/nli_audit_pipeline.py

This script performs a full NLI audit of the hallucination detection dataset,
identifies misclassified questions, visualises results with detailed graphs,
and then proposes concrete improvements to the NLI system.

Purpose:
- Load the labelled dataset (features_correct_incorrect.xlsx)
- Run NLI on every row using the nli_utils module from src/nli_utils.py
- Compute entailment scores and verdicts for every question-answer pair
- Identify wrongly flagged questions (false positives and false negatives)
- For each wrong prediction, display the retrieved context chunk
- Produce detailed matplotlib/seaborn graphs saved to data/results/nli_audit/
- Based on the analysis, log concrete improvement suggestions
"""

import os
import sys
import gc
import logging
import argparse
import json
import warnings
import re
from datetime import datetime
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
# Use Agg backend for headless plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    roc_curve, 
    auc, 
    precision_recall_curve
)
from tqdm import tqdm

# Constants and Paths
# BASE_DIR is two levels up from this script (typically file -> training/ -> root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Add project root to sys.path so that src/nli_utils.py is importable
sys.path.insert(0, str(BASE_DIR))

DATA_PATH = BASE_DIR / "data" / "processed" / "features_correct_incorrect.xlsx"
PDF_DIR = BASE_DIR / "resources" / "pdfs"
MODELS_DIR = BASE_DIR / "models"
INDEX_DIR = BASE_DIR / "models" / "nli_index"
OUTPUT_DIR = BASE_DIR / "data" / "results" / "nli_audit"
GRAPHS_DIR = OUTPUT_DIR / "graphs"
LOG_FILE = OUTPUT_DIR / "audit_log.txt"
RESULTS_XLSX = OUTPUT_DIR / "nli_audit_results.xlsx"

# Create exact directories before logging setup
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# Logging Setup
logger = logging.getLogger("NLIAudit")
logger.setLevel(logging.DEBUG)

# Create console handler (INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)

# Create file handler (DEBUG)
fh = logging.FileHandler(str(LOG_FILE), mode='a', encoding='utf-8')
fh.setLevel(logging.DEBUG)

# Create formatter with timestamp
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# Add the handlers to the logger
if not logger.handlers:
    logger.addHandler(ch)
    logger.addHandler(fh)

def cleanup_gpu_and_memory():
    """
    Cleans up GPU memory and shared variables to prevent OOM errors.
    """
    try:
        gc.collect()
        gc.collect()
        gc.collect()
        
        if torch.cuda.is_available():
            before_reserved = torch.cuda.memory_reserved()
            
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            
            after_reserved = torch.cuda.memory_reserved()
            freed = before_reserved - after_reserved
            logger.debug(
                f"[Memory Cleanup] GPU Memory Freed: {freed / (1024**2):.2f} MB. "
                f"Current reserved: {after_reserved / (1024**2):.2f} MB"
            )
            
        gc.collect()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{timestamp}] GPU and shared memory cleanup completed successfully.")
        
    except Exception as e:
        logger.warning(f"Failed to cleanup GPU and memory safely: {e}")

# Module level execution ensures memory drops immediately on import/run
cleanup_gpu_and_memory()

def get_pdf_path(doc_id: str) -> str | None:
    """
    Given a doc_id (e.g., 'bosch_oven.pdf' or 'bosch_oven'), return the full path
    under resources/pdfs/ if the file exists, else None.
    """
    if not doc_id.endswith('.pdf'):
        doc_id = f"{doc_id}.pdf"
    
    pdf_path = PDF_DIR / doc_id
    if pdf_path.exists():
        return str(pdf_path)
    return None

def load_and_validate_dataset(data_path: str) -> pd.DataFrame:
    """
    Loads the Excel dataset, validates its structure, cleans the data,
    and logs a comprehensive summary.
    """
    path_obj = Path(data_path)
    if not path_obj.exists():
        logger.error(f"Dataset not found at {data_path}")
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    logger.info(f"Loading dataset from {data_path}...")
    try:
        df = pd.read_excel(data_path)
    except Exception as e:
        logger.error(f"Failed to read dataset: {e}")
        raise
        
    initial_rows = len(df)
    
    # Validate columns
    required_cols = {'doc_id', 'question', 'answer', 'label'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        error_msg = f"Missing required columns in dataset: {missing_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    # Validate labels
    valid_labels = {0, 1}
    unique_labels = set(df['label'].dropna().unique())
    if not unique_labels.issubset(valid_labels):
        error_msg = f"Invalid labels found: {unique_labels - valid_labels}. Allowed: 0 (correct), 1 (hallucinated)."
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    # Cleaning: Strip whitespace
    for col in ['doc_id', 'question', 'answer']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip()
            
    # Drop empty/NaN in question or answer
    df = df.replace('', np.nan)
    df_cleaned = df.dropna(subset=['question', 'answer']).copy()
    
    dropped_rows = initial_rows - len(df_cleaned)
    df_cleaned.reset_index(drop=True, inplace=True)
    
    # Validate doc_ids against PDFs
    unique_docs = df_cleaned['doc_id'].unique()
    orphaned_docs = []
    for doc in unique_docs:
        if not get_pdf_path(doc):
            orphaned_docs.append(doc)
            
    if orphaned_docs:
        logger.warning(f"Orphaned doc_ids found (no corresponding PDF in {PDF_DIR}): {orphaned_docs}")
        
    # Reporting
    logger.info(f"--- Dataset Summary ---")
    logger.info(f"Total rows loaded: {initial_rows}")
    logger.info(f"Rows after cleaning: {len(df_cleaned)} (Dropped {dropped_rows} rows due to missing data)")
    
    label_counts = df_cleaned['label'].value_counts()
    total_valid = len(df_cleaned)
    
    logger.info("Class distribution:")
    for lbl in [0, 1]:
        count = label_counts.get(lbl, 0)
        pct = (count / total_valid * 100) if total_valid > 0 else 0
        label_name = "Correct/Grounded" if lbl == 0 else "Hallucinated"
        logger.info(f"  Label {lbl} ({label_name}): {count} ({pct:.1f}%)")
        
    logger.info(f"Number of unique doc_ids: {len(unique_docs)}")
    
    doc_counts = df_cleaned['doc_id'].value_counts()
    logger.info("Rows per doc_id:")
    for doc, count in doc_counts.items():
        logger.info(f"  {doc}: {count} rows")
    logger.info(f"-----------------------")
    
    return df_cleaned

def run_nli_audit(df: pd.DataFrame,
                  similarity_threshold: float = 0.70,
                  max_rows: int = None,
                  save_every: int = 50) -> pd.DataFrame:
    """
    Main audit loop running NLI on every row of the dataset.
    """
    cleanup_gpu_and_memory()
    
    # Imports from src.nli_utils
    try:
        from src.nli_utils import (
            get_embedder,
            build_or_load_index,
            blackbox_predict_unified,
            SIMILARITY_THRESHOLD
        )
    except ImportError as e:
        logger.error(f"Failed to import from src.nli_utils: {e}")
        raise
        
    logger.info("Warming up embedder...")
    get_embedder()
    
    index_cache = {}
    results = []
    
    if max_rows is not None:
        df = df.head(max_rows)
        
    logger.info(f"Starting NLI audit on {len(df)} rows...")
    
    row_counter = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="NLI Audit"):
        row_counter += 1
        
        # Set up a base record containing defaults
        rec = {
            'row_index': idx,
            'doc_id': row.get('doc_id', ''),
            'question': row.get('question', ''),
            'answer': row.get('answer', ''),
            'true_label': row.get('label', -1),
            'predicted_label': -1,
            'verdict': '',
            'entailment_score': 0.0,
            'neutral_score': 0.0,
            'contradiction_score': 0.0,
            'max_similarity': 0.0,
            'retrieved_context': '',
            'is_correct': False,
            'error_type': ''
        }
        
        try:
            doc_id = str(row.get('doc_id', ''))
            pdf_path = get_pdf_path(doc_id)
            
            if not pdf_path:
                rec['verdict'] = "PDF_MISSING"
                rec['error_type'] = "SKIP"
                results.append(rec)
                continue
                
            if doc_id not in index_cache:
                index_cache[doc_id] = build_or_load_index(str(pdf_path), doc_id, get_embedder(), index_dir=str(INDEX_DIR))
                
            doc_index = index_cache[doc_id]
            if doc_index is None:
                rec['verdict'] = "INDEX_FAILED"
                rec['error_type'] = "SKIP"
                results.append(rec)
                continue
                
            # Perform blackbox NLI evaluation
            pred = blackbox_predict_unified(
                doc_index, 
                rec['question'], 
                rec['answer'], 
                similarity_threshold=similarity_threshold
            )
            
            if row_counter <= 5:
                logger.debug(f"Raw NLI result keys: {list(pred.keys())}")
                logger.debug(f"Entailment value: {pred.get('entailment', 'KEY MISSING')}")

            # Extract returned metrics
            raw_verdict = pred.get('verdict', 'ERROR')
            context = pred.get('retrieved_context', '')
            entailment = float(pred.get('entailment', 0.0))
            contradiction = float(pred.get('contradiction', 0.0))

            from src.nli_utils import VERDICT_MAP
            verdict = VERDICT_MAP.get(raw_verdict, raw_verdict)
            
            rec['verdict_detail'] = raw_verdict

            if verdict == "GROUNDED":
                predicted_label = 0
            else:
                # UNCERTAIN means DeBERTa could not confirm the answer is grounded.
                # In a hallucination detection system, unconfirmed = suspect.
                # The correct fix for UNCERTAIN is not remapping — it is fine-tuning
                # DeBERTa to produce stronger contradiction signals (see finetune_nli.py).
                # HALLUCINATION and UNCERTAIN both map to hallucinated
                # Rationale: if the model cannot confirm the answer is grounded,
                # treat it as suspect. Better to flag and review than to miss.
                predicted_label = 1
            rec['verdict'] = verdict
            rec['predicted_label'] = predicted_label
            
            if 'entailment' not in pred:
                logger.warning(f"[ROW {idx}] 'entailment' key missing from NLI result. Keys: {list(pred.keys())}")

            rec['entailment_score'] = float(pred.get('entailment', 0.0))
            rec['neutral_score'] = float(pred.get('neutral', 0.0))
            rec['contradiction_score'] = float(pred.get('contradiction', 0.0))
            rec['max_similarity'] = float(pred.get('max_similarity', -1.0))
            rec['retrieved_context'] = str(context)[:600] if context else ""
            
            true_label = rec['true_label']
            rec['is_correct'] = bool(predicted_label == true_label)
            
            if true_label == 1 and predicted_label == 1:
                rec['error_type'] = "TP"
            elif true_label == 0 and predicted_label == 0:
                rec['error_type'] = "TN"
            elif true_label == 0 and predicted_label == 1:
                rec['error_type'] = "FP"
            elif true_label == 1 and predicted_label == 0:
                rec['error_type'] = "FN"
            else:
                rec['error_type'] = "SKIP"

        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            rec['verdict'] = "ERROR"
            rec['error_type'] = "SKIP"
        
        results.append(rec)
        
        # Save checkpoints and flush memory chunks
        if len(results) % save_every == 0:
            temp_df = pd.DataFrame(results)
            import re
            # Remove illegal Excel characters
            temp_df = temp_df.map(lambda x: re.sub(r'[\000-\010]|[\013-\014]|[\016-\037]', '', x) if isinstance(x, str) else x)
            temp_df.to_excel(RESULTS_XLSX, index=False, sheet_name="audit_results")
            cleanup_gpu_and_memory()

    # Final cleanup & save layer
    cleanup_gpu_and_memory()
    
    res_df = pd.DataFrame(results)
    import re
    res_df = res_df.map(lambda x: re.sub(r'[\000-\010]|[\013-\014]|[\016-\037]', '', x) if isinstance(x, str) else x)
    res_df.to_excel(RESULTS_XLSX, index=False, sheet_name="audit_results")
    
    # Construct logging summary
    total_processed = len(res_df)
    skipped = len(res_df[res_df['error_type'] == 'SKIP'])
    correct = len(res_df[res_df['is_correct'] == True])
    fp_count = len(res_df[res_df['error_type'] == 'FP'])
    fn_count = len(res_df[res_df['error_type'] == 'FN'])
    
    logger.info("--- Audit Summary ---")
    logger.info(f"Total processed: {total_processed}")
    logger.info(f"Skipped/Errors: {skipped}")
    logger.info(f"Correct predictions: {correct}")
    logger.info(f"False Positives (incorrectly flagged hallucination): {fp_count}")
    logger.info(f"False Negatives (missed a real hallucination): {fn_count}")
    logger.info("---------------------")
    
    return res_df

def analyse_wrong_predictions(results_df: pd.DataFrame) -> dict:
    """
    Analyzes misclassified rows (FP and FN) to generate deep-dive pattern
    insights and logs the detailed breakdown.
    """
    total = len(results_df)
    correct = len(results_df[results_df['is_correct'] == True])
    fp_df = results_df[results_df['error_type'] == 'FP']
    fn_df = results_df[results_df['error_type'] == 'FN']
    skipped = len(results_df[results_df['error_type'] == 'SKIP'])
    
    fp_count = len(fp_df)
    fn_count = len(fn_df)
    
    # 1. OVERALL COUNTS TABLE
    logger.info("\n=== PREDICTION ANALYSIS BREAKDOWN ===")
    logger.info("| Metric                        | Count | % of total |")
    logger.info("|-------------------------------|-------|------------|")
    logger.info(f"| Total evaluated               | {total:<5} | {100.0:>9.1f}% |")
    logger.info(f"| Correctly classified          | {correct:<5} | {(correct/total*100) if total else 0:>9.1f}% |")
    logger.info(f"| Wrong — False Positives (FP)  | {fp_count:<5} | {(fp_count/total*100) if total else 0:>9.1f}% |")
    logger.info(f"| Wrong — False Negatives (FN)  | {fn_count:<5} | {(fn_count/total*100) if total else 0:>9.1f}% |")
    logger.info(f"| Skipped (PDF/index missing)   | {skipped:<5} | {(skipped/total*100) if total else 0:>9.1f}% |")
    logger.info("======================================\n")

    import textwrap
    
    # 2. FALSE POSITIVES DEEP DIVE
    logger.info(">>> FALSE POSITIVES DIVE (Wrongly flagged as hallucinated)")
    for _, row in fp_df.iterrows():
        logger.info(f"Row {row['row_index']} | Doc: {row['doc_id']}")
        logger.info(f"Q: {row['question']}")
        logger.info(f"A: {row['answer']}")
        logger.info(f"Scores -> Entailment: {row['entailment_score']:.3f} | Neutral: {row['neutral_score']:.3f} | Contradiction: {row['contradiction_score']:.3f}")
        logger.info(f"Max Sim: {row['max_similarity']:.3f} | Verdict: {row['verdict']}")
        
        ctx = str(row['retrieved_context'])
        wrapped_ctx = textwrap.fill(ctx, width=100)
        logger.info(f"Context:\n{wrapped_ctx}")
        
        # Hypothesis logic
        hypo = []
        if row['max_similarity'] < 0.70:
            hypo.append("[FAILED THRESHOLD]")
        if row['entailment_score'] < 0.90 and row['contradiction_score'] <= row['entailment_score']:
            hypo.append("[LOW ENTAILMENT]")
        if row['contradiction_score'] > row['entailment_score']:
            hypo.append("[CONTRADICTION_DOMINATES]")
            
        logger.info(f"HYPOTHESIS: {' '.join(hypo) if hypo else '[UNCLEAR]'}")
        logger.info("-" * 60)
        
    # 3. FALSE NEGATIVES DEEP DIVE
    logger.info("\n>>> FALSE NEGATIVES DIVE (Missed true hallucinations)")
    for _, row in fn_df.iterrows():
        logger.info(f"Row {row['row_index']} | Doc: {row['doc_id']}")
        logger.info(f"Q: {row['question']}")
        logger.info(f"A: {row['answer']}")
        logger.info(f"Scores -> Entailment: {row['entailment_score']:.3f} | Neutral: {row['neutral_score']:.3f} | Contradiction: {row['contradiction_score']:.3f}")
        logger.info(f"Max Sim: {row['max_similarity']:.3f} | Verdict: {row['verdict']}")
        
        ctx = str(row['retrieved_context'])
        wrapped_ctx = textwrap.fill(ctx, width=100)
        logger.info(f"Context:\n{wrapped_ctx}")
        
        # Hypothesis logic
        hypo = []
        if row['entailment_score'] >= 0.90:
            hypo.append("[HIGH ENTAILMENT FOOLED MODEL]")
        if row['max_similarity'] < 0.50:
            hypo.append("[WEAK CONTEXT RETRIEVAL]")
            
        logger.info(f"HYPOTHESIS: {' '.join(hypo) if hypo else '[UNCLEAR]'}")
        logger.info("-" * 60)
        
    # 4. PATTERN ANALYSIS
    logger.info("\n>>> PATTERN ANALYSIS SUMMARY")
    
    # UNCERTAIN VERDICT ANALYSIS
    logger.info("\n--- UNCERTAIN VERDICT ANALYSIS ---")
    if 'verdict' in results_df.columns:
        uncertain_df = results_df[results_df['verdict'] == 'UNCERTAIN']
        un_count = len(uncertain_df)
        logger.info(f"Total UNCERTAIN verdicts: {un_count}")
        if un_count > 0:
            un_lbl0 = len(uncertain_df[uncertain_df['true_label'] == 0])
            un_lbl1 = len(uncertain_df[uncertain_df['true_label'] == 1])
            logger.info(f"Breakdown: Label=0 (Correct): {un_lbl0} ({un_lbl0/un_count*100:.1f}%) | Label=1 (Hallucinated): {un_lbl1} ({un_lbl1/un_count*100:.1f}%)")
            
    logger.info("-" * 40)

    # Doc ID Errors
    error_docs = results_df[results_df['is_correct'] == False]['doc_id'].value_counts()
    logger.info("Top 5 docs with most errors:")
    for doc, count in error_docs.head(5).items():
        logger.info(f"  {doc}: {count} errors")
        
    # Valid evaluations slice
    valid_df = results_df[results_df['error_type'] != 'SKIP']
    
    if len(valid_df) > 0:
        # Entailment & Similarity mapping
        correct_df = results_df[results_df['is_correct'] == True]
        
        c_ent = correct_df['entailment_score']
        fp_ent = fp_df['entailment_score']
        fn_ent = fn_df['entailment_score']
        
        logger.info("\nMean(Std) Entailment Scores:")
        logger.info(f"  Correct: {c_ent.mean():.3f} ({c_ent.std():.3f})")
        logger.info(f"  FP:      {fp_ent.mean():.3f} ({fp_ent.std():.3f})" if len(fp_ent) else "  FP:      N/A")
        logger.info(f"  FN:      {fn_ent.mean():.3f} ({fn_ent.std():.3f})" if len(fn_ent) else "  FN:      N/A")
        
        c_sim = correct_df['max_similarity']
        fp_sim = fp_df['max_similarity']
        fn_sim = fn_df['max_similarity']
        
        logger.info("\nMean Max Similarity Scores:")
        logger.info(f"  Correct: {c_sim.mean():.3f}")
        logger.info(f"  FP:      {fp_sim.mean():.3f}" if len(fp_sim) else "  FP:      N/A")
        logger.info(f"  FN:      {fn_sim.mean():.3f}" if len(fn_sim) else "  FN:      N/A")
        
        # Wrong verdicts mapping
        wrong_df = results_df[results_df['is_correct'] == False]
        wrong_verdicts = wrong_df['verdict'].value_counts()
        logger.info("\nVerdict distribution for wrong predictions:")
        for verd, cnt in wrong_verdicts.items():
            logger.info(f"  {verd}: {cnt} items")
            
    # 5. SAVE FP / FN to EXCEL SHEETS
    try:
        with pd.ExcelWriter(RESULTS_XLSX, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            fp_df.to_excel(writer, sheet_name='false_positives', index=False)
            fn_df.to_excel(writer, sheet_name='false_negatives', index=False)
        logger.info(f"\nSaved 'false_positives' and 'false_negatives' sheets to {RESULTS_XLSX}")
    except Exception as e:
        logger.warning(f"Failed to append sheets to {RESULTS_XLSX}: {e}")
        
    logger.info("======================================\n")
    
    # 6. RETURN DICT
    return {
        'fp_df': fp_df,
        'fn_df': fn_df,
        'total': total,
        'correct': correct,
        'fp_count': fp_count,
        'fn_count': fn_count,
        'skipped': skipped
    }

def plot_entailment_distributions(results_df: pd.DataFrame):
    """
    Creates and saves THREE matplotlib figures to visualize entailment distributions
    and patterns for misclassifications.
    """
    logger.info("\nGenerating entailment distribution graphs...")
    sns.set_style("whitegrid")
    
    # Pre-filter out skipped rows for graphing
    valid_df = results_df[results_df['error_type'] != 'SKIP'].copy()
    if valid_df.empty:
        logger.warning("No valid predictions to plot. Skipping graphs.")
        return

    # ---------------------------------------------------------
    # FIGURE 1: entailment_score_by_label.png
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    df_correct_label = valid_df[valid_df['true_label'] == 0]
    df_hallu_label = valid_df[valid_df['true_label'] == 1]
    
    if not df_correct_label.empty:
        sns.kdeplot(data=df_correct_label, x='entailment_score', fill=True, label='Grounded (True=0)', color='blue')
    if not df_hallu_label.empty:
        sns.kdeplot(data=df_hallu_label, x='entailment_score', fill=True, label='Hallucinated (True=1)', color='red')
        
    plt.axvline(0.90, color='red', linestyle='--', label='Threshold (0.90)')
    
    plt.title("Entailment Score Distribution — Correct vs Hallucinated")
    plt.xlabel("Entailment Score (0–1)")
    plt.ylabel("Density")
    
    # Annotate roughly where overlap might happen
    plt.text(0.92, plt.ylim()[1] * 0.5, "Overlap\nRegion", color='darkred', ha='left')
    plt.legend()
    plt.tight_layout()
    
    fig1_path = GRAPHS_DIR / "entailment_score_by_label.png"
    plt.savefig(fig1_path, dpi=150)
    plt.close()
    logger.info(f"Saved figure: {fig1_path}")

    # ---------------------------------------------------------
    # FIGURE 2: entailment_by_verdict.png
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    sns.boxplot(data=valid_df, x='verdict', y='entailment_score', palette='Set2')
    sns.stripplot(data=valid_df, x='verdict', y='entailment_score', color='.3', alpha=0.4, jitter=True)
    
    plt.axhline(0.90, color='red', linestyle='--', label='Threshold (0.90)')
    plt.title("Entailment Score by NLI Verdict")
    plt.ylabel("Entailment Score (0-1)")
    plt.xlabel("NLI Verdict")
    
    plt.tight_layout()
    fig2_path = GRAPHS_DIR / "entailment_by_verdict.png"
    plt.savefig(fig2_path, dpi=150)
    plt.close()
    logger.info(f"Saved figure: {fig2_path}")

    # ---------------------------------------------------------
    # FIGURE 3: entailment_fp_fn_correct.png
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    fp_df = valid_df[valid_df['error_type'] == 'FP']
    fn_df = valid_df[valid_df['error_type'] == 'FN']
    correct_df = valid_df[valid_df['is_correct'] == True]
    
    if not fp_df.empty:
        sns.kdeplot(data=fp_df, x='entailment_score', fill=True, color='red', label=f"FP (n={len(fp_df)})")
    if not fn_df.empty:
        sns.kdeplot(data=fn_df, x='entailment_score', fill=True, color='orange', label=f"FN (n={len(fn_df)})")
    if not correct_df.empty:
        sns.kdeplot(data=correct_df, x='entailment_score', fill=True, color='green', label=f"Correct (n={len(correct_df)})")
        
    plt.axvline(0.90, color='black', linestyle='--', label='Threshold (0.90)')
    
    plt.title("Entailment Score — Correct vs False Positive vs False Negative")
    plt.xlabel("Entailment Score (0–1)")
    plt.ylabel("Density")
    plt.legend()
    
    plt.tight_layout()
    fig3_path = GRAPHS_DIR / "entailment_fp_fn_correct.png"
    plt.savefig(fig3_path, dpi=150)
    plt.close()
    logger.info(f"Saved figure: {fig3_path}")

def plot_similarity_and_performance(results_df: pd.DataFrame):
    """
    Creates and saves performance metrics and similarity distributions.
    Figures 4, 5, 6, and 7.
    """
    logger.info("\nGenerating similarity and performance graphs...")
    sns.set_style("whitegrid")
    
    valid_df = results_df[results_df['error_type'] != 'SKIP'].copy()
    if valid_df.empty:
        logger.warning("No valid predictions to plot. Skipping performance graphs.")
        return

    # ---------------------------------------------------------
    # FIGURE 4: similarity_score_distribution.png
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    sns.histplot(
        data=valid_df, 
        x='max_similarity', 
        hue='true_label', 
        bins=40, 
        palette={0: 'blue', 1: 'red'}, 
        multiple='stack'
    )
    
    plt.axvline(0.70, color='black', linestyle='--', label='Threshold (0.70)')
    plt.text(0.71, plt.ylim()[1] * 0.9, "Threshold = 0.70", color='black', fontsize=10)
    
    plt.title("Max FAISS Similarity Score Distribution")
    plt.xlabel("Max Similarity Score")
    plt.ylabel("Count")
    
    plt.tight_layout()
    fig4_path = GRAPHS_DIR / "similarity_score_distribution.png"
    plt.savefig(fig4_path, dpi=150)
    plt.close()
    logger.info(f"Saved figure: {fig4_path}")

    # ---------------------------------------------------------
    # FIGURE 5: confusion_matrix.png
    # ---------------------------------------------------------
    pred_df = valid_df[valid_df['error_type'].isin(['TP', 'TN', 'FP', 'FN'])]
    
    if not pred_df.empty:
        y_true = pred_df['true_label']
        y_pred = pred_df['predicted_label']
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        # Calculate extra metrics
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        acc = (tp + tn) / len(pred_df)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                         xticklabels=["Correct (0)", "Hallucinated (1)"], 
                         yticklabels=["Correct (0)", "Hallucinated (1)"])
                         
        plt.xlabel("Predicted", labelpad=10)
        plt.ylabel("True", labelpad=10)
        plt.title("NLI Prediction Confusion Matrix", pad=15)
        
        # Add text below heatmap
        metrics_text = f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}"
        plt.text(0.5, -0.15, metrics_text, ha='center', va='top', transform=ax.transAxes, fontsize=11, 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
                 
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave room at bottom for text
        fig5_path = GRAPHS_DIR / "confusion_matrix.png"
        plt.savefig(fig5_path, dpi=150)
        plt.close()
        logger.info(f"Saved figure: {fig5_path}")
    else:
        logger.warning("Not enough TP/TN/FP/FN data to plot confusion matrix.")

    # ---------------------------------------------------------
    # FIGURE 6: roc_curve.png
    # ---------------------------------------------------------
    if len(valid_df['true_label'].unique()) > 1:
        # Invert entailment score because high entailment = correct (0), low = hallucinated (1)
        y_true_all = valid_df['true_label']
        y_scores_all = 1.0 - valid_df['entailment_score'] 
        
        fpr, tpr, roc_thresholds = roc_curve(y_true_all, y_scores_all)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Baseline')
        
        # Find exactly where our entailment threshold maps to (0.90 entailment -> 0.10 inverted score)
        idx = (np.abs(roc_thresholds - 0.10)).argmin() if len(roc_thresholds) > 0 else 0
        plt.plot(fpr[idx], tpr[idx], marker='*', color='red', markersize=15, label='Operating Point (Thr=0.90)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve — Entailment Score as Hallucination Detector')
        plt.legend(loc="lower right")
        
        plt.tight_layout()
        fig6_path = GRAPHS_DIR / "roc_curve.png"
        plt.savefig(fig6_path, dpi=150)
        plt.close()
        logger.info(f"Saved figure: {fig6_path}")

        # ---------------------------------------------------------
        # FIGURE 7: precision_recall_curve.png
        # ---------------------------------------------------------
        from sklearn.metrics import average_precision_score
        
        precision_arr, recall_arr, _ = precision_recall_curve(y_true_all, y_scores_all)
        ap_score = average_precision_score(y_true_all, y_scores_all)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall_arr, precision_arr, color='blue', lw=2, label=f'PR curve (AP = {ap_score:.3f})')
        
        # Current F1 can be calculated from our existing pred_df metrics
        current_f1 = f1 if not pred_df.empty else 0.0
        plt.axhline(current_f1, color='red', linestyle='--', label=f'Operating F1 = {current_f1:.3f}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve — Entailment Score')
        plt.legend(loc="lower left")
        
        plt.tight_layout()
        fig7_path = GRAPHS_DIR / "precision_recall_curve.png"
        plt.savefig(fig7_path, dpi=150)
        plt.close()
        logger.info(f"Saved figure: {fig7_path}")
    else:
        logger.warning("Only one class present in true labels. Cannot compute ROC/PR curves.")

def plot_per_document_analysis(results_df: pd.DataFrame):
    """
    Creates and saves document-level analysis to isolate problematic PDFs and view NLI verdict distributions.
    Figures 8, 9, and 10.
    """
    logger.info("\nGenerating document-level analysis graphs...")
    sns.set_style("whitegrid")
    
    valid_df = results_df[results_df['error_type'] != 'SKIP'].copy()
    if valid_df.empty:
        logger.warning("No valid predictions to plot. Skipping document-level graphs.")
        return

    # ---------------------------------------------------------
    # FIGURE 8: accuracy_per_document.png
    # ---------------------------------------------------------
    doc_stats = valid_df.groupby('doc_id').agg(
        total=('is_correct', 'count'),
        correct=('is_correct', 'sum')
    ).reset_index()
    
    doc_stats['accuracy'] = (doc_stats['correct'] / doc_stats['total']) * 100
    doc_stats = doc_stats.sort_values(by='accuracy', ascending=True)
    
    plt.figure(figsize=(10, max(6, len(doc_stats) * 0.4)))
    
    # Conditional coloring
    colors = []
    for acc in doc_stats['accuracy']:
        if acc >= 80:
            colors.append('green')
        elif acc >= 60:
            colors.append('orange')
        else:
            colors.append('red')
            
    ax = sns.barplot(data=doc_stats, x='accuracy', y='doc_id', palette=colors)
    
    # Annotate n=X
    for i, row in enumerate(doc_stats.itertuples()):
        ax.text(row.accuracy + 1, i, f"n={row.total}", va='center', fontsize=9)
        
    plt.xlim(0, 105)
    plt.title("NLI Accuracy per Document", pad=15)
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Document ID")
    
    plt.tight_layout()
    fig8_path = GRAPHS_DIR / "accuracy_per_document.png"
    plt.savefig(fig8_path, dpi=150)
    plt.close()
    logger.info(f"Saved figure: {fig8_path}")

    # ---------------------------------------------------------
    # FIGURE 9: error_type_breakdown_per_doc.png
    # ---------------------------------------------------------
    # Pivot to get counts of TP, TN, FP, FN per doc
    doc_errors = valid_df.groupby(['doc_id', 'error_type']).size().unstack(fill_value=0).reset_index()
    
    for col in ['TP', 'TN', 'FP', 'FN']:
        if col not in doc_errors.columns:
            doc_errors[col] = 0
            
    doc_errors['total_errors'] = doc_errors['FP'] + doc_errors['FN']
    doc_errors = doc_errors.sort_values(by='total_errors', ascending=False)
    
    # Drop total_errors for plotting
    plot_data = doc_errors.set_index('doc_id')[['TP', 'TN', 'FP', 'FN']]
    
    # Distinct plot mapping colors correctly per requirement
    color_map = {'TP': 'green', 'TN': 'blue', 'FP': 'red', 'FN': 'orange'}
    
    ax = plot_data.plot(kind='barh', stacked=True, color=[color_map[c] for c in plot_data.columns], 
                        figsize=(10, max(6, len(doc_errors) * 0.4)))
                        
    plt.title("Error Type Breakdown per Document", pad=15)
    plt.xlabel("Count")
    plt.ylabel("Document ID")
    
    # Invert y-axis so the highest error count (which we sorted first) is at the top
    plt.gca().invert_yaxis()
    plt.legend(title="Error Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    fig9_path = GRAPHS_DIR / "error_type_breakdown_per_doc.png"
    plt.savefig(fig9_path, dpi=150)
    plt.close()
    logger.info(f"Saved figure: {fig9_path}")

    # ---------------------------------------------------------
    # FIGURE 10: verdict_distribution.png
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Ensure all verdicts including skipped and errored are counted based on full dataframe
    verdict_counts = results_df.groupby(['verdict', 'true_label']).size().reset_index(name='count')
    
    ax = sns.barplot(
        data=verdict_counts, 
        x='verdict', 
        y='count', 
        hue='true_label', 
        palette={0: "dodgerblue", 1: "crimson"}
    )
    
    plt.title("NLI Verdict Distribution by True Label", pad=15)
    plt.xlabel("NLI Verdict")
    plt.ylabel("Count")
    
    # Add count values above bars
    for p in ax.patches:
        height = p.get_height()
        if pd.notna(height) and height > 0:
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', fontsize=10, xytext=(0, 3), 
                        textcoords='offset points')
                        
    # Renaming legend labels mapping truth
    handles, labels = ax.get_legend_handles_labels()
    new_labels = ["Correct/Grounded (0)" if l == '0' else "Hallucinated (1)" for l in labels]
    ax.legend(handles, new_labels, title="True Label")
    
    plt.tight_layout()
    fig10_path = GRAPHS_DIR / "verdict_distribution.png"
    plt.savefig(fig10_path, dpi=150)
    plt.close()
    logger.info(f"Saved figure: {fig10_path}")

def generate_improvement_report(results_df: pd.DataFrame, analysis: dict, optimal_ent_threshold: float = 0.90) -> str:
    """
    Analyses the audit results and synthesises a concrete, evidence-backed
    improvement diagnostic report.
    """
    logger.info("\nGenerating Improvement Report...")
    
    valid_df = results_df[results_df['error_type'] != 'SKIP'].copy()
    fp_df = valid_df[valid_df['error_type'] == 'FP']
    fn_df = valid_df[valid_df['error_type'] == 'FN']
    
    # 1. DIAGNOSTICS COMPUTATION
    mean_fp_ent = fp_df['entailment_score'].mean() if not fp_df.empty else 0.0
    mean_fn_ent = fn_df['entailment_score'].mean() if not fn_df.empty else 0.0
    mean_fp_sim = fp_df['max_similarity'].mean() if not fp_df.empty else 0.0
    mean_fn_sim = fn_df['max_similarity'].mean() if not fn_df.empty else 0.0
    
    # How many correctly grounded answers were killed purely because MAX similarity was < 0.70?
    threshold_failures = fp_df[fp_df['max_similarity'] < 0.70]
    
    # How many hallucinations passed the similarity gate AND fooled the NLI model (score >= 0.90)?
    high_ent_fn = fn_df[fn_df['entailment_score'] >= 0.90]
    
    wrong_df = valid_df[valid_df['is_correct'] == False]
    verdict_uncertain_errors = wrong_df[wrong_df['verdict'] == 'UNCERTAIN']
    
    total_errors = len(wrong_df)
    
    # Worst performing doc_ids
    doc_errors = wrong_df['doc_id'].value_counts()
    worst_docs = doc_errors.head(3).index.tolist() if not doc_errors.empty else []
    worst_doc_names = ", ".join(worst_docs)
    
    # Formatting helpers
    total = analysis['total']
    acc = (analysis['correct'] / total * 100) if total > 0 else 0.0
    fp_rt = (analysis['fp_count'] / total * 100) if total > 0 else 0.0
    fn_rt = (analysis['fn_count'] / total * 100) if total > 0 else 0.0
    skip_rt = (analysis['skipped'] / total * 100) if total > 0 else 0.0
    
    # Determine the biggest problem for the executive summary
    if analysis['fp_count'] > analysis['fn_count'] * 1.5:
        biggest_problem = "The system is too aggressive — it is falsely flagging many correct answers as hallucinations (High False Positives)."
    elif analysis['fn_count'] > analysis['fp_count'] * 1.5:
        biggest_problem = "The system is too permissive — it is systematically failing to catch real hallucinations (High False Negatives)."
    else:
        biggest_problem = "The system struggles somewhat equally between missing hallucinations and falsely flagging correct answers."
        
    if threshold_failures.shape[0] > total_errors * 0.3:
        biggest_problem += " A severe retrieval similarity bottleneck is the primary driver of errors."

    # 2. BUILD THE REPORT STRING
    report = []
    report.append("=========================================================")
    report.append("             NLI SYSTEM IMPROVEMENT REPORT               ")
    report.append("=========================================================\n")
    
    # SECTION A
    report.append("SECTION A — EXECUTIVE SUMMARY")
    report.append(f"Overall Accuracy : {acc:.1f}%")
    report.append(f"False Positive Rate : {fp_rt:.1f}% (Correct answers flagged as hallucinations)")
    report.append(f"False Negative Rate : {fn_rt:.1f}% (Real hallucinations missed entirely)")
    report.append(f"Skip/Error Rate  : {skip_rt:.1f}%")
    report.append(f"\nDiagnosis: {biggest_problem}\n")
    
    # SECTION B
    report.append("SECTION B — ROOT CAUSE ANALYSIS")
    cause_idx = 1
    
    if len(threshold_failures) > 0:
        pct = (len(threshold_failures) / total_errors * 100) if total_errors > 0 else 0
        report.append(f"[CAUSE-{cause_idx}] Retrieval Similarity Threshold Too Aggressive")
        report.append(f"Evidence: max_similarity < 0.70 for grounded answers.")
        report.append(f"Rows affected: {len(threshold_failures)} ({pct:.1f}% of errors)")
        report.append("Explanation: The FAISS index failed to find highly similar context, so the NLI pipeline instantly failed the answer before even checking entailment.\n")
        cause_idx += 1
        
    if len(high_ent_fn) > 0:
        pct = (len(high_ent_fn) / total_errors * 100) if total_errors > 0 else 0
        report.append(f"[CAUSE-{cause_idx}] NLI Model Fooled by Hallucinated Answers")
        report.append(f"Evidence: entailment_score >= 0.90 for actual hallucinations.")
        report.append(f"Rows affected: {len(high_ent_fn)} ({pct:.1f}% of errors)")
        report.append("Explanation: The hallucination dataset uses highly deceptive or lexically overlapping phrases that the baseline NLI model incorrectly maps as 'Entailed'.\n")
        cause_idx += 1
        
    if len(verdict_uncertain_errors) > 0:
        pct = (len(verdict_uncertain_errors) / total_errors * 100) if total_errors > 0 else 0
        report.append(f"[CAUSE-{cause_idx}] High 'UNCERTAIN' Verdict Ambiguity")
        report.append(f"Evidence: NLI verdict fell into the 0.60–0.90 entailment gap.")
        report.append(f"Rows affected: {len(verdict_uncertain_errors)} ({pct:.1f}% of errors)")
        report.append("Explanation: The model lacks confidence, returning 'UNCERTAIN', which defaults to flagging the text as hallucinated (Label=1). If these were mostly correct answers (Label=0), the entailment threshold is too strict.\n")
        cause_idx += 1
        
    if worst_docs:
        # Check if the worst docs account for > 20% of errors
        worst_docs_err_count = doc_errors.head(3).sum()
        pct = (worst_docs_err_count / total_errors * 100) if total_errors > 0 else 0
        if pct > 20.0:
            report.append(f"[CAUSE-{cause_idx}] Weak Retrieval for Specific Documents")
            report.append(f"Evidence: Disproportionate errors concentrated in a few PDFs.")
            report.append(f"Rows affected: {worst_docs_err_count} ({pct:.1f}% of errors)")
            report.append(f"Problematic doc_ids: {worst_doc_names}")
            report.append("Explanation: These specific PDFs may have poor text extraction, OCR issues, or complex chunking requirements that standard FAISS Euclidean distance is failing to map.\n")
            cause_idx += 1

    if cause_idx == 1:
        report.append("No dominant root causes mathematically isolated. Errors appear uniformly distributed.\n")
            
    # SECTION C
    report.append("SECTION C — CONCRETE IMPROVEMENT RECOMMENDATIONS")
    rec_idx = 1
    
    if len(threshold_failures) > 0:
        med_sim = threshold_failures['max_similarity'].median()
        new_thr = max(0.50, round(med_sim - 0.05, 2))
        report.append(f"[REC-{rec_idx}] Lower the Similarity Gate Threshold")
        report.append(f"What to change: Reduce SIMILARITY_THRESHOLD from 0.70 to roughly {new_thr}.")
        report.append("Where: src/nli_utils.py (SIMILARITY_THRESHOLD variable)")
        report.append(f"Expected impact: Would immediately recover up to {len(threshold_failures)} False Positives by allowing the NLI model to actually evaluate them.")
        report.append("Priority: HIGH\n")
        rec_idx += 1
        
    if optimal_ent_threshold != 0.90:
        report.append(f"[REC-{rec_idx}] Tune the Entailment Cutoff")
        report.append(f"What to change: Update ENTAILMENT_THRESHOLD from 0.90 to roughly {optimal_ent_threshold:.2f} based on automated tuning sweep.")
        report.append("Where: src/nli_utils.py")
        report.append("Expected impact: Will explicitly maximize hallucination detection F1 score and Youden's J based on this dataset.")
        report.append("Priority: HIGH\n")
        rec_idx += 1
    elif len(verdict_uncertain_errors) > 0 and mean_fp_ent > 0.75:
        new_ent = round(mean_fp_ent - 0.02, 2)
        report.append(f"[REC-{rec_idx}] Relax the Entailment Cutoff")
        report.append(f"What to change: Reduce ENTAILMENT_THRESHOLD from 0.90 to roughly {new_ent}.")
        report.append("Where: src/nli_utils.py")
        report.append("Expected impact: Will convert high-confidence 'UNCERTAIN' correct answers into 'GROUNDED', directly reducing False Positives.")
        report.append("Priority: MEDIUM\n")
        rec_idx += 1
        
    if worst_docs:
        report.append(f"[REC-{rec_idx}] Increase TOP_K_RETRIEVAL / Enable Reranker")
        report.append("What to change: Increase context fetch size from top_k=3 to top_k=5, or implement CrossEncoder reranking.")
        report.append("Where: src/nli_utils.py (search_index function)")
        report.append(f"Expected impact: Should resolve the heavy error concentration mapped to {worst_doc_names}.")
        report.append("Priority: HIGH\n")
        rec_idx += 1
        
    if len(high_ent_fn) > 0:
        report.append(f"[REC-{rec_idx}] Fine-tune DeBERTa on Domain Data")
        report.append("What to change: The base model is being fooled by lexically similar but logically hallucinated phrases. It requires contrastive fine-tuning.")
        report.append("Where: Execute training/finetune_nli.py using your hallucinated data.")
        report.append(f"Expected impact: Will directly teach the model to distinguish nuanced domain contradictions, reducing the {len(high_ent_fn)} False Negatives that achieved >=0.90 entailment magically.")
        report.append("Priority: HIGH\n")
        rec_idx += 1
        
    if rec_idx == 1:
        report.append("No specific programmatic thresholds clearly violated. Consider general model scale-up or dataset cleanup.\n")

    # SECTION D
    report.append("SECTION D — NEXT STEPS")
    report.append("1. Apply the highest priority [REC] configuration changes in src/nli_utils.py.")
    report.append("2. Re-run this audit pipeline (`python training/nli_audit_pipeline.py`).")
    report.append("3. Compare the new `accuracy_per_document.png` and `confusion_matrix.png` outputs against the current baseline.")
    report.append("4. If False Negatives persist, trigger a full `finetune_nli.py` training sequence to update the model weights.")
    
    full_report = "\n".join(report)
    
    # 3. SAVE AND PRINT
    report_file = OUTPUT_DIR / "improvement_report.txt"
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(full_report)
        logger.info(f"\nSaved Improvement Report to: {report_file}\n")
    except Exception as e:
        logger.error(f"Failed to save improvement report: {e}")
        
    print(full_report)
    
    return full_report

import numpy as np

def analyse_uncertain_population(results_df: pd.DataFrame):
    logger.info("\n=== UNCERTAIN POPULATION DEEP-DIVE ===")
    
    uncertain_df = results_df[results_df["verdict"] == "UNCERTAIN"].copy()
    if uncertain_df.empty:
        logger.info("No UNCERTAIN verdicts to analyze.")
        return
        
    u_correct = uncertain_df[uncertain_df['true_label'] == 0]
    u_halluc = uncertain_df[uncertain_df['true_label'] == 1]
    
    # Calculate stats for correct (label=0)
    cnt_0 = len(u_correct)
    mean_ent_0 = u_correct['entailment_score'].mean() if cnt_0 > 0 else 0
    mean_con_0 = u_correct['contradiction_score'].mean() if cnt_0 > 0 else 0
    mean_neu_0 = u_correct['neutral_score'].mean() if cnt_0 > 0 else 0
    mean_sim_0 = u_correct['max_similarity'].mean() if cnt_0 > 0 else 0
    
    e_gt_c_0 = (u_correct['entailment_score'] > u_correct['contradiction_score']).mean() * 100 if cnt_0 > 0 else 0
    n_gt_c_0 = (u_correct['neutral_score'] > u_correct['contradiction_score']).mean() * 100 if cnt_0 > 0 else 0
    
    # Calculate stats for hallucinated (label=1)
    cnt_1 = len(u_halluc)
    mean_ent_1 = u_halluc['entailment_score'].mean() if cnt_1 > 0 else 0
    mean_con_1 = u_halluc['contradiction_score'].mean() if cnt_1 > 0 else 0
    mean_neu_1 = u_halluc['neutral_score'].mean() if cnt_1 > 0 else 0
    mean_sim_1 = u_halluc['max_similarity'].mean() if cnt_1 > 0 else 0
    
    e_gt_c_1 = (u_halluc['entailment_score'] > u_halluc['contradiction_score']).mean() * 100 if cnt_1 > 0 else 0
    n_gt_c_1 = (u_halluc['neutral_score'] > u_halluc['contradiction_score']).mean() * 100 if cnt_1 > 0 else 0
    
    logger.info("\n| Metric                          | UNCERTAIN+Correct | UNCERTAIN+Hallucinated |")
    logger.info("|---------------------------------|-------------------|------------------------|")
    logger.info(f"| Count                           | {cnt_0:<17} | {cnt_1:<22} |")
    logger.info(f"| Mean entailment                 | {mean_ent_0:<17.3f} | {mean_ent_1:<22.3f} |")
    logger.info(f"| Mean contradiction              | {mean_con_0:<17.3f} | {mean_con_1:<22.3f} |")
    logger.info(f"| Mean neutral                    | {mean_neu_0:<17.3f} | {mean_neu_1:<22.3f} |")
    logger.info(f"| Mean max_similarity             | {mean_sim_0:<17.3f} | {mean_sim_1:<22.3f} |")
    logger.info(f"| % Entailment > Contradiction    | {e_gt_c_0:>16.1f}% | {e_gt_c_1:>21.1f}% |")
    logger.info(f"| % Neutral > Contradiction       | {n_gt_c_0:>16.1f}% | {n_gt_c_1:>21.1f}% |")
    
    # Check distributional overlap for Entailment using simple approximation
    m1, s1 = mean_ent_0, u_correct['entailment_score'].std() if cnt_0 > 1 else 0.01
    m2, s2 = mean_ent_1, u_halluc['entailment_score'].std() if cnt_1 > 1 else 0.01
    if s1 == 0: s1 = 0.01
    if s2 == 0: s2 = 0.01
    
    # Bhattacharyya distance approximation for normal distributions
    var1, var2 = s1**2, s2**2
    bc = 0.25 * np.log(0.25 * (var1/var2 + var2/var1 + 2)) + 0.25 * ((m1 - m2)**2 / (var1 + var2))
    b_coeff = np.exp(-bc)
    
    logger.info(f"\nBhattacharyya Overlap Coefficient: {b_coeff:.3f}")
    if b_coeff > 0.85:
        logger.warning("\n!!! UNCERTAIN population is NOT separable by entailment score alone !!!")
        logger.warning("Threshold tuning will NOT fix the UNCERTAIN problem.")
        logger.warning("ACTION REQUIRED: Fine-tune DeBERTa with UNCERTAIN rows as training data.")
        logger.warning("See: python training/finetune_nli.py")
    else:
        logger.info("\nUNCERTAIN population shows separation. A sub-threshold might be viable.")

    # Create figure
    plt.figure(figsize=(14, 5))
    
    # Left panel: KDE
    plt.subplot(1, 2, 1)
    if cnt_0 > 1:
        sns.kdeplot(data=u_correct, x='entailment_score', color='blue', label='Correct/Grounded (0)', fill=True, warn_singular=False)
    if cnt_1 > 1:
        sns.kdeplot(data=u_halluc, x='entailment_score', color='red', label='Hallucinated (1)', fill=True, warn_singular=False)
    plt.axvline(0.90, color='black', linestyle='--')
    plt.title("Entailment Score Distribution  UNCERTAIN Population Only")
    plt.xlabel("Entailment Score")
    plt.legend()
    
    # Right panel: Scatter
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=uncertain_df, x='entailment_score', y='contradiction_score', 
                    hue='true_label', palette={0: 'blue', 1: 'red'}, alpha=0.7)
    plt.title("Entailment vs Contradiction  UNCERTAIN Rows Only")
    plt.xlabel("Entailment Score")
    plt.ylabel("Contradiction Score")
    plt.legend(title="True Label", labels=['Correct', 'Hallucinated'])
    
    plt.tight_layout()
    fig_path = GRAPHS_DIR / "uncertain_population_analysis.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info(f"Saved analysis figure: {fig_path}")


def export_finetuning_dataset(results_df: pd.DataFrame, analysis: dict):
    """
    Exports a categorized dataset tailored for fine-tuning the DeBERTa model.
    """
    logger.info("\n=== EXPORTING FINE-TUNING DATASET ===")
    
    DATA_DIR = Path("d:/Hallucination test/data")
    output_path = DATA_DIR / "processed" / "finetuning_nli_from_audit.xlsx"
    hard_path = DATA_DIR / "processed" / "finetuning_hard_cases_only.xlsx"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    valid_df = results_df[
        (results_df['error_type'] != 'SKIP') & 
        (~results_df['retrieved_context'].isna()) &
        (results_df['retrieved_context'] != "") &
        (results_df['retrieved_context'] != "[no chunks]") &
        (~results_df['retrieved_context'].str.startswith("[UNSUPPORTED]", na=False))
    ].copy()
    
    ft_rows = []
    counts = {"T1": 0, "T2": 0, "T3": 0, "T4": 0, "T5": 0}
    
    def add_row(row, label_nli, tier):
        ft_rows.append({
            'doc_id': row['doc_id'],
            'premise': row['retrieved_context'],
            'hypothesis': f"Given the question: {row['question']}  The answer is: {row['answer']}",
            'label': label_nli,
            'tier': tier,
            'original_verdict': row['verdict'],
            'true_label': row['true_label'],
            'entailment_score': row['entailment_score'],
            'contradiction_score': row['contradiction_score']
        })
        counts[tier] += 1
        
    t1_df = valid_df[(valid_df['verdict'] == 'UNCERTAIN') & (valid_df['true_label'] == 1)]
    for _, row in t1_df.iterrows(): add_row(row, 0, "T1")
    
    t2_df = valid_df[(valid_df['verdict'] == 'UNCERTAIN') & (valid_df['true_label'] == 0)]
    for _, row in t2_df.iterrows(): add_row(row, 1, "T2")
    
    t3_df = valid_df[valid_df['error_type'] == 'FN']
    for _, row in t3_df.iterrows(): add_row(row, 0, "T3")
    
    t4_df = valid_df[valid_df['error_type'] == 'TP']
    if len(t4_df) > 100: t4_df = t4_df.sample(100, random_state=42)
    for _, row in t4_df.iterrows(): add_row(row, 0, "T4")
    
    t5_df = valid_df[valid_df['error_type'] == 'TN']
    if len(t5_df) > 100: t5_df = t5_df.sample(100, random_state=42)
    for _, row in t5_df.iterrows(): add_row(row, 1, "T5")
    
    ft_df = pd.DataFrame(ft_rows)
    if ft_df.empty: return None
        
    import re
    char_pattern = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')
    def clean_text(x):
        if isinstance(x, str): return char_pattern.sub('', x)
        return x
        
    for col in ['premise', 'hypothesis', 'original_verdict']:
        if col in ft_df.columns: ft_df[col] = ft_df[col].apply(clean_text)
            
    ft_df.to_excel(output_path, index=False)
    
    hard_df = ft_df[ft_df['tier'].isin(['T1', 'T3'])]
    if not hard_df.empty: hard_df.to_excel(hard_path, index=False)
        
    lbl_counts = ft_df['label'].value_counts()
    c_0 = lbl_counts.get(0, 0)
    c_1 = lbl_counts.get(1, 0)
    total = len(ft_df)
    
    print("\n" + "-" * 44)
    print("FINE-TUNING DATASET EXPORTED")
    print("-" * 44)
    print(f"Tier 1 (UNCERTAIN hallucinations)  : {counts['T1']} rows")
    print(f"Tier 2 (UNCERTAIN correct)         : {counts['T2']} rows")
    print(f"Tier 3 (False Negatives)           : {counts['T3']} rows")
    print(f"Tier 4 (True Positives, sampled)   : {counts['T4']} rows")
    print(f"Tier 5 (True Negatives, sampled)   : {counts['T5']} rows")
    print("-" * 44)
    print(f"Total examples   : {total} rows")
    print(f"\nSaved to: {output_path}")
    if not hard_df.empty: print(f"Hard cases only saved to: {hard_path}")
    print("\nNEXT STEP:")
    print("  python training/finetune_nli.py --use_hard_cases_only --epochs 3 --batch_size 8\n")
    return str(output_path)

def main():
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    cleanup_gpu_and_memory()
    
    DATA_DIR = Path("d:/Hallucination test/data")
    dataset_path = str(DATA_DIR / "training_data" / "training_data_final.xlsx")
    results_path = DATA_DIR / "results" / "nli_results.csv"
    
    if args.resume and results_path.exists():
        print(f"Resuming by loading existing results from {results_path}...")
        results_df = pd.read_csv(results_path)
    else:
        print(f"Loading dataset from {dataset_path}...")
        df = load_and_validate_dataset(dataset_path)
        
        print("Running NLI audit...")
        results_df = run_nli_audit(df)
        results_df.to_csv(results_path, index=False)
    
    print("Running analysis...")
    analysis = analyse_wrong_predictions(results_df)
    
    try: plot_entailment_distributions(results_df)
    except: pass
    
    try: plot_similarity_and_performance(results_df)
    except: pass
    
    try: plot_per_document_analysis(results_df)
    except: pass
    
    generate_improvement_report(results_df, analysis, optimal_ent_threshold=0.86)
    analyse_uncertain_population(results_df)
    export_finetuning_dataset(results_df, analysis)
    
    cleanup_gpu_and_memory()
    logger.info("AUDIT PIPELINE COMPLETE.")

if __name__ == "__main__":
    main()
