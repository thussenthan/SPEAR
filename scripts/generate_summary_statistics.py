#!/usr/bin/env python3
"""
Generate summary statistics for all SPEAR model runs.
This script scans output/results/ and creates a comprehensive summary table.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
from typing import Optional
import sys


def find_metrics_files(results_root: Path) -> list[dict]:
    """Discover all metrics files in the results directory."""
    records = []
    
    for run_dir in sorted(results_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "archive":
            continue
            
        run_name = run_dir.name
        models_dir = run_dir / "models"
        
        if not models_dir.exists():
            continue
            
        # Parse run metadata from name
        parts = run_name.split("_")
        gene_count = None
        dataset = None
        partition = None
        model_type = None
        
        for i, part in enumerate(parts):
            if "genes" in part.lower():
                gene_count = part.replace("genes", "")
            elif part in ["embryonic", "endothelial"]:
                dataset = part
            elif part in ["cpu", "gpu", "dense", "compute"]:
                partition = part
            elif i == len(parts) - 1:  # last element is often the model
                model_type = part
                
        for model_dir in sorted(models_dir.iterdir()):
            if not model_dir.is_dir():
                continue
                
            model_id = model_dir.name
            
            # Look for metrics files
            metrics_file = None
            for candidate in ["metrics_per_gene.csv", "metrics_by_gene.csv", "metrics_cv.csv"]:
                if (model_dir / candidate).exists():
                    metrics_file = model_dir / candidate
                    break
                    
            # Look for predictions
            predictions_file = None
            for candidate in ["predictions_raw.csv", "predictions.csv"]:
                if (model_dir / candidate).exists():
                    predictions_file = model_dir / candidate
                    break
                    
            # Look for training history
            history_file = None
            for candidate in ["training_history.csv", "training_history_loss.csv"]:
                if (model_dir / candidate).exists():
                    history_file = model_dir / candidate
                    break
                    
            # Check for run configuration
            config_file = run_dir / "run_configuration.json"
            config = None
            if config_file.exists():
                try:
                    with open(config_file) as f:
                        config = json.load(f)
                except Exception:
                    pass
                    
            records.append({
                "run_name": run_name,
                "model_id": model_id,
                "gene_count": gene_count,
                "dataset": dataset,
                "partition": partition,
                "model_type": model_type or model_id,
                "metrics_file": metrics_file,
                "predictions_file": predictions_file,
                "history_file": history_file,
                "config": config,
                "run_path": run_dir,
                "model_path": model_dir,
            })
            
    return records


def extract_metrics_summary(metrics_file: Path) -> Optional[dict]:
    """Extract summary statistics from a metrics file."""
    try:
        df = pd.read_csv(metrics_file)
        
        # Check if there's a split column
        has_split = 'split' in df.columns
        
        summary = {
            'num_genes_total': len(df),
            'file_exists': True,
        }
        
        # If there's a split column, aggregate by split
        if has_split:
            for split in ['test', 'val', 'train']:
                split_df = df[df['split'] == split]
                if len(split_df) == 0:
                    continue
                    
                summary[f'num_genes_{split}'] = len(split_df)
                
                for metric in ['pearson', 'spearman', 'r2', 'mse', 'rmse', 'mae']:
                    if metric in split_df.columns:
                        values = split_df[metric].dropna()
                        if len(values) > 0:
                            summary[f'{split}_{metric}_mean'] = values.mean()
                            summary[f'{split}_{metric}_median'] = values.median()
                            summary[f'{split}_{metric}_std'] = values.std()
                            summary[f'{split}_{metric}_min'] = values.min()
                            summary[f'{split}_{metric}_max'] = values.max()
        else:
            # No split column - aggregate all rows
            for metric in ['pearson', 'spearman', 'r2', 'mse', 'rmse', 'mae']:
                # Try both direct column name and with test_ prefix
                col = None
                if metric in df.columns:
                    col = metric
                elif f'test_{metric}' in df.columns:
                    col = f'test_{metric}'
                    
                if col:
                    values = df[col].dropna()
                    if len(values) > 0:
                        summary[f'test_{metric}_mean'] = values.mean()
                        summary[f'test_{metric}_median'] = values.median()
                        summary[f'test_{metric}_std'] = values.std()
                        summary[f'test_{metric}_min'] = values.min()
                        summary[f'test_{metric}_max'] = values.max()
                
        return summary
        
    except Exception as e:
        print(f"Error reading {metrics_file}: {e}", file=sys.stderr)
        return {'file_exists': False, 'error': str(e)}


def generate_summary_table(results_root: Path, output_path: Path) -> pd.DataFrame:
    """Generate comprehensive summary table of all runs."""
    print(f"Scanning {results_root} for model results...")
    
    records = find_metrics_files(results_root)
    print(f"Found {len(records)} model runs")
    
    summary_data = []
    
    for record in records:
        row = {
            'run_name': record['run_name'],
            'model_id': record['model_id'],
            'model_type': record['model_type'],
            'dataset': record['dataset'],
            'gene_count': record['gene_count'],
            'partition': record['partition'],
            'has_metrics': record['metrics_file'] is not None,
            'has_predictions': record['predictions_file'] is not None,
            'has_history': record['history_file'] is not None,
            'run_path': str(record['run_path']),
            'model_path': str(record['model_path']),
        }
        
        # Extract metrics if available
        if record['metrics_file']:
            metrics = extract_metrics_summary(record['metrics_file'])
            if metrics:
                row.update(metrics)
                
        summary_data.append(row)
        
    df = pd.DataFrame(summary_data)
    
    # Sort by dataset, gene count, and performance
    sort_cols = ['dataset', 'gene_count', 'model_id']
    if 'test_pearson_mean' in df.columns:
        sort_cols.append('test_pearson_mean')
        df = df.sort_values(sort_cols, ascending=[True, True, True, False])
    else:
        df = df.sort_values(sort_cols[:3])
        
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nSummary table saved to: {output_path}")
    
    return df


def print_summary_report(df: pd.DataFrame):
    """Print a human-readable summary report."""
    print("\n" + "="*80)
    print("SPEAR MODEL RUNS SUMMARY")
    print("="*80)
    
    print(f"\nTotal runs found: {len(df)}")
    
    # Group by dataset
    if 'dataset' in df.columns:
        print("\nRuns by dataset:")
        for dataset, group in df.groupby('dataset'):
            if pd.notna(dataset):
                print(f"  {dataset}: {len(group)} runs")
                
    # Group by gene count
    if 'gene_count' in df.columns:
        print("\nRuns by gene count:")
        for gene_count, group in df.groupby('gene_count'):
            if pd.notna(gene_count):
                print(f"  {gene_count} genes: {len(group)} runs")
                
    # Group by model type
    if 'model_type' in df.columns:
        print("\nRuns by model type:")
        for model, group in df.groupby('model_type'):
            if pd.notna(model):
                print(f"  {model}: {len(group)} runs")
                
    # Completion status
    print("\nCompletion status:")
    print(f"  With metrics: {df['has_metrics'].sum()} / {len(df)}")
    print(f"  With predictions: {df['has_predictions'].sum()} / {len(df)}")
    print(f"  With training history: {df['has_history'].sum()} / {len(df)}")
    
    # Performance summary if available
    if 'test_pearson_mean' in df.columns:
        completed = df[df['has_metrics'] == True].copy()
        
        if len(completed) > 0:
            print("\n" + "="*80)
            print("PERFORMANCE SUMMARY (Completed Runs Only)")
            print("="*80)
            
            for dataset in completed['dataset'].dropna().unique():
                dataset_df = completed[completed['dataset'] == dataset]
                
                print(f"\n{dataset.capitalize()} Dataset:")
                print("-" * 40)
                
                for gene_count in sorted(dataset_df['gene_count'].dropna().unique()):
                    gene_df = dataset_df[dataset_df['gene_count'] == gene_count]
                    
                    if len(gene_df) > 0:
                        print(f"\n  {gene_count} genes:")
                        
                        # Top 5 models by test Pearson
                        top_models = gene_df.nlargest(5, 'test_pearson_mean')[
                            ['model_id', 'test_pearson_mean', 'test_pearson_std']
                        ]
                        
                        for idx, row in top_models.iterrows():
                            mean_val = row['test_pearson_mean']
                            std_val = row.get('test_pearson_std', np.nan)
                            if pd.notna(std_val):
                                print(f"    {row['model_id']:30s}: {mean_val:.4f} ± {std_val:.4f}")
                            else:
                                print(f"    {row['model_id']:30s}: {mean_val:.4f}")
                                
    print("\n" + "="*80)


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    results_root = project_root / "output" / "results"
    reports_dir = project_root / "analysis" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = reports_dir / "summary_metrics_all_models.csv"
    
    if not results_root.exists():
        print(f"Error: Results directory not found: {results_root}", file=sys.stderr)
        sys.exit(1)
        
    df = generate_summary_table(results_root, output_path)
    print_summary_report(df)
    
    print(f"\nFor detailed analysis, see: {output_path}")
    

if __name__ == "__main__":
    main()
