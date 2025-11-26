#!/usr/bin/env python3
"""
Convenience script to run Petri evaluations for ALERTBENCH.

This script provides an easy way to run Petri audits on your question dataset.
"""

import sys
import os

# Add current directory (petri/) to path
sys.path.insert(0, os.path.dirname(__file__))

from petri_integration import run_alertbench_petri_evaluation

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run ALERTBENCH evaluation using Petri multi-turn audits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with default settings (run from project root)
  python petri/run_petri_eval.py --questions src/alertbench/experiment-1.csv
  
  # Test multiple models
  python petri/run_petri_eval.py --questions src/alertbench/experiment-1.csv \\
    --target-models anthropic/claude-3-7-sonnet-20250219 openai/gpt-4o
  
  # Use contrastive testing (self vs. other)
  python petri/run_petri_eval.py --questions src/alertbench/experiment-1.csv --contrastive
  
  # Custom output directory
  python petri/run_petri_eval.py --questions src/alertbench/experiment-1.csv \\
    --output-dir ./my_petri_results
        """
    )
    
    parser.add_argument(
        "--questions",
        required=True,
        help="Path to questions CSV file (must have Category and Question columns)"
    )
    parser.add_argument(
        "--target-models",
        nargs="+",
        default=["anthropic/claude-3-7-sonnet-20250219"],
        help="Target models to test (e.g., anthropic/claude-3-7-sonnet-20250219)"
    )
    parser.add_argument(
        "--output-dir",
        default="./petri_outputs",
        help="Output directory for transcripts and results"
    )
    parser.add_argument(
        "--contrastive",
        action="store_true",
        help="Use contrastive testing: compare responses to self-targeted vs. other-targeted questions"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=40,
        help="Maximum conversation turns per audit (default: 40)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ALERTBENCH - Petri Integration")
    print("=" * 70)
    print(f"Questions: {args.questions}")
    print(f"Target Models: {', '.join(args.target_models)}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Contrastive Testing: {args.contrastive}")
    print(f"Max Turns: {args.max_turns}")
    print("=" * 70)
    print()
    
    if not os.path.exists(args.questions):
        print(f"❌ Error: Questions file not found: {args.questions}")
        sys.exit(1)
    
    try:
        results_df = run_alertbench_petri_evaluation(
            questions_csv=args.questions,
            target_models=args.target_models,
            output_dir=args.output_dir,
            use_contrastive=args.contrastive,
            max_turns=args.max_turns
        )
        
        if not results_df.empty:
            print("\n" + "=" * 70)
            print("Summary Statistics")
            print("=" * 70)
            print(f"Total transcripts: {len(results_df)}")
            print(f"Adversarial responses: {results_df['adversarial'].sum()} ({results_df['adversarial'].mean()*100:.1f}%)")
            print(f"Cooperative responses: {(results_df['adversarial']==0).sum()} ({(results_df['adversarial']==0).mean()*100:.1f}%)")
            
            if "category" in results_df.columns:
                print("\nBy Category:")
                for cat in results_df["category"].unique():
                    cat_df = results_df[results_df["category"] == cat]
                    adv_pct = cat_df["adversarial"].mean() * 100
                    print(f"  {cat}: {cat_df['adversarial'].sum()}/{len(cat_df)} ({adv_pct:.1f}% adversarial)")
            
            print("\n" + "=" * 70)
            print(f"✅ Evaluation complete! Results saved to {args.output_dir}/petri_results.csv")
            print("=" * 70)
        else:
            print("\n⚠️  No results collected. Check error messages above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

