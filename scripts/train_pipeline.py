"""
End-to-end training script.
Usage:
    python scripts/train_pipeline.py                    # train ensemble
    python scripts/train_pipeline.py --model xgb        # specific model
    python scripts/train_pipeline.py --build-index      # also build search index
    python scripts/train_pipeline.py --model ensemble --build-index
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.models.training import train
from src.data.preprocessing import load_raw_data, clean_data
from src.features.engineering import add_engineered_features


def main():
    parser = argparse.ArgumentParser(description="Train churn prediction pipeline")
    parser.add_argument("--model", default="ensemble",
                        choices=["rf", "xgb", "lgbm", "lr", "ensemble"],
                        help="Model type to train")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--no-save", action="store_true", help="Don't save artifacts")
    parser.add_argument("--build-index", action="store_true",
                        help="Build semantic search index after training")
    args = parser.parse_args()

    logger.info(f"Training model: {args.model}")
    results = train(model_type=args.model, save=not args.no_save, cv_folds=args.cv_folds)

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"  Model:       {results['model_type']}")
    print(f"  Test AUC:    {results['test']['roc_auc']:.4f}")
    print(f"  Test F1:     {results['test']['f1']:.4f}")
    print(f"  CV AUC:      {results['cv_auc_mean']:.4f} ± {results['cv_auc_std']:.4f}")
    print(f"  Duration:    {results['training_time_s']}s")
    print("=" * 50)

    if args.build_index:
        logger.info("Building semantic search index…")
        from src.search.semantic_search import SemanticSearchEngine
        from src.config import settings

        df = add_engineered_features(clean_data(load_raw_data()))
        engine = SemanticSearchEngine()
        engine.build_index(df)
        engine.save_index(settings.MODEL_ARTIFACT_PATH / "search_index.npz")
        logger.info("Search index built and saved")

    return results


if __name__ == "__main__":
    main()
