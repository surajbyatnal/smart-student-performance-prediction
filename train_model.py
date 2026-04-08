"""
train_model.py
==============
Run this once before starting the Flask server to pre-train and
persist the Decision Tree model to disk.

Usage:
    python train_model.py
    python train_model.py --samples 3000
"""

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

from model import StudentModel


def main():
    parser = argparse.ArgumentParser(description="Train EduSense Decision Tree model")
    parser.add_argument(
        "--samples", type=int, default=1500,
        help="Number of synthetic training samples (default: 1500)"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to CSV file for training data (if provided, uses real data instead of synthetic)"
    )
    args = parser.parse_args()

    m = StudentModel()
    if args.csv:
        m.train(csv_path=args.csv)
    else:
        m.train(n_samples=args.samples)

    print("\n" + "=" * 55)
    print("  Model training complete!")
    print(f"  Training samples : {m.training_size}")
    print(f"  Test accuracy    : {m.accuracy * 100:.1f}%")
    print(f"  Max depth        : {m.clf.max_depth}")
    print("  Saved to         : edusense_model.pkl")
    print("=" * 55)
    print("\nFeature importances:")
    importances = m.clf.feature_importances_
    for name, imp in sorted(
        zip(m.FEATURE_NAMES, importances), key=lambda x: -x[1]
    ):
        bar = "█" * int(imp * 40)
        print(f"  {name:<15} {imp*100:5.1f}%  {bar}")
    print()


if __name__ == "__main__":
    main()
