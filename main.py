#!/usr/bin/env python3
"""
main.py - AI-Powered Medical Image Analysis | Command Line Interface
====================================================================
One entry point for the entire pipeline.  This is the file recruiters will
run first - keep it simple and well documented.

Usage
-----
    python main.py simulate               # generate the virtual dataset
    python main.py train                  # train the CNN
    python main.py evaluate               # test metrics + confusion matrix + ROC
    python main.py predict                # sample predictions + Grad-CAM grid
    python main.py predict --image path   # predict a single X-ray image
    python main.py explore                # dataset exploration figures
    python main.py demo                   # EVERYTHING: simulate -> train -> evaluate -> predict

    python main.py train --model mobilenetv2   # transfer-learning option
    python main.py train --epochs 15           # custom epoch count
"""

import argparse
from pathlib import Path

from src import config


def cmd_simulate(args):
    """Generate the virtual simulation dataset."""
    from src.simulate_data import generate_dataset
    generate_dataset(config.DATA_DIR, args.size, args.train, args.val, args.test, config.SEED)


def cmd_explore(_args):
    """Create dataset-exploration figures (samples, distribution, preprocessing)."""
    from src import visualize
    config.ensure_dirs()
    visualize.plot_sample_grid()
    visualize.plot_class_distribution()
    visualize.plot_preprocessing_demo()
    print("[main] Exploration figures saved to outputs/figures/")


def cmd_train(args):
    """Train the model and save it to models/."""
    from src.train import train_model
    train_model(model_type=args.model, epochs=args.epochs)


def cmd_evaluate(args):
    """Evaluate the trained model on the test set."""
    from src.evaluate import evaluate_model
    evaluate_model(model_path=args.model_path)


def cmd_predict(args):
    """Single-image prediction OR the balanced sample grid."""
    from src.predict import predict_image, predict_sample_grid
    if args.image:
        result = predict_image(args.image)
        print("\nPREDICTION RESULT")
        print("-" * 40)
        for key, value in result.items():
            print(f"  {key:24s}: {value}")
    else:
        records = predict_sample_grid(n=args.n)
        print(f"\n[main] Predicted {len(records)} test samples "
              f"-> outputs/figures/prediction_grid.png")


def cmd_demo(args):
    """Full end-to-end run - perfect for the Day-5 proof video."""
    cmd_simulate(args)
    cmd_explore(args)
    cmd_train(argparse.Namespace(model="cnn", epochs=args.epochs))
    cmd_evaluate(argparse.Namespace(model_path=None))
    cmd_predict(argparse.Namespace(image=None, n=8))
    print("\n" + "=" * 60)
    print("DEMO COMPLETE - all artefacts are in outputs/ and models/")
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI-Powered Medical Image Analysis (Pneumonia Detection from X-Rays)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_sim = sub.add_parser("simulate", help="generate the virtual X-ray dataset")
    p_sim.add_argument("--train", type=int, default=400, help="train images per class")
    p_sim.add_argument("--val", type=int, default=80, help="val images per class")
    p_sim.add_argument("--test", type=int, default=100, help="test images per class")
    p_sim.add_argument("--size", type=int, default=config.IMG_SIZE)
    p_sim.set_defaults(func=cmd_simulate)

    p_exp = sub.add_parser("explore", help="dataset exploration figures")
    p_exp.set_defaults(func=cmd_explore)

    p_tr = sub.add_parser("train", help="train the pneumonia detector")
    p_tr.add_argument("--model", choices=["cnn", "mobilenetv2"], default="cnn")
    p_tr.add_argument("--epochs", type=int, default=config.EPOCHS)
    p_tr.set_defaults(func=cmd_train)

    p_ev = sub.add_parser("evaluate", help="evaluate on the test set")
    p_ev.add_argument("--model-path", type=str, default=None)
    p_ev.set_defaults(func=cmd_evaluate)

    p_pr = sub.add_parser("predict", help="predict one image or a sample grid")
    p_pr.add_argument("--image", type=str, default=None, help="path to one X-ray image")
    p_pr.add_argument("--n", type=int, default=8, help="samples for the grid mode")
    p_pr.set_defaults(func=cmd_predict)

    p_dm = sub.add_parser("demo", help="run the complete pipeline end-to-end")
    p_dm.add_argument("--epochs", type=int, default=config.EPOCHS)
    p_dm.add_argument("--train", type=int, default=400)
    p_dm.add_argument("--val", type=int, default=80)
    p_dm.add_argument("--test", type=int, default=100)
    p_dm.add_argument("--size", type=int, default=config.IMG_SIZE)
    p_dm.set_defaults(func=cmd_demo)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    config.ensure_dirs()
    args.func(args)


if __name__ == "__main__":
    main()
