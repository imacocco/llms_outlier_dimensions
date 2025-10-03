# main.py

import argparse
import yaml

from src import preprocess, running, evaluate, analysis  # Adjust if your structure differs

def parse_args():
    parser = argparse.ArgumentParser(description="Run the full pipeline")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def main():
    args = parse_args()
    config = load_config(args.config)

    print("Starting preprocessing...")
    batches = preprocess.run(config)

    print("Starting model run...")
    model = running.run(batches, config)

    print("Starting evaluation...")
    evaluate.run(config)

    print("Starting analysis and plotting...")
    analysis.run(config)


    print("✅ Pipeline finished successfully.")

if __name__ == '__main__':
    main()
