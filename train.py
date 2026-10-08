from __future__ import annotations

from src.dataset_generator import generate_modulation_dataset
from src.train_model import main as train_main


def main() -> None:
    dataset_path = generate_modulation_dataset(num_samples_per_modulation=1000)
    train_main()
    print(f'Dataset created at: {dataset_path}')
    print('Training complete.')


if __name__ == '__main__':
    main()
