from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_utils import PROJECT_ROOT, train_and_export


def main() -> None:
    metadata = train_and_export(random_state=42)
    summary = pd.DataFrame.from_dict(metadata["metrics"], orient="index")
    print("Saved model artifacts to:", Path(PROJECT_ROOT / "model").resolve())
    print("Saved test data to:", Path(PROJECT_ROOT / "test_data.csv").resolve())
    print()
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
