from pathlib import Path

figure_dir = Path(__file__).parent.joinpath("figures")
data_file = Path(__file__).parent.joinpath("data.csv")

if not figure_dir.exists():
    figure_dir.mkdir(parents=True, exist_ok=True)
