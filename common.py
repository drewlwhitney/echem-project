from pathlib import Path

FARADAY = 9.64853321233100184e4
GASSY = 8.31446261815324
C25 = 298.15  # K

figure_dir = Path(__file__).parent.joinpath("figures")
data_file = Path(__file__).parent.joinpath("data.csv")

if not figure_dir.exists():
    figure_dir.mkdir(parents=True, exist_ok=True)
