import numpy as np
import polars as pl
from matplotlib import pyplot as plt

from common import data_file, figure_dir

N_ELEC = 2
FARADAY = 9.64853321233100184e4
GASSY = 8.31446261815324
C25 = 298.15  # K

# calculate open-circuit potential and implement the Nernst equation
gibbs_reaction = -237.129 * 1000  # kJ -> J
print(f"Gibbs = {gibbs_reaction / 1000:.3f} kJ/mol")
standard_state_potential = -gibbs_reaction / (N_ELEC * FARADAY)
print(f"U° = {standard_state_potential:.3f} V")


def nernst(atmospheres: np.ndarray) -> np.ndarray:
    """
    Calculate the Nernst potential based on the Hydrogen pressure in
    atmospheres.
    """
    return standard_state_potential - GASSY * C25 / (N_ELEC * FARADAY) * np.log(
        np.reciprocal(atmospheres)
    )


figure, axis = plt.subplots()

atmospheres = np.arange(0.2, 50, 1e-4)

axis.plot(atmospheres, nernst(atmospheres), label="OCV")
axis.plot(
    1.0, standard_state_potential, marker="o", label="Standard State Potential"
)

axis.set_xlabel("Hydrogen Pressure (atm)")
axis.set_ylabel("Open Circuit Voltage (V)")
axis.margins(0, 0)
axis.legend()
figure.set_size_inches(5, 5)
figure.tight_layout()

figure.savefig(figure_dir.joinpath("nernst.svg"))

# load the literature data and plot it with the OCV
data = pl.scan_csv(
    data_file,
    has_header=False,
    new_columns=["current_density", "voltage"],
    schema_overrides=[pl.Float64, pl.Float64],
).collect()
print(data)

figure, axis = plt.subplots()

axis.scatter(
    data["current_density"],
    data["voltage"],
    marker=".",
    label="Literature Data",
)

axis.axhline(
    standard_state_potential, color="black", linestyle="--", label="OCV"
)

axis.set_ylim(0)
axis.margins(0, 0)
axis.set_xlabel("Current Density (mA/cm²)")
axis.set_ylabel("Voltage (V)")
axis.legend()
figure.set_size_inches(5, 5)
figure.tight_layout()

figure.savefig(figure_dir.joinpath("data-and-OCV.svg"))
