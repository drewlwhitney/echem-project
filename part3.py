from math import exp

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from scipy import optimize

from common import C25, FARADAY, GASSY, data_file, figure_dir
from part2 import standard_state_potential as STANDARD_STATE_POTENTIAL

# these numbers seem high, but I got them from a paper
BETA = 0.5
I0_HYDROGEN = 0.01  # A/cm^2
I0_OXYGEN = 2.5e-6  # A/cm^2


def anode_butler_volmer(overpotentials: float) -> float:
    return I0_HYDROGEN * (
        exp(-BETA * 2 * FARADAY * overpotentials / (GASSY * C25))
        - exp(BETA * 2 * FARADAY * overpotentials / (GASSY * C25))
    )


def cathode_butler_volmer(overpotential: float) -> float:
    return I0_OXYGEN * (
        exp(-BETA * 4 * FARADAY * overpotential / (GASSY * C25))
        - exp((1 - BETA) * 4 * FARADAY * overpotential / (GASSY * C25))
    )


def main():
    currents = np.arange(0, 0.32, 0.001)

    anode_overpotentials = np.vectorize(
        lambda current: (
            optimize.root_scalar(
                lambda v: anode_butler_volmer(v) - (-current),
                bracket=(0, STANDARD_STATE_POTENTIAL),
            ).root
        )
    )(currents)

    cathode_overpotentials = np.vectorize(
        lambda current: (
            optimize.root_scalar(
                lambda v: cathode_butler_volmer(v) - (-current),
                bracket=(0, STANDARD_STATE_POTENTIAL),
            ).root
        )
    )(currents)

    kinetics_voltages = (
        STANDARD_STATE_POTENTIAL - cathode_overpotentials - anode_overpotentials
    )

    # load the literature data and plot it with the OCV
    data = pl.scan_csv(
        data_file,
        has_header=False,
        new_columns=["current_density", "voltage"],
        schema_overrides=[pl.Float64, pl.Float64],
    ).collect()

    figure, axis = plt.subplots()

    # literature data
    axis.plot(
        data["current_density"] * 10,  # data is scaled wrong
        data["voltage"],
        marker=".",
        label="Literature Data",
        linestyle="None",
    )

    # kinetics
    axis.plot(
        currents * 1000,  # A -> mA
        kinetics_voltages,
        label="Kinetcs",
    )

    # OCV
    axis.axhline(
        STANDARD_STATE_POTENTIAL, color="black", linestyle="--", label="OCV"
    )

    axis.set_ylim(0)
    axis.margins(0, 0)
    axis.set_xlabel("Current Density (mA/cm²)")
    axis.set_ylabel("Voltage (V)")
    axis.legend()
    figure.set_size_inches(5, 5)
    figure.tight_layout()

    figure.savefig(figure_dir.joinpath("data-and-OCV-and-kinetics.svg"))

    plt.show()


if __name__ == "__main__":
    main()
