import numpy as np
from matplotlib import pyplot as plt

from common import C25, FARADAY, GASSY
from part2 import standard_state_potential as CATHODE_STANDARD_STATE_POTENTIAL

ANODE_STANDARD_STATE_POTENTIAL = 0.0

# these numbers seem high, but I got them from a paper
BETA = 0.5
I0_HYDROGEN = 0.01  # A/cm^2
I0_OXYGEN = 2.5e-6  # A/cm^2


def anode_butler_volmer(potentials: np.ndarray) -> np.ndarray:
    overpotentials = potentials - ANODE_STANDARD_STATE_POTENTIAL
    # TODO: check NELEC
    return I0_HYDROGEN * (
        np.exp(-BETA * 2 * FARADAY * overpotentials / (GASSY * C25))
        - np.exp((1 - BETA) * 2 * FARADAY * overpotentials / (GASSY * C25))
    )


def cathode_butler_volmer(potentials: np.ndarray) -> np.ndarray:
    overpotentials = potentials - CATHODE_STANDARD_STATE_POTENTIAL
    # TODO: check NELEC
    return I0_OXYGEN * (
        np.exp(-BETA * 4 * FARADAY * overpotentials / (GASSY * C25))
        - np.exp((1 - BETA) * 4 * FARADAY * overpotentials / (GASSY * C25))
    )


potentials = np.arange(0, CATHODE_STANDARD_STATE_POTENTIAL, 0.1)
# these values don't make any sense
print(cathode_butler_volmer(potentials))
print(anode_butler_volmer(potentials))

