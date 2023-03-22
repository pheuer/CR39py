"""
Detector response functions for CR39
"""

import numpy as np

from cr39py.util.units import unit_registry as u


__all__ = ["track_energy", "track_diameter"]


def track_energy(diameter, charge_number, atomic_number, etch_time, vB=2.66, k=0.8, n=1.2):
    """
    Calculate the energy of a particle given the diameter of the track it leaves in CR39.
    see B. Lahmann et al. *Rev. Sci. Instrum.* 91, 053502 (2020); doi: 10.1063/5.0004129.
    :param diameter: the track diameter (μm)
    :param p: The particle type (string)
    :param etch_time: the time over which the CR39 was etched in NaOH (hours)
    :param vB: the bulk-etch speed (μm/hour)
    :param k: one of the response parameters in the two-parameter model
    :param n: the other response parameter in the two-parameter model
    """

    p = Particle(particle)

    etch_time = etch_time.m_as(u.hour)

    return (
        charge_number**2
        * atomic_number
        * ((2 * etch_time * vB / diameter - 1) / k) ** (1 / n)
        * u.MeV
    )


def track_diameter(energy, charge_number, atomic_number, etch_time, vB=2.66, k=0.8, n=1.2):
    """
    calculate the diameter of the track left in CR39 by a particle of a given energy

    :param energy: the particle energy (MeV)
    :param p: The particle type (string)
    :param etch_time: the time over which the CR39 was etched in NaOH (hours)
    :param vB: the bulk-etch speed (μm/hour)
    :param k: one of the response parameters in the two-parameter model
    :param n: the other response parameter in the two-parameter model
    """

    energy = energy.m_as(u.MeV)

    return np.where(
        energy > 0,
        2
        * etch_time
        * vB
        / (1 + k * (energy / (charge_number**2 * atomic_number)) ** n),
        np.nan,
    )
