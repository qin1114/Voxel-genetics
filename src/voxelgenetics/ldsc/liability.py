"""
voxelgenetics.pheno.liability
-----------------------------

Functions for converting observed-scale heritability and genetic covariance
to the liability scale.
"""

from scipy.stats import norm


def convert_to_liability_scale(h2_obs, samp_prev, pop_prev):
    """
    Convert observed-scale heritability to liability-scale heritability.

    Parameters
    ----------
    h2_obs : float or ndarray
        Observed-scale heritability (or genetic covariance).
    samp_prev : float
        Sample prevalence (proportion of cases in the sample).
    pop_prev : float
        Population prevalence (proportion of cases in the general population).

    Returns
    -------
    h2_liab : float or ndarray
        Liability-scale heritability (or genetic covariance).
    conversion_factor : float
        The conversion factor used.
    """
    if not (0 < samp_prev < 1):
        raise ValueError(f"`samp_prev` must be between 0 and 1, got {samp_prev}")
    if not (0 < pop_prev < 1):
        raise ValueError(f"`pop_prev` must be between 0 and 1, got {pop_prev}")

    # Calculate the threshold on the standard normal distribution
    threshold = norm.ppf(pop_prev)
    
    # Calculate the density at the threshold
    z = norm.pdf(threshold)

    K = pop_prev
    P = samp_prev

    # Formula: K^2 * (1-K)^2 / [P * (1-P) * z^2]
    numerator = (K ** 2) * ((1 - K) ** 2)
    denominator = P * (1 - P) * (z ** 2)
    
    conversion_factor = numerator / denominator

    h2_liab = h2_obs * conversion_factor

    return h2_liab, conversion_factor