import numpy as np
import logging

logger = logging.getLogger(__name__)


def check_count(counts: dict, shots: int) -> None:
    c = 0
    for v in counts.values():
        if v < 0:
            raise ValueError("Negative count!")
        c += v
    if c != shots:
        raise ValueError("Wrong count!")


def check_probs(probs: dict) -> None:
    c = 0
    for v in probs.values():
        if v < 0:
            raise ValueError(f"Negative probability!\n{probs}")
        c += v
    if not np.isclose(c, 1):
        raise ValueError(f"Probability not normalised!\n{probs}")


def get_probability_ratio(A: dict, B: dict, to_consider: int) -> float:
    min_prob = sorted(A.values())[min(len(A.keys()) - 1, to_consider - 1)]
    normalizer = sum([v for v in A.values() if v >= min_prob])
    return (sum([np.sqrt(A[k] * B[k]) / normalizer
                 for k in A.keys() if k in B.keys() and A[k] >= min_prob]) ** 2)


def get_fidelity(A: dict, B: dict) -> float:
    return sum([np.sqrt(A[k] * B[k]) for k in A.keys() if k in B.keys()]) ** 2


def filter_normalize_probs(
        probs: dict,
        threshold: float | int,
        verbose: bool = True) -> dict:
    return normalize_probs(filter_probs(probs, threshold, verbose=verbose))


def normalize_probs(probs: dict) -> dict:
    normalizer = sum(probs.values())
    norm_probs = {k: v / normalizer for k, v in probs.items()}
    check_probs(norm_probs)
    return norm_probs


def filter_probs(probs: dict, threshold: float, verbose: bool = True) -> dict:
    filtered_probs = {k: v for k, v in probs.items() if v >= threshold}
    percentage = (
            100 * (len(probs.keys()) - len(filtered_probs.keys()))
            / len(probs.keys())
    )
    if verbose:
        logger.info(f"\n number of removed states after filtering (%): "
                    f"{percentage}\n")
    return filtered_probs


def keep_left_keys(A: dict, B: dict) -> dict:
    return {k: v for k, v in B.items() if k in A.keys()}
