import math
import random
from typing import Callable
from numba import njit
from .utils import *
# import logger
import logging

logger = logging.getLogger(__name__)


def reduce_probs(raw_probs: dict, max_values: int, shots: int) -> dict:
    additional_samples = {}
    if len(raw_probs) <= max_values:
        return raw_probs
    reduced_sample = raw_probs
    while len(reduced_sample.keys()) > max_values:
        min_prob = min(reduced_sample.values())
        keys = [k for k in reduced_sample.keys()]
        for k in keys:
            if reduced_sample[k] <= min_prob:
                additional_samples[k] = reduced_sample[k]
                reduced_sample.pop(k)
    missing = max_values - len(reduced_sample)
    if missing > 0:
        sampled_keys = random.choices(
            list(additional_samples.keys()),
            weights=list(additional_samples.values()),
            k=missing)
        reduced_sample = {k: v * shots for k, v in reduced_sample.items()}
        for k in sampled_keys:
            if k in reduced_sample.keys():
                reduced_sample[k] += 1
            else:
                reduced_sample[k] = 1
    return normalize_probs(reduced_sample)


def get_read_out_error(mitigation_probs: dict, sample: str) -> float:
    if sample not in mitigation_probs.keys():
        val = 0
    else:
        val = mitigation_probs[sample] ** (1 / len(sample))
    return 1 - val


@njit
def hamming_dist(str1: str, str2: str) -> int:
    summ = 0
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            summ += 1
    return summ


def n_states_hamming(flips: int, nq: int) -> float:
    return math.factorial(nq) / (
            math.factorial(flips) * math.factorial(nq - flips))


@njit
def p2_fun(p1, p2):
    return p2


@njit
def p1xp2_fun(p1, p2):
    return p1 * p2


@njit
def p1_frac_p2_fun(p1, p2):
    return p1 / p2


@njit
def get_activation_function(mode):
    if mode == "p2":
        return p2_fun
    elif mode == "p1*p2":
        return p1xp2_fun
    elif mode == "p1/p2":
        return p1_frac_p2_fun
    raise ValueError()


@njit
def get_chs(keys: list, values: list, max_hd: int) -> np.ndarray:
    chs = np.zeros(max_hd + 1, dtype="float64")
    for i in range(len(keys)):
        # print(i)
        for j in range(len(keys)):
            d_tmp = hamming_dist(keys[i], keys[j])
            if d_tmp <= max_hd:
                chs[d_tmp] += values[j]
    return chs


@njit
def is_reachable(
        hd: int,
        p1: float,
        p2: float,
        weights: np.ndarray) -> bool:
    if not np.all(weights[:hd + 1] > 10 ** -10):
        return False
    return p1 >= p2


@njit
def get_hammered_prob(
        keys: list,
        probs: list,
        w: np.ndarray,
        max_hd: int,
        weight: float) -> tuple[np.ndarray, np.ndarray]:
    final_values = np.zeros(len(probs), dtype="float64")
    counts = np.zeros(len(probs), dtype="float64")
    for i in range(len(keys)):
        p1 = probs[i]
        score = 0
        for j in range(len(keys)):
            d_tmp = hamming_dist(keys[i], keys[j])
            p2 = probs[j]
            if is_reachable(d_tmp, p1, p2, w) and d_tmp <= max_hd:
                if d_tmp == 0:
                    score += w[d_tmp] * p2 * weight
                else:
                    score += (w[d_tmp] ** (1 / len(keys[i]))) * p1 * p2 * weight
                counts[i] += 1
        final_values[i] = score
    if np.all(counts == 0):
        return np.array([v for v in probs], dtype="float64"), counts
    return final_values, counts


def iterate_again(
        prob_error: float,
        starting_probs: dict,
        old_probs: dict,
        new_probs: dict,
        shots: int) -> bool:
    gain = 1 - get_fidelity(new_probs, old_probs)
    size = len(new_probs)
    logger.info(f"Gain {gain}")
    summ = 0
    for k, v in starting_probs.items():
        if k in new_probs.keys() and (v - new_probs[k] > 0):
            summ += (v - new_probs[k]) * shots  # * v / v
        elif k not in new_probs.keys():
            summ += v * shots

    logger.info(f"Removed counts: {summ}")
    logger.info(f"Remaining states count: {size}")
    return 0.9 * prob_error * shots > summ and gain > 10 ** -7


class Hammer:

    def __init__(self,
                 iterative: bool = True,
                 iter_weights: bool = True,
                 max_hd: int | None = None,
                 callable1: Callable | None= None,
                 callable2: Callable | None= None,
                 ):
        self.iterative = iterative
        self.iter_weights = iter_weights
        self.max_hd = max_hd
        logger.info(f"Hamming distance threshold: {self.max_hd}")

        self.get_hammered_prob = callable1 if callable1 is not None else get_hammered_prob
        self._keep_mitigating = callable2 if callable2 is not None else iterate_again

    def _old_hammer(self,
                    ra_error,
                    probs_in: dict,
                    shots: int,
                    weight: float = 1) -> dict:
        if np.isclose(ra_error, 1):
            return probs_in

        num_qubits = len(next(iter(probs_in.keys())))
        chs = get_chs(
            list(probs_in.keys()), list(probs_in.values()), num_qubits
        )

        w = np.zeros(len(chs))
        for i in range(len(chs)):
            if chs[i] > 0:
                w[i] = 1 / chs[i]

        _keys = list(probs_in.keys())
        _values = [probs_in[k] for k in _keys]
        hammered_values, _ = self.get_hammered_prob(
            keys=_keys,
            probs=_values,
            w=w,
            max_hd=self.max_hd if self.max_hd is not None else num_qubits,
            weight=weight
        )
        probs_out = {k: hammered_values[n] for n, k in enumerate(_keys)}

        summ = sum(probs_out.values())
        norm_probs = {k: p / summ for k, p in probs_out.items()}
        norm_probs = filter_normalize_probs(norm_probs, 0.5 * 1 / shots)
        return norm_probs

    def _iter_hammer(
            self,
            ra_error: float,
            prob_error: float,
            old_probs: dict,
            shots: int) -> dict:
        Nq = len(next(iter(old_probs.keys())))
        starting_probs = {k: v for k, v in old_probs.items()}
        new_probs = None
        it = 0
        keep_mitigate = True
        while keep_mitigate:
            weight = 1 / np.exp(it * Nq) if self.iter_weights else 1
            new_probs = self._old_hammer(
                ra_error, old_probs, shots, weight=weight
            )
            logger.info(new_probs)
            keep_mitigate = self._keep_mitigating(
                prob_error,
                starting_probs,
                old_probs,
                new_probs,
                shots
            )
            if keep_mitigate:
                old_probs = new_probs
            it += 1
        if new_probs is None:
            return old_probs
        return new_probs

    def mitigate(self,
                 ra_error: float,
                 prob_error: float,
                 probs: dict,
                 shots: int) -> dict:
        if self.iterative:
            return self._iter_hammer(ra_error, prob_error, probs, shots)
        else:
            return self._old_hammer(ra_error, probs, shots)
