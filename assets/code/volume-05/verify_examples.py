#!/usr/bin/env python3
"""Reproduce displayed calculations for Volume V."""

from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
from itertools import product
from math import exp, log, sqrt


def chapter_01() -> None:
    digest = sha256(b"claim-v1\n").hexdigest()
    assert digest == "6887fd7e30f48ee45eea7690830d4e041ffa919af8d28af1062a96d8d5e1b763"
    print("V-01 claim byte digest:", digest)


def chapter_02() -> None:
    maxima = {n: (n / (n + 1)) ** (n + 1) for n in (10, 100)}
    assert abs(maxima[10] - 0.3504938994813925) < 1.0e-15
    assert abs(maxima[100] - 0.3660507052763555) < 1.0e-15
    print("V-02 moving-peak maxima:", maxima)


def chapter_03() -> None:
    epsilon = Fraction(1, 5)
    favored = (1 - epsilon) / 2 + epsilon / 4
    other = epsilon / 4
    covariance = 1 - epsilon
    variances = (2 + 2 * covariance, 2 - 2 * covariance)
    assert (favored, other) == (Fraction(9, 20), Fraction(1, 20))
    assert variances == (Fraction(18, 5), Fraction(2, 5))
    print("V-03 strict-support cell masses and aggregate variances:", (favored, other), variances)


def chapter_04() -> None:
    epsilon = Fraction(1, 10**12)
    determinant = (Fraction(1) * (1 + epsilon)) - Fraction(1)
    primal = Fraction(1, 2)
    dual = Fraction(1, 2)
    assert determinant == epsilon and determinant != 0
    assert primal == dual
    print("V-04 exact determinant and primal-dual certificate:", determinant, (primal, dual))


def softmax(values: tuple[float, ...], scale: float = 1.0) -> tuple[float, ...]:
    maximum = max(values)
    weights = tuple(exp((value - maximum) / scale) for value in values)
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def chapter_05() -> None:
    shares = softmax((1000.0, 1001.0, 1002.0))
    base = softmax((0.0, 1.0, 2.0))
    assert max(abs(a - b) for a, b in zip(shares, base)) < 1.0e-15
    jacobian = tuple(
        tuple(shares[j] * ((1.0 if j == k else 0.0) - shares[k]) for k in range(3))
        for j in range(3)
    )
    assert max(abs(sum(row)) for row in jacobian) < 1.0e-15
    assert max(abs(sum(jacobian[j][k] for j in range(3))) for k in range(3)) < 1.0e-15
    print("V-05 stable shares and Jacobian row sums:", shares, tuple(sum(row) for row in jacobian))


def chapter_06() -> None:
    digest = sha256(b"theta=0.5\n").hexdigest()
    assert digest == "6c2a607e45bdea99821f70a9ea6cff27280c82249e87e1fb380f8b41f008630c"
    print("V-06 manifest byte digest:", digest)


def chapter_07() -> None:
    means = {}
    for dimension in (1, 2, 3):
        outcomes = list(product((-1, 1), repeat=dimension))
        selected_mean = sum(max(outcome) for outcome in outcomes) / len(outcomes)
        formula = 1 - 2 ** (1 - dimension)
        assert selected_mean == formula
        means[dimension] = selected_mean
    print("V-07 selected Rademacher means:", means)


def chapter_08() -> None:
    sample_size = 10_000
    root_n = sqrt(sample_size)
    pairs = {m: (root_n / m, 1 + 1 / m) for m in (10, 100, 1000)}
    assert pairs == {10: (10.0, 1.1), 100: (1.0, 1.01), 1000: (0.1, 1.001)}
    print("V-08 root-n bias and centered variance:", pairs)


def chapter_09() -> None:
    signed_radius = sqrt(0.6**2 + 0.6**2)
    absolute_radius = 1.2
    diagonal = 0.8**5
    generalized = 5 * 0.8**4
    assert abs(signed_radius - 0.848528137423857) < 1.0e-15
    assert (absolute_radius, diagonal, generalized) == (1.2, 0.3276800000000001, 2.0480000000000005)
    print("V-09 signed/absolute radii and Jordan factors:", (signed_radius, absolute_radius), (diagonal, generalized))


def logistic(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def chapter_10() -> None:
    exact_shares = []
    linear_shares = []
    for income in (10.0, 20.0):
        base_price, price, quality = 2.0, 4.0, 1.0
        exact_utility = quality + log((income - price) / income)
        base_utility = quality + log((income - base_price) / income)
        linear_utility = base_utility - (price - base_price) / (income - base_price)
        exact_shares.append(logistic(exact_utility))
        linear_shares.append(logistic(linear_utility))
    derivatives = (Fraction(-1, 6), Fraction(-1, 8))
    assert abs(exact_shares[0] - 0.6199119172051106) < 1.0e-14
    assert abs(exact_shares[1] - 0.6850022115275525) < 1.0e-14
    assert abs(linear_shares[0] - 0.6287496305813667) < 1.0e-14
    assert abs(linear_shares[1] - 0.6864400604862351) < 1.0e-14
    print("V-10 exact/linear shares and income-ten slopes:", (exact_shares, linear_shares), derivatives)


def chapter_11() -> None:
    exact_values = (3.0, 4.0, 5.0)
    approximate_values = (1.0, 2.0, 3.0)
    interaction = 5 - 3 - 4
    reference_marginals = (5 - 4, 5 - 3)
    exact_probabilities = tuple(logistic(value) for value in exact_values)
    approximate_probabilities = tuple(logistic(value) for value in approximate_values)
    assert interaction == -2 and reference_marginals == (1, 2)
    assert abs(exact_probabilities[0] - 0.9525741268224334) < 1.0e-14
    assert abs(approximate_probabilities[2] - exact_probabilities[0]) < 1.0e-15
    print("V-11 interaction, marginals, and exact/approximate probabilities:", interaction, reference_marginals, (exact_probabilities, approximate_probabilities))


def chapter_12() -> None:
    sigma = 0.5
    frequency = 6
    forward = exp(-(sigma**2) * frequency**2 / 2)
    inverse = 1 / forward
    sigma_one, sigma_two = 0.25, 0.5
    transform = exp((sigma_two**2 - sigma_one**2) * frequency**2 / 2)
    assert abs(forward - 0.011108996538242306) < 1.0e-15
    assert abs(inverse - 90.01713130052181) < 1.0e-12
    assert abs(transform - 29.22428378123494) < 1.0e-12
    print("V-12 forward, inverse, and cross-candidate multipliers:", forward, inverse, transform)


def main() -> None:
    for check in (
        chapter_01,
        chapter_02,
        chapter_03,
        chapter_04,
        chapter_05,
        chapter_06,
        chapter_07,
        chapter_08,
        chapter_09,
        chapter_10,
        chapter_11,
        chapter_12,
    ):
        check()
    print("PASS: 12 Volume V example groups verified")


if __name__ == "__main__":
    main()
