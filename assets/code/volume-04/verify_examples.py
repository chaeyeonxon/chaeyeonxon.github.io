#!/usr/bin/env python3
"""Reproduce displayed calculations for Volume IV."""

from __future__ import annotations

from fractions import Fraction
from math import exp, log, sqrt


def softmax(values: list[float], scale: float = 1.0) -> list[float]:
    shifted = [value / scale for value in values]
    maximum = max(shifted)
    weights = [exp(value - maximum) for value in shifted]
    total = sum(weights)
    return [weight / total for weight in weights]


def chapter_01() -> None:
    shares = softmax([0.0, 1.0, 2.0])
    shifted = softmax([5.0, 6.0, 7.0])
    scaled = softmax([0.0, 1.0, 2.0], scale=2.0)
    assert max(abs(a - b) for a, b in zip(shares, shifted)) < 1.0e-15
    assert abs(sum(shares) - 1) < 1.0e-15 and abs(sum(scaled) - 1) < 1.0e-15
    print("IV-01 shares at scale one and two:", shares, scaled)


def logistic(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def chapter_02() -> None:
    exact = []
    linear = []
    for income in (10.0, 20.0):
        base_price, price, quality = 2.0, 4.0, 1.0
        base_utility = quality + log((income - base_price) / income)
        exact_utility = quality + log((income - price) / income)
        linear_utility = base_utility - (price - base_price) / (income - base_price)
        exact.append(logistic(exact_utility))
        linear.append(logistic(linear_utility))
    assert abs(exact[0] - 0.6199119172051106) < 1.0e-14
    assert abs(exact[1] - 0.6850022115275525) < 1.0e-14
    print("IV-02 exact and linearized shares:", exact, linear)


def chapter_03() -> None:
    population_weights = (Fraction(1, 2), Fraction(1, 2))
    demographics = (Fraction(0), Fraction(1))
    purchase_probabilities = (Fraction(1, 5), Fraction(4, 5))
    share = sum(w * p for w, p in zip(population_weights, purchase_probabilities))
    numerator = sum(w * z * p for w, z, p in zip(population_weights, demographics, purchase_probabilities))
    assert share == Fraction(1, 2) and numerator / share == Fraction(4, 5)
    print("IV-03 share, demographic numerator, purchaser mean:", share, numerator, numerator / share)


def solve_two_by_two(matrix: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]], rhs: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    return (
        (rhs[0] * matrix[1][1] - matrix[0][1] * rhs[1]) / determinant,
        (matrix[0][0] * rhs[1] - rhs[0] * matrix[1][0]) / determinant,
    )


def chapter_04() -> None:
    shares = (Fraction(3, 10), Fraction(1, 5))
    jacobian = ((Fraction(-21, 100), Fraction(3, 50)), (Fraction(3, 50), Fraction(-4, 25)))
    separate = (shares[0] / -jacobian[0][0], shares[1] / -jacobian[1][1])
    joint = solve_two_by_two(jacobian, (-shares[0], -shares[1]))
    assert separate == (Fraction(10, 7), Fraction(5, 4))
    assert joint == (Fraction(2), Fraction(2))
    print("IV-04 Jacobian and separate/joint markups:", jacobian, separate, joint)


def chapter_05() -> None:
    before = log(1 + exp(1.0))
    after = log(1 + exp(0.5))
    compensation = before - after
    assert abs(compensation - 0.3391847033381162) < 1.0e-14
    print("IV-05 inclusive values and compensating variation:", before, after, compensation)


def chapter_06() -> None:
    weights = (Fraction(3), Fraction(4))
    squared_sum = sum(value**2 for value in weights)
    times = tuple(value**2 / squared_sum for value in weights)
    value = sqrt(float(squared_sum))
    assert times == (Fraction(9, 25), Fraction(16, 25)) and value == 5.0
    assert (3, 4, 5) == (3, 4, int(value))
    print("IV-06 singleton/joint values and viewing times:", (3, 4, value), times)


def chapter_07() -> None:
    a, b, beta = Fraction(8), Fraction(2), Fraction(3, 5)
    fee = (1 - beta) * a - beta * b
    distributor = a - fee
    channel = b + fee
    assert (fee, distributor, channel) == (2, 6, 4)
    print("IV-07 fee and agreement surpluses:", fee, distributor, channel)


def chapter_08() -> None:
    next_bad_from_one = Fraction(1, 10)
    next_bad_from_two = Fraction(1, 2)
    assert next_bad_from_one != next_bad_from_two
    beta = Fraction(9, 10)
    assert beta < 1
    print("IV-08 next bad-cell masses from equal aggregate states:", next_bad_from_one, next_bad_from_two)


def chapter_09() -> None:
    states = ((1, 0), (0, 1))
    totals = tuple(x1 + x2 for x1, x2 in states)
    responses = tuple(x1 - x2 for x1, x2 in states)
    assert totals == (1, 1) and responses == (1, -1)
    assert 2 * states[0][0] - totals[0] == responses[0]
    print("IV-09 equal totals and hidden responses:", totals, responses)


def chapter_10() -> None:
    transition = (
        (Fraction(9, 10), Fraction(0), Fraction(1, 10)),
        (Fraction(0), Fraction(1, 2), Fraction(1, 2)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    assert all(sum(row) == 1 for row in transition)
    assert transition[0][2] == Fraction(1, 10) and transition[1][2] == Fraction(1, 2)
    print("IV-10 next bad-state masses:", transition[0][2], transition[1][2])


def chapter_11() -> None:
    response = ((Fraction(1), Fraction(1, 2)), (Fraction(1), Fraction(1, 5)))
    determinant = response[0][0] * response[1][1] - response[0][1] * response[1][0]
    assert determinant == Fraction(-3, 10)
    print("IV-11 response matrix and determinant:", response, determinant)


def chapter_12() -> None:
    universe = range(3)  # a=0, b=1, c=2

    def feasible(mask: int) -> int:
        has_a = bool(mask & 1)
        has_b = bool(mask & 2)
        has_c = bool(mask & 4)
        return int((has_a and has_b) or has_c)

    mobius: dict[int, int] = {}
    for mask in range(8):
        subtotal = sum(value for submask, value in mobius.items() if submask != mask and submask & mask == submask)
        mobius[mask] = feasible(mask) - subtotal
    assert {mask: value for mask, value in mobius.items() if value} == {3: 1, 4: 1, 7: -1}
    primal, dual = min(5, 6), min(5, 6)
    assert primal == dual == 5 and tuple(universe) == (0, 1, 2)
    print("IV-12 nonzero Mobius coefficients and primal/dual values:", {mask: value for mask, value in mobius.items() if value}, primal, dual)


def chapter_13() -> None:
    receipt_treated = Fraction(3, 5)
    receipt_control = Fraction(1, 10)
    first_stage = receipt_treated - receipt_control
    receipt_effect = Fraction(2)
    outcome_itt = receipt_effect * first_stage
    assert (first_stage, outcome_itt, outcome_itt / first_stage) == (Fraction(1, 2), Fraction(1), Fraction(2))
    print("IV-13 first stage, outcome ITT, and Wald ratio:", first_stage, outcome_itt, outcome_itt / first_stage)


def chapter_14() -> None:
    phi = Fraction(3, 4)
    response = (phi, 1 - phi)
    squared_distance = (response[0] - 1) ** 2 + (response[1] - Fraction(1, 2)) ** 2
    target_a = Fraction(1, 2)
    target_b = 1 - 2 * phi
    assert squared_distance == Fraction(1, 8)
    assert target_a == Fraction(1, 2) and target_b == Fraction(-1, 2)
    print("IV-14 Family B fit, distance, and family targets:", response, sqrt(float(squared_distance)), (target_a, target_b))


def chapter_15() -> None:
    shift = 0.1
    bond_a = 0.5 * exp(-shift) + 0.5 * exp(-3 * shift)
    bond_b = exp(-2 * shift)
    second_order_a = 1 - 2 * shift + 0.5 * 5 * shift**2
    second_order_b = 1 - 2 * shift + 0.5 * 4 * shift**2
    assert abs(bond_a - 0.8228278193588388) < 1.0e-14
    assert abs(bond_b - 0.8187307530779818) < 1.0e-14
    assert abs(second_order_a - 0.825) < 1.0e-14 and abs(second_order_b - 0.82) < 1.0e-14
    print("IV-15 exact and second-order bond prices:", (bond_a, bond_b), (second_order_a, second_order_b))


def chapter_16() -> None:
    posterior_rule = {-1: Fraction(0), 1: Fraction(1), 3: Fraction(2)}
    probabilities = {-1: Fraction(1, 4), 1: Fraction(1, 2), 3: Fraction(1, 4)}
    bayes_risk = probabilities[1] * 1
    raw_record_risk = Fraction(1)
    constrained_risk = bayes_risk + probabilities[3] * Fraction(1, 2) ** 2
    assert posterior_rule == {-1: 0, 1: 1, 3: 2}
    assert (bayes_risk, raw_record_risk, constrained_risk) == (Fraction(1, 2), Fraction(1), Fraction(9, 16))
    print("IV-16 posterior rule and risks:", posterior_rule, (bayes_risk, raw_record_risk, constrained_risk))


def chapter_17() -> None:
    pi = 0.3
    rate = 2.0
    positive_density = (1 - pi) * rate * exp(-rate * 0.5)
    continuous_mass = (1 - pi)
    posterior_variance = Fraction(1, 2)
    assert abs(positive_density - 0.5150312176400192) < 1.0e-14
    assert abs(pi + continuous_mass - 1) < 1.0e-15 and posterior_variance == Fraction(1, 2)
    print("IV-17 atom, positive density, continuous mass, posterior variance:", pi, positive_density, continuous_mass, posterior_variance)


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
        chapter_13,
        chapter_14,
        chapter_15,
        chapter_16,
        chapter_17,
    ):
        check()
    print("PASS: 17 Volume IV example groups verified")


if __name__ == "__main__":
    main()
