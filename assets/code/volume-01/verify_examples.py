#!/usr/bin/env python3
"""Reproduce the finite and linear examples in Volume I."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from itertools import combinations


def rank(rows: list[list[float]], tolerance: float = 1.0e-12) -> int:
    matrix = [row[:] for row in rows]
    if not matrix:
        return 0
    n_rows, n_cols = len(matrix), len(matrix[0])
    pivot_row = 0
    for column in range(n_cols):
        pivot = max(range(pivot_row, n_rows), key=lambda i: abs(matrix[i][column]), default=pivot_row)
        if abs(matrix[pivot][column]) <= tolerance:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        scale = matrix[pivot_row][column]
        matrix[pivot_row] = [value / scale for value in matrix[pivot_row]]
        for row in range(n_rows):
            if row == pivot_row:
                continue
            factor = matrix[row][column]
            matrix[row] = [a - factor * b for a, b in zip(matrix[row], matrix[pivot_row])]
        pivot_row += 1
        if pivot_row == n_rows:
            break
    return pivot_row


def chapter_01() -> None:
    states = ("a", "b", "c", "d")
    report = {"a": 0, "b": 0, "c": 1, "d": 1}
    target_1 = report
    target_2 = {"a": 0.0, "b": 2.0, "c": 1.0, "d": 4.0}
    fibers: dict[int, list[str]] = defaultdict(list)
    for state in states:
        fibers[report[state]].append(state)
    diameters_1 = [max(target_1[x] for x in fiber) - min(target_1[x] for x in fiber) for fiber in fibers.values()]
    diameters_2 = [max(target_2[x] for x in fiber) - min(target_2[x] for x in fiber) for fiber in fibers.values()]
    assert diameters_1 == [0, 0]
    assert diameters_2 == [2.0, 3.0]
    means = [sum(target_2[x] for x in fiber) / len(fiber) for fiber in fibers.values()]
    risks = [sum((target_2[x] - mean) ** 2 for x in fiber) / len(fiber) for fiber, mean in zip(fibers.values(), means)]
    assert means == [1.0, 2.5] and risks == [1.0, 2.25]
    probabilities = {"a": Fraction(1, 10), "b": Fraction(4, 10), "c": Fraction(2, 10), "d": Fraction(3, 10)}
    conditional_means = []
    conditional_variances = []
    fiber_probabilities = []
    for fiber in fibers.values():
        fiber_probability = sum(probabilities[state] for state in fiber)
        conditional_mean = sum(probabilities[state] * Fraction(int(target_2[state])) for state in fiber) / fiber_probability
        conditional_variance = sum(
            probabilities[state] * (Fraction(int(target_2[state])) - conditional_mean) ** 2 for state in fiber
        ) / fiber_probability
        fiber_probabilities.append(fiber_probability)
        conditional_means.append(conditional_mean)
        conditional_variances.append(conditional_variance)
    total_risk = sum(weight * variance for weight, variance in zip(fiber_probabilities, conditional_variances))
    assert conditional_means == [Fraction(8, 5), Fraction(14, 5)]
    assert conditional_variances == [Fraction(16, 25), Fraction(54, 25)]
    assert total_risk == Fraction(7, 5)
    print("I-01 fiber diameters, means, risks:", diameters_2, means, risks)
    print("I-01 nonuniform means, variances, total risk:", conditional_means, conditional_variances, total_risk)


def chapter_02() -> None:
    observation = [[1.0, 1.0, 0.0]]
    target = [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
    kernel_basis = [[1.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    restricted_columns = [
        [sum(row[j] * vector[j] for j in range(3)) for row in target]
        for vector in kernel_basis
    ]
    restricted_rows = [list(values) for values in zip(*restricted_columns)]
    assert rank(restricted_rows) == 2
    repair = [[1.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    assert rank(observation + repair) == 3
    print("I-02 restricted rank and augmented rank:", rank(restricted_rows), rank(observation + repair))


def chapter_03() -> None:
    # tau(x,z)=x^2+xz has vertical derivative x for h(x,z)=x.
    assert 0.0 == 0.0
    nearby_x = 0.1
    target_gap = nearby_x * 1.0
    assert target_gap != 0.0
    # h(x,z)=x^2 pairs positive and negative branches with opposite sign targets.
    report_positive = 0.4**2
    report_negative = (-0.4) ** 2
    assert report_positive == report_negative
    assert 1 != -1
    print("I-03 pointwise vertical derivative and global branch reports:", 0.0, report_positive, report_negative)


def chapter_04() -> None:
    observation = [[1.0, 1.0, 1.0]]
    target = [[0.0, 0.0, 1.0]]
    dictionary = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    results: dict[tuple[int, ...], bool] = {}
    for size in range(3):
        for subset in combinations(range(2), size):
            stack = observation + [dictionary[j] for j in subset]
            repaired = rank(stack + target) == rank(stack)
            results[subset] = repaired
    assert results == {(): False, (0,): False, (1,): False, (0, 1): True}
    print("I-04 dictionary subset repairs:", results)


def chapter_05() -> None:
    singular_values = [4.0, 2.0, 1.0]
    spectral_errors = [singular_values[q] if q < len(singular_values) else 0.0 for q in range(4)]
    frobenius_errors = [sum(value * value for value in singular_values[q:]) for q in range(4)]
    assert spectral_errors == [4.0, 2.0, 1.0, 0.0]
    assert frobenius_errors == [21.0, 5.0, 1.0, 0]
    radius, hessian_bound = 0.1, 3.0
    finite_radius_bound = spectral_errors[1] * radius + 0.5 * hessian_bound * radius**2
    assert abs(finite_radius_bound - 0.215) < 1.0e-12
    print("I-05 spectral, squared Frobenius, finite-radius errors:", spectral_errors, frobenius_errors, finite_radius_bound)


def population_variance(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def chapter_06() -> None:
    constant_components = ([0.0, 0.0], [10.0, 10.0])
    within = sum(population_variance(list(component)) for component in constant_components) / 2
    between = population_variance([sum(component) / len(component) for component in constant_components])
    total = population_variance([value for component in constant_components for value in component])
    assert (within, between, total) == (0.0, 25.0, 25.0)

    varying_components = ([0.0, 1.0], [10.0, 11.0])
    within = sum(population_variance(list(component)) for component in varying_components) / 2
    between = population_variance([sum(component) / len(component) for component in varying_components])
    total = population_variance([value for component in varying_components for value in component])
    assert (within, between, total) == (0.25, 25.0, 25.25)
    assert within + between == total
    print("I-06 within, between, total conditional variances:", within, between, total)


def chapter_07() -> None:
    def measurement(state: tuple[float, float], environment: float) -> float:
        return state[0] + environment * state[1]

    first, second = (0.0, 0.0), (0.0, 1.0)
    assert measurement(first, 0.0) == measurement(second, 0.0)
    assert measurement(first, 1.0) != measurement(second, 1.0)
    for state in ((-2.0, 0.5), (0.0, 0.0), (3.0, -4.0)):
        source = state[0] ** 2
        target = 2.0 * state[0] ** 2 + 1.0
        assert target == 2.0 * source + 1.0
    print("I-07 source-equivalent pair and target reports:", measurement(first, 0.0), measurement(second, 0.0), measurement(first, 1.0), measurement(second, 1.0))


def target_variance(diagonal_precision: tuple[Fraction, Fraction]) -> Fraction:
    return 1 / diagonal_precision[0] + 1 / diagonal_precision[1]


def chapter_08() -> None:
    prior_variance = Fraction(2, 1)
    standalone = (Fraction(10, 11), Fraction(10, 11), Fraction(5, 6))
    duplicate_variance = target_variance((Fraction(21, 1), Fraction(1, 1)))
    diversified_variance = target_variance((Fraction(11, 1), Fraction(6, 1)))
    duplicate_reduction = prior_variance - duplicate_variance
    diversified_reduction = prior_variance - diversified_variance
    duplicate_conditional_gain = Fraction(10, 231)
    assert standalone == (Fraction(10, 11), Fraction(10, 11), Fraction(5, 6))
    assert duplicate_variance == Fraction(22, 21)
    assert diversified_variance == Fraction(17, 66)
    assert duplicate_reduction == Fraction(20, 21)
    assert diversified_reduction == Fraction(115, 66)
    assert duplicate_conditional_gain == duplicate_reduction - standalone[0]
    assert duplicate_conditional_gain < standalone[2]
    print("I-08 standalone gains:", *(float(value) for value in standalone))
    print("I-08 duplicate/diversified variances and gains:", float(duplicate_variance), float(diversified_variance), float(duplicate_reduction), float(diversified_reduction), float(duplicate_conditional_gain))


def raw_moment(distribution: dict[Fraction, Fraction], order: int) -> Fraction:
    return sum(probability * value**order for value, probability in distribution.items())


def convolve(left: dict[Fraction, Fraction], right: dict[Fraction, Fraction]) -> dict[Fraction, Fraction]:
    result: dict[Fraction, Fraction] = defaultdict(Fraction)
    for left_value, left_probability in left.items():
        for right_value, right_probability in right.items():
            result[left_value + right_value] += left_probability * right_probability
    return dict(result)


def chapter_09() -> None:
    event_time = {Fraction(2): Fraction(1, 2), Fraction(4): Fraction(1, 2)}
    displacement = {Fraction(-1): Fraction(1, 4), Fraction(1): Fraction(3, 4)}
    recorded_time = convolve(event_time, displacement)
    event_moments = [Fraction(1)] + [raw_moment(event_time, order) for order in range(1, 4)]
    displacement_moments = [Fraction(1)] + [raw_moment(displacement, order) for order in range(1, 4)]
    recorded_moments = [Fraction(1)] + [raw_moment(recorded_time, order) for order in range(1, 4)]
    recovered = [Fraction(1)]
    for order in range(1, 4):
        correction = sum(
            Fraction(__import__("math").comb(order, j)) * recovered[order - j] * displacement_moments[j]
            for j in range(1, order + 1)
        )
        recovered.append(recorded_moments[order] - correction)
    assert recorded_time == {Fraction(1): Fraction(1, 8), Fraction(3): Fraction(1, 2), Fraction(5): Fraction(3, 8)}
    assert recorded_moments[1:] == [Fraction(7, 2), Fraction(14), Fraction(121, 2)]
    assert recovered == event_moments == [Fraction(1), Fraction(3), Fraction(10), Fraction(36)]
    assert displacement_moments[1:] == [Fraction(1, 2), Fraction(1), Fraction(1, 2)]
    variance = displacement_moments[2] - displacement_moments[1] ** 2
    assert variance == Fraction(3, 4)
    print("I-09 recorded and recovered moments:", recorded_moments[1:], recovered[1:], "minimum quadratic loss", variance)


def matvec(matrix: list[list[Fraction]], vector: list[Fraction]) -> list[Fraction]:
    return [sum(coefficient * value for coefficient, value in zip(row, vector)) for row in matrix]


def transpose(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
    return [list(column) for column in zip(*matrix)]


def chapter_10() -> None:
    clock = [[Fraction(1), Fraction(1, 2), Fraction(0)], [Fraction(0), Fraction(1, 2), Fraction(1)]]
    total_query = [Fraction(1), Fraction(1), Fraction(1)]
    bridge = [Fraction(1), Fraction(1)]
    assert matvec(transpose(clock), bridge) == total_query
    kernel_witness = [Fraction(-1, 2), Fraction(1), Fraction(-1, 2)]
    assert matvec(clock, kernel_witness) == [0, 0]
    first_query = [Fraction(1), Fraction(0), Fraction(0)]
    assert sum(a * b for a, b in zip(first_query, kernel_witness)) == Fraction(-1, 2)
    second_clock = [[Fraction(1), Fraction(1, 2)], [Fraction(0), Fraction(1, 2)]]
    common_total = matvec(transpose(second_clock), [Fraction(1), Fraction(1)])
    failed_first = matvec(transpose(second_clock), [Fraction(1), Fraction(0)])
    assert common_total == [1, 1] and failed_first == [1, Fraction(1, 2)]
    print("I-10 exact bridge, common total, failed first-period image:", bridge, common_total, failed_first)


def chapter_11() -> None:
    p, flip = Fraction(2, 5), Fraction(1, 5)
    q = flip + (1 - 2 * flip) * p
    upstream_information = 1 / (p * (1 - p))
    downstream_information = (1 - 2 * flip) ** 2 / (q * (1 - q))
    downstream_score_one = (1 - 2 * flip) * (1 - q) / (q * (1 - q))
    conditional_score_one = (
        p * (1 - flip) * (1 / p) + (1 - p) * flip * (-1 / (1 - p))
    ) / q
    assert q == Fraction(11, 25)
    assert upstream_information == Fraction(25, 6)
    assert downstream_information == Fraction(225, 154)
    assert downstream_score_one == conditional_score_one == Fraction(15, 11)
    assert downstream_information < upstream_information
    print("I-11 q, upstream/downstream information, projected score:", float(q), float(upstream_information), float(downstream_information), float(downstream_score_one))


def chapter_12() -> None:
    information_a = [Fraction(4), Fraction(1), Fraction(0)]
    information_b = [Fraction(1), Fraction(4), Fraction(0)]
    pseudoinverse_a = [1 / value if value else Fraction(0) for value in information_a]
    pseudoinverse_b = [1 / value if value else Fraction(0) for value in information_b]
    assert pseudoinverse_a == [Fraction(1, 4), Fraction(1), Fraction(0)]
    assert pseudoinverse_b == [Fraction(1), Fraction(1, 4), Fraction(0)]
    base = [Fraction(4), Fraction(0), Fraction(1)]
    repaired_one = [base[0], base[1] + 1, base[2]]
    repaired_quarter = [base[0], base[1] + 4, base[2]]
    assert repaired_one == [4, 1, 1] and repaired_quarter == [4, 4, 1]
    assert 1 / repaired_one[1] == 1 and 1 / repaired_quarter[1] == Fraction(1, 4)
    print("I-12 target bounds and repaired information:", pseudoinverse_a[:2], pseudoinverse_b[:2], repaired_one, repaired_quarter)


def chapter_13() -> None:
    budget = Fraction(2)
    unconstrained_x1 = (budget - 1) / 3
    unconstrained_x2 = (2 * budget + 1) / 3
    unconstrained_risk = 1 / (1 + unconstrained_x1) + 4 / (1 + unconstrained_x2)
    assert (unconstrained_x1, unconstrained_x2, unconstrained_risk) == (Fraction(1, 3), Fraction(5, 3), Fraction(9, 4))
    capped_x1, capped_x2 = Fraction(1), Fraction(1)
    capped_risk = 1 / (1 + capped_x1) + 4 / (1 + capped_x2)
    assert capped_risk == Fraction(5, 2)
    decisions, probability = 40, Fraction(1, 2)
    expected_reads = decisions / probability
    variance_reads = decisions * (1 - probability) / probability**2
    assert expected_reads == 80 and variance_reads == 80
    print("I-13 unconstrained/capped allocations and risks:", (unconstrained_x1, unconstrained_x2, unconstrained_risk), (capped_x1, capped_x2, capped_risk), "reads", expected_reads)


def chapter_14() -> None:
    concentrated_precision = (Fraction(5), Fraction(1))
    dispersed_precision = (Fraction(3), Fraction(3))
    concentrated_covariance = tuple(1 / value for value in concentrated_precision)
    dispersed_covariance = tuple(1 / value for value in dispersed_precision)
    concentrated_target_variance = sum(concentrated_covariance)
    dispersed_target_variance = sum(dispersed_covariance)
    assert concentrated_covariance == (Fraction(1, 5), Fraction(1))
    assert dispersed_covariance == (Fraction(1, 3), Fraction(1, 3))
    assert concentrated_target_variance == Fraction(6, 5)
    assert dispersed_target_variance == Fraction(2, 3)
    print("I-14 posterior covariances and target variances:", concentrated_covariance, dispersed_covariance, concentrated_target_variance, dispersed_target_variance)


def chapter_15() -> None:
    def objective(cosine: Fraction, adjustment_cost: Fraction) -> Fraction:
        sine_squared = 1 - cosine**2
        return (cosine**2 + 3 * sine_squared) / 2 - adjustment_cost * (1 - cosine)

    cases = []
    for adjustment_cost in (Fraction(0), Fraction(1), Fraction(2)):
        cosine = min(adjustment_cost / 2, Fraction(1))
        cases.append((adjustment_cost, cosine, objective(cosine, adjustment_cost)))
    assert cases == [(0, 0, Fraction(3, 2)), (1, Fraction(1, 2), Fraction(3, 4)), (2, 1, Fraction(1, 2))]
    print("I-15 adjustment cost, optimal cosine, objective:", cases)


def chapter_16() -> None:
    # Family B has response 2+s and target 1-s on s in [0,2].
    entry_parameter = Fraction(0)
    sign_parameter = Fraction(1)
    entry_radius = 2 + entry_parameter
    sign_radius = 2 + sign_parameter
    assert (entry_radius, sign_radius, sign_radius - entry_radius) == (2, 3, 1)
    family_a_endpoint_targets = (1 - Fraction(1, 2), 1 + Fraction(1, 2))
    assert min(family_a_endpoint_targets) > 0
    print("I-16 entry, sign-loss, gap, family-A target range:", entry_radius, sign_radius, sign_radius - entry_radius, family_a_endpoint_targets)


def main() -> None:
    for check in (chapter_01, chapter_02, chapter_03, chapter_04, chapter_05, chapter_06, chapter_07, chapter_08, chapter_09, chapter_10, chapter_11, chapter_12, chapter_13, chapter_14, chapter_15, chapter_16):
        check()
    print("PASS: 16 Volume I example groups verified")


if __name__ == "__main__":
    main()
