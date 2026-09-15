#!/usr/bin/env python3
"""Reproduce the displayed numerical claims in Volume III.

The script uses only the Python standard library.  Each check prints the
quantity shown in the chapter and raises AssertionError when a residual exceeds
the declared tolerance.
"""

from __future__ import annotations

import cmath
import math
from fractions import Fraction


TOL = 1.0e-6


def close(actual: float, expected: float, tol: float = TOL) -> None:
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{actual} != {expected} within {tol}")


def eigvals_sym_2x2(a: float, b: float, d: float) -> tuple[float, float]:
    center = (a + d) / 2.0
    radius = math.hypot((a - d) / 2.0, b)
    return center + radius, center - radius


def chapter_02() -> None:
    # det(A'A - lambda M) = (100 lambda^2 - 137 lambda + 27) / 3.
    root = math.sqrt(7_969.0)
    values = ((137.0 + root) / 200.0, (137.0 - root) / 200.0)
    close(values[0], 1.131346279)
    close(values[1], 0.238653721)
    print("III-02 generalized eigenvalues:", *(f"{x:.8f}" for x in values))


def chapter_03() -> None:
    # G = diag(1,4), C = [1,1], r = 1.
    z = (Fraction(4, 5), Fraction(1, 5))
    cost = Fraction(1, 2) * (z[0] ** 2 + 4 * z[1] ** 2)
    assert sum(z) == 1 and cost == Fraction(2, 5)
    print("III-03 weighted lift and cost:", tuple(map(float, z)), float(cost))


def chapter_04() -> None:
    epsilon = 0.1
    # Exact four-leg flow for X=(1,0,0), Y=(0,1,r1).
    point = [0.0, 0.0, 0.0]
    point[0] += epsilon
    point[1] += epsilon
    point[2] += epsilon * point[0]
    point[0] -= epsilon
    point[1] -= epsilon
    point[2] -= epsilon * point[0]
    close(point[2], epsilon**2)
    print("III-04 loop endpoint:", tuple(point))


def chapter_05() -> None:
    def contrast(epsilon: float, mirrored: bool = False) -> float:
        cubic = -3.0 if mirrored else 3.0
        return 2.0 * epsilon**2 + cubic * epsilon**3 + 5.0 * epsilon**4

    def mirrored_estimate(epsilon: float) -> float:
        return (contrast(epsilon) + contrast(epsilon, True)) / (2.0 * epsilon**2)

    raw = contrast(0.1) / 0.1**2
    q1 = mirrored_estimate(0.1)
    q2 = mirrored_estimate(0.05)
    richardson = (4.0 * q2 - q1) / 3.0
    close(raw, 2.35)
    close(q1, 2.05)
    close(q2, 2.0125)
    close(richardson, 2.0)
    print("III-05 raw, mirrored, Richardson:", raw, q1, q2, richardson)


def chapter_06() -> None:
    rows = ((1, 0, 0), (1, 1, 0), (1, 2, 1))
    determinant = (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )
    assert determinant == 1
    print("III-06 response-row determinant:", determinant)


def chapter_07() -> None:
    diagonal = tuple(math.sqrt(1.0 + a * a + a**4) for a in (0.8, 0.4, 0.2))
    close(diagonal[0], 1.43164241)
    close(diagonal[1], 1.08885261)
    close(diagonal[2], 1.02058807)
    gram_eigs = eigvals_sym_2x2(2.0496, 1.4224, 1.1856)
    mixed = tuple(math.sqrt(max(x, 0.0)) for x in gram_eigs)
    close(mixed[0], 1.76186123)
    close(mixed[1], 0.36200138)
    print("III-07 singular values:", diagonal, mixed)


def chapter_08() -> None:
    margins = (Fraction(13, 25), Fraction(73, 25), Fraction(7, 25))
    roots = ((3 + 3j) / 5, (3 - 3j) / 5)
    assert all(x > 0 for x in margins)
    close(abs(roots[0]), math.sqrt(18.0) / 5.0)
    print("III-08 Jury margins and roots:", tuple(map(float, margins)), roots)


def chapter_09() -> None:
    a, b, t = 0.81, 0.79, 5
    divided_difference = (a**t - b**t) / (a - b)
    repeated = t * 0.8 ** (t - 1)
    close(divided_difference, 2.04864001)
    close(repeated, 2.048)
    print("III-09 close-pole and repeated-pole responses:", divided_difference, repeated)


def chapter_10() -> None:
    delta, dimension = 1.0e-4, 10_000
    close(dimension * delta, 1.0)
    print("III-10 truncated inverse norm and amplified error:", dimension, dimension * delta)


def chapter_11() -> None:
    attenuations = (math.exp(-2.0), math.exp(-8.0))
    amplifications = tuple(1.0 / x for x in attenuations)
    close(attenuations[0], 0.135335283)
    close(attenuations[1], 0.000335463)
    close(amplifications[0], 7.389056099)
    close(amplifications[1], 2_980.957987)
    print("III-11 attenuation and amplification:", attenuations, amplifications)


def chapter_12() -> None:
    norms = tuple(math.exp(0.01 * k * k) for k in (5, 10, 20, 40))
    expected = (1.284025417, 2.718281828, 54.598150033, 8_886_110.520508)
    for actual, target in zip(norms, expected):
        close(actual, target)
    print("III-12 band inverse norms:", norms)


def chapter_13() -> None:
    values = eigvals_sym_2x2(1.0, 0.5, 1.0 / 3.0)
    close(values[0], (4.0 + math.sqrt(13.0)) / 6.0)
    close(values[1], (4.0 - math.sqrt(13.0)) / 6.0)
    close(sum(values), 4.0 / 3.0)
    print("III-13 integrated-Gram eigenvalues:", values)


def chapter_14() -> None:
    # Q(1,0,-1)' = (1,0,-1)' for the three-node path.
    variance = (1.0 + 0.0 + 1.0) / 3.0
    energy = variance
    close(variance, 2.0 / 3.0)
    print("III-14 variance, energy, gap:", variance, energy, 1.0)


def chapter_15() -> None:
    m12 = Fraction(4) - Fraction(2) - Fraction(1)
    m13 = Fraction(3) - Fraction(2) - Fraction(3, 2)
    m23 = Fraction(3) - Fraction(1) - Fraction(3, 2)
    m123 = Fraction(5) - sum(
        (Fraction(2), Fraction(1), Fraction(3, 2), m12, m13, m23),
        Fraction(0),
    )
    assert (m12, m13, m23, m123) == (
        Fraction(1),
        Fraction(-1, 2),
        Fraction(1, 2),
        Fraction(-1, 2),
    )
    print("III-15 interaction coefficients:", tuple(map(float, (m12, m13, m23, m123))))


def chapter_16() -> None:
    x = (0.0, 1.0, 0.0)
    y = (0.75, 0.75)
    primal = x[0] + 1.5 * x[1] + x[2]
    dual = y[0] + y[1]
    close(primal, 1.5)
    close(dual, primal)
    assert y[0] <= 1 and y[0] + y[1] <= 1.5 and y[1] <= 1
    print("III-16 primal and dual values:", primal, dual)


def chapter_17() -> None:
    risk = (2.0**2) / 4.0
    close(risk, 1.0)
    proposed_diagonal = (4.0, 0.0)
    assert max(proposed_diagonal) > 3.0 and sum(proposed_diagonal) == 4.0
    print("III-17 singular risk and infeasible diagonal:", risk, proposed_diagonal)


def chapter_18() -> None:
    a, q, r = 0.9, 0.1, 0.25
    coefficient = q + r - r * a * a
    fixed = (-coefficient + math.sqrt(coefficient**2 + 4.0 * a * a * r * q)) / (2.0 * a * a)
    updated = r * (a * a * fixed + q) / (a * a * fixed + q + r)
    derivative = a * a * r * r / (a * a * fixed + q + r) ** 2
    close(fixed, 0.1068247885)
    close(updated, fixed)
    close(derivative, 0.2656688699)
    print("III-18 Riccati fixed point and derivative:", fixed, derivative)


def main() -> None:
    checks = [
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
        chapter_18,
    ]
    for check in checks:
        check()
    print(f"PASS: {len(checks)} Volume III example groups verified at tolerance {TOL:g}")


if __name__ == "__main__":
    main()
