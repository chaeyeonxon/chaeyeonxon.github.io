#!/usr/bin/env python3
"""Reproduce displayed calculations for Volume II."""

from __future__ import annotations

from fractions import Fraction
from math import atan, log, pi, sin, sqrt


def chapter_01() -> None:
    joint = {
        (1, 1): Fraction(3, 8),
        (1, 0): Fraction(1, 8),
        (-1, 1): Fraction(1, 8),
        (-1, 0): Fraction(3, 8),
    }
    assert sum(joint.values()) == 1
    selected_probability = sum(probability for (z, s), probability in joint.items() if s == 1)
    selected_mean = sum(z * probability for (z, s), probability in joint.items() if s == 1) / selected_probability
    unselected_mean = sum(z * probability for (z, s), probability in joint.items() if s == 0) / (1 - selected_probability)
    selected_variance = 1 - selected_mean**2
    product_mean = sum(z * s * probability for (z, s), probability in joint.items())
    assert (selected_probability, selected_mean, unselected_mean, selected_variance, product_mean) == (
        Fraction(1, 2), Fraction(1, 2), Fraction(-1, 2), Fraction(3, 4), Fraction(1, 4)
    )
    assert selected_probability * selected_mean + (1 - selected_probability) * unselected_mean == 0
    assert selected_variance + Fraction(1, 4) == 1
    print("II-01 selection probability, conditional means, variance, E[ZS]:", selected_probability, selected_mean, unselected_mean, selected_variance, product_mean)


def chapter_02() -> None:
    # For M, the median of three Uniform(-1,1) draws, f(m)=3(1-m^2)/4.
    expected_absolute_median = Fraction(3, 8)
    expected_squared_median = Fraction(1, 5)
    selected_mean = -expected_absolute_median
    selected_variance = expected_squared_median - expected_absolute_median**2
    assert selected_variance == Fraction(19, 320)
    print("II-02 E|median|, selected mean and variance:", expected_absolute_median, selected_mean, selected_variance)


def chapter_03() -> None:
    def outcome(a: int, b: int) -> int:
        return 1 + 2 * a + 3 * b + 4 * a * b

    potential = {(a, b): outcome(a, b) for a in (0, 1) for b in (0, 1)}
    main_effect = sum(Fraction(outcome(1, b) - outcome(0, b), 2) for b in (0, 1))
    interaction = outcome(1, 1) - outcome(1, 0) - outcome(0, 1) + outcome(0, 0)
    ht_expectation = sum(
        Fraction(1, 4) * (2 * a * outcome(a, b) - 2 * (1 - a) * outcome(a, b))
        for a in (0, 1) for b in (0, 1)
    )
    assert potential == {(0, 0): 1, (0, 1): 4, (1, 0): 3, (1, 1): 10}
    assert main_effect == interaction == ht_expectation == 4
    print("II-03 outcomes, main effect, interaction, HT expectation:", potential, main_effect, interaction, ht_expectation)


def chapter_04() -> None:
    lower = (Fraction(3, 5), Fraction(0), Fraction(2, 5))
    upper = (Fraction(0), Fraction(3, 5), Fraction(2, 5))

    def target(probabilities: tuple[Fraction, Fraction, Fraction]) -> Fraction:
        return probabilities[1] + 2 * probabilities[2]

    assert target(lower) == Fraction(4, 5)
    assert target(upper) == Fraction(7, 5)
    projected_lower = 2 * (1 - Fraction(13, 20))
    projected_upper = 2 - Fraction(11, 20)
    assert projected_lower == Fraction(7, 10)
    assert projected_upper == Fraction(29, 20)
    print("II-04 sharp and projected bounds:", (target(lower), target(upper)), (projected_lower, projected_upper))


def chapter_05() -> None:
    for m in (10, 100, 1000):
        n, n_i, n_s = m**4, m, m**3
        bias_ratio = Fraction(n, n_s**2)
        centered_ratio = Fraction(n, n_i * n_s)
        assert bias_ratio == Fraction(1, m**2)
        assert centered_ratio == 1
    print("II-05 ratios at m=10,100,1000:", [(Fraction(1, m**2), Fraction(1)) for m in (10, 100, 1000)])


def chapter_06() -> None:
    sample_mean, nuisance, theta_zero = Fraction(21, 20), Fraction(101, 50), Fraction(3)
    theta_hat = sample_mean + nuisance
    sampling_component = sample_mean - 1
    nuisance_component = nuisance - 2
    assert theta_hat == Fraction(307, 100)
    assert theta_hat - theta_zero == sampling_component + nuisance_component == Fraction(7, 100)
    assert theta_hat + Fraction(1, 100) == Fraction(77, 25)
    assert theta_hat - Fraction(1, 100) == Fraction(153, 50)
    print("II-06 estimate and error components:", theta_hat, sampling_component, nuisance_component)


def chapter_07() -> None:
    n_i, n_s = 10, 100
    independent_variance = Fraction(1, n_i * n_s)
    shared_variance = Fraction(1, n_s)
    assert independent_variance == Fraction(1, 1000)
    assert shared_variance == Fraction(1, 100)
    assert shared_variance / independent_variance == n_i
    print("II-07 independent/shared variances:", independent_variance, shared_variance)


def chapter_08() -> None:
    score_distribution = {Fraction(0): Fraction(3, 4), Fraction(2): Fraction(1, 4)}
    mean = sum(value * probability for value, probability in score_distribution.items())
    second_moment = sum(value**2 * probability for value, probability in score_distribution.items())
    variance = second_moment - mean**2
    assert mean == Fraction(1, 2) and variance == Fraction(3, 4)
    one_draw_mean = Fraction(1)
    three_draw_mean = 2 * (Fraction(1, 3) * Fraction(3, 8) + Fraction(1) * Fraction(1, 8))
    assert three_draw_mean == Fraction(1, 2)
    asymptotic_scaled_mean = sqrt(2 / pi)
    assert 0.797 < asymptotic_scaled_mean < 0.799
    print("II-08 two-draw mean/variance, one/three-draw means, asymptotic constant:", mean, variance, one_draw_mean, three_draw_mean, asymptotic_scaled_mean)


def chapter_09() -> None:
    derivative = (Fraction(1), Fraction(2))
    error = (Fraction(1), Fraction(-1))

    def outer(left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]:
        return tuple(tuple(a * b for b in right) for a in left)  # type: ignore[return-value]

    true_gram = outer(derivative, derivative)
    error_gram = outer(error, error)
    naive = tuple(tuple(true_gram[i][j] + error_gram[i][j] for j in range(2)) for i in range(2))
    assert true_gram == ((1, 2), (2, 4))
    assert naive == ((2, 1), (1, 5))
    eta = Fraction(1, 10)
    squared_distance = eta**4 / (1 + eta**4)
    assert squared_distance == Fraction(1, 10001)
    print("II-09 true/naive Gram and squared range distance at eta=.1:", true_gram, naive, squared_distance)


def chapter_10() -> None:
    epsilon = 0.1
    high = 2 + sqrt(1 + epsilon**2)
    low = 2 - sqrt(1 + epsilon**2)
    angle = 0.5 * atan(epsilon)
    sine_angle = sin(angle)
    assert abs(high - 3.004987562112089) < 1.0e-14
    assert abs(low - 0.995012437887911) < 1.0e-14
    assert sine_angle <= epsilon / 2 + 1.0e-15
    print("II-10 eigenvalues, angle, sine and bound:", high, low, angle, sine_angle, epsilon / 2)


def chapter_11() -> None:
    coefficient, standard_error, scale = Fraction(2, 5), Fraction(1, 10), Fraction(100)
    scaled_coefficient = coefficient / scale
    scaled_standard_error = standard_error / scale
    assert scaled_coefficient == Fraction(1, 250)
    assert scaled_standard_error == Fraction(1, 1000)
    assert coefficient / standard_error == scaled_coefficient / scaled_standard_error == 4
    print("II-11 original/scaled coefficient, standard error, t-statistic:", coefficient, standard_error, scaled_coefficient, scaled_standard_error, coefficient / standard_error)


def chapter_12() -> None:
    directions = ((1, -2), (-2, 1), (1, 1), (0, 3))
    derivatives = tuple(min(direction) for direction in directions)
    assert derivatives == (-2, -2, 1, 0)
    print("II-12 directional derivatives:", tuple(zip(directions, derivatives)))


def chapter_13() -> None:
    candidate_count, sample_size, delta = 10, 1000, 0.05
    epsilon = sqrt(log(2 * candidate_count / delta) / (2 * sample_size))
    excess = 2 * epsilon
    assert abs(epsilon - 0.054733283051119734) < 1.0e-14
    assert abs(excess - 0.10946656610223947) < 1.0e-14
    print("II-13 uniform and excess-risk bounds:", epsilon, excess)


def main() -> None:
    for check in (chapter_01, chapter_02, chapter_03, chapter_04, chapter_05, chapter_06, chapter_07, chapter_08, chapter_09, chapter_10, chapter_11, chapter_12, chapter_13):
        check()
    print("PASS: 13 Volume II example groups verified")


if __name__ == "__main__":
    main()
