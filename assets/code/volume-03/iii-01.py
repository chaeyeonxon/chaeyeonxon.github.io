"""Reproduce the numerical calculations in Chapter III-01.

The script uses only Python's standard library. It solves

    q + w**2 = p
    w + q**2 = s

with Newton's method and compares the analytic policy derivative with
central finite differences at a regular and a nearly singular equilibrium.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, sqrt


@dataclass(frozen=True)
class Equilibrium:
    q: float
    w: float

    @property
    def p(self) -> float:
        return self.q + self.w**2

    @property
    def s(self) -> float:
        return self.w + self.q**2


def residual(q: float, w: float, p: float, s: float) -> tuple[float, float]:
    return q + w * w - p, w + q * q - s


def jacobian(q: float, w: float) -> tuple[tuple[float, float], tuple[float, float]]:
    return ((1.0, 2.0 * w), (2.0 * q, 1.0))


def determinant(q: float, w: float) -> float:
    return 1.0 - 4.0 * q * w


def solve_equilibrium(
    p: float,
    s: float,
    initial: Equilibrium,
    tolerance: float = 1e-14,
    max_iterations: int = 80,
) -> Equilibrium:
    q, w = initial.q, initial.w
    for _ in range(max_iterations):
        f1, f2 = residual(q, w, p, s)
        det = determinant(q, w)
        if abs(det) <= 1e-14:
            raise ArithmeticError("The Newton Jacobian is numerically singular.")

        # Solve J * step = -F for the 2-by-2 Jacobian.
        dq = (-f1 + 2.0 * w * f2) / det
        dw = (2.0 * q * f1 - f2) / det
        q += dq
        w += dw

        if max(abs(dq), abs(dw)) <= tolerance:
            f1, f2 = residual(q, w, p, s)
            if max(abs(f1), abs(f2)) <= 10.0 * tolerance:
                return Equilibrium(q, w)

    raise ArithmeticError("Newton's method did not reach the residual tolerance.")


def policy_derivative(point: Equilibrium) -> tuple[float, float]:
    det = determinant(point.q, point.w)
    return 1.0 / det, -2.0 * point.q / det


def condition_number_2(point: Equilibrium) -> float:
    """Return the spectral condition number of the 2-by-2 Jacobian."""
    (a, b), (c, d) = jacobian(point.q, point.w)
    trace = a * a + b * b + c * c + d * d
    determinant_squared = (a * d - b * c) ** 2
    discriminant = max(0.0, trace * trace - 4.0 * determinant_squared)
    largest = (trace + sqrt(discriminant)) / 2.0
    smallest = (trace - sqrt(discriminant)) / 2.0
    return sqrt(largest / smallest)


def central_difference(point: Equilibrium, step: float) -> tuple[float, float]:
    plus = solve_equilibrium(point.p + step, point.s, point)
    minus = solve_equilibrium(point.p - step, point.s, point)
    return (
        (plus.q - minus.q) / (2.0 * step),
        (plus.w - minus.w) / (2.0 * step),
    )


def report(point: Equilibrium) -> None:
    exact_q, exact_w = policy_derivative(point)
    print(
        f"reference=({point.q:.2f}, {point.w:.2f})  "
        f"p={point.p:.10f}  s={point.s:.10f}  "
        f"det(J)={determinant(point.q, point.w):.10f}  "
        f"kappa_2(J)={condition_number_2(point):.10f}"
    )
    print(f"analytic derivative=({exact_q:.10f}, {exact_w:.10f})")
    print("h          FD q_p          FD w_p          Euclidean error")
    for step in (1e-2, 1e-4, 1e-6, 1e-8):
        fd_q, fd_w = central_difference(point, step)
        error = hypot(fd_q - exact_q, fd_w - exact_w)
        print(f"{step:1.0e}  {fd_q: .10f}  {fd_w: .10f}  {error:.3e}")
    print()


if __name__ == "__main__":
    report(Equilibrium(0.20, 0.30))
    report(Equilibrium(0.49, 0.49))
