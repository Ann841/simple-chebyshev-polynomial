"""Chebyshev polynomials of the first and second kind.

Both kinds satisfy a three-term recurrence; only the initial conditions
differ.  We compute via the recurrence (Clenshaw-style forward iteration)
rather than a closed form because the recurrence is numerically stable
across the entire real line and avoids computing transcendental
functions inside a hot loop.
"""

from __future__ import annotations

__all__ = ["chebyshev_t", "chebyshev_u"]


def chebyshev_t(n: int, x: float) -> float:
    """Return the value of the Chebyshev polynomial of the first kind, T_n(x).

    The recurrence is

        T_0(x) = 1,   T_1(x) = x,   T_{k+1}(x) = 2 x T_k(x) - T_{k-1}(x).

    Using the recurrence instead of the cos(n arccos x) closed form is
    deliberate: the closed form is undefined for |x| > 1 (arccos domain
    error) and loses precision near x = +/-1 due to cancellation in the
    arccos/cos pair.  The recurrence stays well-behaved for all real x.

    Parameters
    ----------
    n:
        Non-negative degree.  Negative ``n`` raises ``ValueError``.
    x:
        Real number at which to evaluate the polynomial.

    Returns
    -------
    float
        Value of T_n at x.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 1.0
    if n == 1:
        return float(x)

    t_prev = 1.0          # T_0
    t_curr = float(x)     # T_1
    two_x = 2.0 * float(x)
    for _ in range(2, n + 1):
        t_prev, t_curr = t_curr, two_x * t_curr - t_prev
    return t_curr


def chebyshev_u(n: int, x: float) -> float:
    """Return the value of the Chebyshev polynomial of the second kind, U_n(x).

    The recurrence is

        U_0(x) = 1,   U_1(x) = 2x,   U_{k+1}(x) = 2 x U_k(x) - U_{k-1}(x).

    The same stability argument as for ``chebyshev_t`` applies: the
    recurrence is valid for all real x, whereas the sin((n+1) arccos x)
    closed form breaks down for |x| > 1.

    Parameters
    ----------
    n:
        Non-negative degree.  Negative ``n`` raises ``ValueError``.
    x:
        Real number at which to evaluate the polynomial.

    Returns
    -------
    float
        Value of U_n at x.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 1.0
    if n == 1:
        return 2.0 * float(x)

    u_prev = 1.0          # U_0
    u_curr = 2.0 * float(x)  # U_1
    two_x = 2.0 * float(x)
    for _ in range(2, n + 1):
        u_prev, u_curr = u_curr, two_x * u_curr - u_prev
    return u_curr
