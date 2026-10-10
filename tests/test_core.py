import math
import unittest

from simple_chebyshev_polynomial import chebyshev_t, chebyshev_u


def _approx_equal(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


class TestChebyshevT(unittest.TestCase):
    """Tests for the first-kind polynomial T_n."""

    def test_t0_is_one(self):
        for x in (-2.0, -1.0, 0.0, 0.5, 1.0, 3.14):
            with self.subTest(x=x):
                self.assertEqual(chebyshev_t(0, x), 1.0)

    def test_t1_is_x(self):
        for x in (-2.0, -1.0, 0.0, 0.5, 1.0, 3.14):
            with self.subTest(x=x):
                self.assertEqual(chebyshev_t(1, x), x)

    def test_t2_recurrence(self):
        # T_2(x) = 2x^2 - 1
        for x in (-1.0, 0.0, 0.5, 1.0):
            with self.subTest(x=x):
                self.assertTrue(
                    _approx_equal(chebyshev_t(2, x), 2 * x * x - 1)
                )

    def test_t3_recurrence(self):
        # T_3(x) = 4x^3 - 3x
        for x in (-1.0, -0.5, 0.0, 0.5, 1.0):
            with self.subTest(x=x):
                self.assertTrue(
                    _approx_equal(chebyshev_t(3, x), 4 * x ** 3 - 3 * x)
                )

    def test_t4_recurrence(self):
        # T_4(x) = 8x^4 - 8x^2 + 1
        for x in (-1.0, -0.25, 0.0, 0.25, 1.0):
            with self.subTest(x=x):
                self.assertTrue(
                    _approx_equal(chebyshev_t(4, x), 8 * x ** 4 - 8 * x ** 2 + 1)
                )

    def test_cos_identity_on_unit_interval(self):
        # For |x| <= 1: T_n(cos(theta)) = cos(n theta).
        for n in (0, 1, 5, 10):
            for theta in (0.0, 0.3, 1.0, math.pi / 2, math.pi):
                x = math.cos(theta)
                with self.subTest(n=n, theta=theta):
                    self.assertTrue(
                        _approx_equal(
                            chebyshev_t(n, x),
                            math.cos(n * theta),
                            tol=1e-10,
                        )
                    )

    def test_value_at_one(self):
        # T_n(1) = 1 for all n >= 0.
        for n in (0, 1, 2, 7, 20):
            with self.subTest(n=n):
                self.assertTrue(_approx_equal(chebyshev_t(n, 1.0), 1.0))

    def test_value_at_minus_one(self):
        # T_n(-1) = (-1)^n.
        for n in (0, 1, 2, 7, 20):
            with self.subTest(n=n):
                self.assertTrue(
                    _approx_equal(chebyshev_t(n, -1.0), (-1.0) ** n)
                )

    def test_outside_unit_interval(self):
        # For |x| > 1 the recurrence still works; verify against the
        # identity T_n(x) = cosh(n arcosh x).
        for n in (2, 3, 5):
            for x in (1.5, 2.0, 10.0):
                with self.subTest(n=n, x=x):
                    expected = math.cosh(n * math.acosh(x))
                    self.assertTrue(
                        _approx_equal(chebyshev_t(n, x), expected, tol=1e-10)
                    )

    def test_negative_n_raises(self):
        with self.assertRaises(ValueError):
            chebyshev_t(-1, 0.5)

    def test_zero_x(self):
        # T_n(0) = cos(n pi/2): 0 for odd n, (-1)^(n/2) for even n.
        self.assertEqual(chebyshev_t(0, 0.0), 1.0)
        self.assertTrue(_approx_equal(chebyshev_t(1, 0.0), 0.0))
        self.assertTrue(_approx_equal(chebyshev_t(2, 0.0), -1.0))
        self.assertTrue(_approx_equal(chebyshev_t(3, 0.0), 0.0))
        self.assertTrue(_approx_equal(chebyshev_t(4, 0.0), 1.0))

    def test_large_n_stability_at_one(self):
        # T_50(1) should still be exactly 1 up to rounding.
        self.assertTrue(_approx_equal(chebyshev_t(50, 1.0), 1.0, tol=1e-9))


class TestChebyshevU(unittest.TestCase):
    """Tests for the second-kind polynomial U_n."""

    def test_u0_is_one(self):
        for x in (-2.0, -1.0, 0.0, 0.5, 1.0, 3.14):
            with self.subTest(x=x):
                self.assertEqual(chebyshev_u(0, x), 1.0)

    def test_u1_is_two_x(self):
        for x in (-2.0, -1.0, 0.0, 0.5, 1.0, 3.14):
            with self.subTest(x=x):
                self.assertTrue(_approx_equal(chebyshev_u(1, x), 2 * x))

    def test_u2_recurrence(self):
        # U_2(x) = 4x^2 - 1
        for x in (-1.0, 0.0, 0.5, 1.0):
            with self.subTest(x=x):
                self.assertTrue(
                    _approx_equal(chebyshev_u(2, x), 4 * x * x - 1)
                )

    def test_u3_recurrence(self):
        # U_3(x) = 8x^3 - 4x
        for x in (-1.0, -0.5, 0.0, 0.5, 1.0):
            with self.subTest(x=x):
                self.assertTrue(
                    _approx_equal(chebyshev_u(3, x), 8 * x ** 3 - 4 * x)
                )

    def test_u4_recurrence(self):
        # U_4(x) = 16x^4 - 12x^2 + 1
        for x in (-1.0, -0.25, 0.0, 0.25, 1.0):
            with self.subTest(x=x):
                self.assertTrue(
                    _approx_equal(
                        chebyshev_u(4, x), 16 * x ** 4 - 12 * x ** 2 + 1
                    )
                )

    def test_sin_identity_on_unit_interval(self):
        # For |x| <= 1: U_n(cos(theta)) = sin((n+1) theta) / sin(theta).
        for n in (1, 3, 6):
            for theta in (0.2, 0.7, 1.0, 2.5):
                x = math.cos(theta)
                with self.subTest(n=n, theta=theta):
                    expected = math.sin((n + 1) * theta) / math.sin(theta)
                    self.assertTrue(
                        _approx_equal(chebyshev_u(n, x), expected, tol=1e-10)
                    )

    def test_value_at_one(self):
        # U_n(1) = n + 1.
        for n in (0, 1, 2, 5, 15):
            with self.subTest(n=n):
                self.assertTrue(_approx_equal(chebyshev_u(n, 1.0), n + 1))

    def test_value_at_minus_one(self):
        # U_n(-1) = (-1)^n (n + 1).
        for n in (0, 1, 2, 5, 15):
            with self.subTest(n=n):
                self.assertTrue(
                    _approx_equal(
                        chebyshev_u(n, -1.0), (-1.0) ** n * (n + 1)
                    )
                )

    def test_outside_unit_interval(self):
        # For |x| > 1: U_n(x) = sinh((n+1) arcosh x) / sinh(arcosh x).
        for n in (2, 3, 5):
            for x in (1.5, 2.0, 10.0):
                with self.subTest(n=n, x=x):
                    a = math.acosh(x)
                    expected = math.sinh((n + 1) * a) / math.sinh(a)
                    self.assertTrue(
                        _approx_equal(chebyshev_u(n, x), expected, tol=1e-10)
                    )

    def test_negative_n_raises(self):
        with self.assertRaises(ValueError):
            chebyshev_u(-1, 0.5)

    def test_zero_x(self):
        # U_n(0) = sin((n+1) pi/2) / sin(pi/2) = (-1)^((n-1)/2) for odd n,
        # 0 for even n > 0.
        self.assertEqual(chebyshev_u(0, 0.0), 1.0)
        self.assertTrue(_approx_equal(chebyshev_u(1, 0.0), 0.0))
        self.assertTrue(_approx_equal(chebyshev_u(2, 0.0), -1.0))
        self.assertTrue(_approx_equal(chebyshev_u(3, 0.0), 0.0))
        self.assertTrue(_approx_equal(chebyshev_u(4, 0.0), 1.0))

    def test_large_n_stability_at_one(self):
        # U_50(1) = 51 up to rounding.
        self.assertTrue(_approx_equal(chebyshev_u(50, 1.0), 51.0, tol=1e-9))


if __name__ == "__main__":
    unittest.main()
