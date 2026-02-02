from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional
import math

F = Callable[[float, float], float]
D = Callable[[float, float], float]


@dataclass
class Solution:
    xs: List[float]
    ys: List[float]


def linspace(a: float, b: float, n: int) -> List[float]:
    if n < 2:
        return [a]
    h = (b - a) / (n - 1)
    return [a + i * h for i in range(n)]


# 1) Picard’s method
def picard_method(
    f: F,
    x0: float,
    y0: float,
    x_end: float,
    n_steps: int,
    n_iter: int = 5,
) -> Solution:
    xs = linspace(x0, x_end, n_steps + 1)
    h = (x_end - x0) / n_steps

    y_prev = [y0 for _ in xs]

    for _ in range(n_iter):
        y_new = [y0]
        integral = 0.0
        for i in range(1, len(xs)):
            t0, t1 = xs[i - 1], xs[i]
            integral += 0.5 * h * (f(t0, y_prev[i - 1]) + f(t1, y_prev[i]))
            y_new.append(y0 + integral)
        y_prev = y_new

    return Solution(xs, y_prev)


# 2) Taylor’s series method

def taylor_method(
    f: F,
    derivatives: List[D],
    x0: float,
    y0: float,
    x_end: float,
    n_steps: int,
) -> Solution:
    xs = linspace(x0, x_end, n_steps + 1)
    h = (x_end - x0) / n_steps
    ys = [y0]

    for i in range(n_steps):
        x = xs[i]
        y = ys[-1]
        y_next = y + h * f(x, y)

        for j, g in enumerate(derivatives):
            order = j + 2
            y_next += (h ** order) / math.factorial(order) * g(x, y)

        ys.append(y_next)

    return Solution(xs, ys)


# 3)  Euler method
def euler_method(f: F, x0: float, y0: float, x_end: float, n_steps: int) -> Solution:
    xs = linspace(x0, x_end, n_steps + 1)
    h = (x_end - x0) / n_steps
    ys = [y0]

    for i in range(n_steps):
        x = xs[i]
        y = ys[-1]
        ys.append(y + h * f(x, y))

    return Solution(xs, ys)


# 4) Modified Euler method
def modified_euler_method(f: F, x0: float, y0: float, x_end: float, n_steps: int) -> Solution:
    xs = linspace(x0, x_end, n_steps + 1)
    h = (x_end - x0) / n_steps
    ys = [y0]

    for i in range(n_steps):
        x = xs[i]
        y = ys[-1]
        k1 = f(x, y)
        y_predict = y + h * k1
        k2 = f(x + h, y_predict)
        ys.append(y + (h / 2.0) * (k1 + k2))

    return Solution(xs, ys)


# 5) Runge-Kutta methods
def rk2_midpoint(f: F, x0: float, y0: float, x_end: float, n_steps: int) -> Solution:
    xs = linspace(x0, x_end, n_steps + 1)
    h = (x_end - x0) / n_steps
    ys = [y0]

    for i in range(n_steps):
        x = xs[i]
        y = ys[-1]
        k1 = f(x, y)
        k2 = f(x + h/2.0, y + h/2.0 * k1)
        ys.append(y + h * k2)

    return Solution(xs, ys)


def rk4(f: F, x0: float, y0: float, x_end: float, n_steps: int) -> Solution:
    xs = linspace(x0, x_end, n_steps + 1)
    h = (x_end - x0) / n_steps
    ys = [y0]

    for i in range(n_steps):
        x = xs[i]
        y = ys[-1]
        k1 = f(x, y)
        k2 = f(x + h/2.0, y + h/2.0 * k1)
        k3 = f(x + h/2.0, y + h/2.0 * k2)
        k4 = f(x + h, y + h * k3)
        ys.append(y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4))

    return Solution(xs, ys)


# ПРИМЕР ИСПОЛЬЗОВАНИЯ
if __name__ == "__main__":
    # Пример ОДУ: y' = y - x^2 + 1, y(0)=0.5
    def f(x: float, y: float) -> float:
        return y - x*x + 1

    x0, y0 = 0.0, 0.5
    x_end = 2.0
    n_steps = 10

    sol_euler = euler_method(f, x0, y0, x_end, n_steps)
    sol_heun = modified_euler_method(f, x0, y0, x_end, n_steps)
    sol_rk4 = rk4(f, x0, y0, x_end, n_steps)
    sol_picard = picard_method(f, x0, y0, x_end, n_steps, n_iter=6)



    def y2(x: float, y: float) -> float:
        return -2*x + (y - x*x + 1)

    sol_taylor2 = taylor_method(f, derivatives=[y2], x0=x0, y0=y0, x_end=x_end, n_steps=n_steps)


    print("x\tEuler\t\tHeun\t\tRK4\t\tPicard\t\tTaylor2")
    for i in range(len(sol_euler.xs)):
        x = sol_euler.xs[i]
        print(f"{x:.2f}\t{sol_euler.ys[i]:.6f}\t{sol_heun.ys[i]:.6f}\t"
              f"{sol_rk4.ys[i]:.6f}\t{sol_picard.ys[i]:.6f}\t{sol_taylor2.ys[i]:.6f}")
