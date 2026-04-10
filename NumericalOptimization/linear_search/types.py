"""线搜索方法名称与类型定义。"""

from typing import Literal, TypeAlias

LineSearchName: TypeAlias = Literal[
    "golden",
    "fibonacci",
    "newton",
    "secant",
    "parabola",
    "armijo_goldstein",
    "wolf_powell",
    "simple_rule",
]

golden: LineSearchName = "golden"
fibonacci: LineSearchName = "fibonacci"
newton: LineSearchName = "newton"
secant: LineSearchName = "secant"
parabola: LineSearchName = "parabola"
armijo_goldstein: LineSearchName = "armijo_goldstein"
wolf_powell: LineSearchName = "wolf_powell"
simple_rule: LineSearchName = "simple_rule"

__all__ = [
    # "LineSearchName",
    "golden",
    "fibonacci",
    "newton",
    "secant",
    "parabola",
    "armijo_goldstein",
    "wolf_powell",
    "simple_rule",
]

