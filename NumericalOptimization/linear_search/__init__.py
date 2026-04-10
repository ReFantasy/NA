"""线搜索算法与参数定义。"""

from dataclasses import dataclass, field
import sys
from typing import Callable, Optional


@dataclass
class LineSearchParams:
    """线搜索统一参数结构体。"""

    name: str = "golden"  # 线搜索方法名称

    # 精确线搜索参数
    a: float = 0.0
    b: float = 3.0
    epsilon: float = 1e-3

    # 非精确线搜索参数
    alpha: float = 1.0
    rho: float = 0.3
    t: float = 1.1

    sigma: float = 0.5  # Wolf-Powell步长准则特有参数


from .golden import *
from .fibonacci import fibonacci
from .newton import *
from .secant import *
from .parabola import *
from .armijo import *
from .wolf import *
from .simple_rule import *
