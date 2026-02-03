from dataclasses import dataclass

# from typing import Optional


@dataclass
class Info:
    current_iter: int = -1
    max_iter = 10000
    # height: float = 1.75  # 默认值
    # email: Optional[str] = None  # 可选字段

    # 可以定义方法
    # def is_adult(self) -> bool:
    #     return self.age >= 18
