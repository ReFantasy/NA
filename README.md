# 数值计算及最优化理论算法实现 🏗️ 

## 安装和运行
Install [uv](https://uv.doczh.com) 

```bash
uv sync
uv pip install -e .
uv run example/nle_newton.py
```

## 单元测试
```bash
uv run pytest
```

## 目录结构
```
project/
├── NumericalAnalysis/      数值计算源码
│   ├── Eigen/
│   │   └── qr.py
│   └── utils.py
│       
├── NumericalOptimization/  数值优化源码
└── README.md
```

## 参考文献
[1] 王兵团.数值分析简明教程（第2版）:大学数学系列丛书[M].北京:清华大学出版社,2020.

[2] 龙强、赵克全.非线性最优化算法与实践（微课视频版）:跟我一起学人工智能[M].北京:清华大学出版社,2025.

[3] 高立.数值最优化方法[M].北京:北京大学出版社,2020.

[4] 张平文、李铁军.数值分析[M].北京:北京大学出版社,2018.
