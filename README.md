# NumericalAnalysis(NA)
数值计算及最优化理论算法实现
## Build and Run
Install [uv](https://uv.doczh.com) 

```bash
uv sync
uv pip install -e .
uv run example/nle_newton.py
```

## Test
```bash
uv run pytest
```

## Directory
```
project/
├── NumericalAnalysis/     数值计算源码
│   ├── Eigen/
│   │   └── qr.py
│   └── utils.py
│       
├── NumericalOptimization/  数值优化源码
└── README.md
```
