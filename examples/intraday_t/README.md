# 单股票日内做T策略 (DoubleEnsemble + Rolling + Nested TWAP)

围绕**单只股票**的日内做T（T+0模拟）策略框架。

## 架构总览

```
日级别决策循环
├── DoubleEnsemble 模型预测下一日收益（基于Alpha158因子）
├── 滚动训练 (RollingGen, 默认每20交易日重训)
└── IntradayTStrategy
        - score > buy_thresh  → 当日 BUY
        - score < sell_thresh → 当日 SELL（允许首笔卖空：消耗已有持仓）
        ↓
        TradeDecision 传递给 NestedExecutor
        ↓
分钟级执行
└── TWAPStrategy 在交易日内将日级订单按时间均匀拆单 (1min粒度)
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `config_doubleensemble.yaml` | 模型与回测配置（已设置单股+初始持仓+嵌套TWAP） |
| `intraday_t_strategy.py` | 自定义日级T策略，允许已有持仓时第一笔为卖出 |
| `workflow.py` | 全流程入口（数据→训练→预测→嵌套TWAP回测） |
| `rolling_train.py` | 滚动训练驱动脚本 |

## 标的与参数

- 标的：**SZ303516** （6位数字`303516`，深市；如属其他市场请改 `instruments`）
- 初始持仓：**2000 股**（可在 `config_doubleensemble.yaml` `account` 节修改）
- 初始现金：**500,000**
- 回测区间：默认 `2022-01-01 ~ 2024-12-31`
- 滚动步长：20 交易日重训一次，预测视野 horizon=1 日

## 数据要求

Qlib 数据存放路径（已自动建好）：

```
~/.qlib/qlib_data/cn_data/         # 日频数据（必需）
~/.qlib/qlib_data/cn_data_1min/    # 分钟频数据（嵌套TWAP执行需要）
```

数据获取方式（任选）：

1. **Qlib官方下载**（仅日频，覆盖A股全市场）：
   ```bash
   python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
   ```

2. **手动放置**：
   将自有数据按 Qlib 二进制格式放入对应 `features/<instrument>/` 目录。
   分钟数据可参考 `scripts/data_collector/yahoo/` 与 `scripts/dump_bin.py`。

## 运行

```bash
# 单次训练 + 回测（开发调试用，约5-10分钟）
python workflow.py run_once

# 滚动训练 + 回测（推荐，约30-60分钟）
python rolling_train.py run
```

## 输出

- `mlruns/` ：MLflow 实验记录（信号、IC、回测报告）
- 控制台打印：
  - `IC / ICIR / Rank IC`
  - `Annualized Return / Sharpe / Max Drawdown`
  - 嵌套频率指标（1day / 1min）

## 注意事项

1. **首笔卖出**：`IntradayTStrategy` 已显式支持，前提是 `account.position_dict` 中预置了股票持仓。
2. **不支持裸卖空**：超出当前持仓的 SELL 会被自动截断，不会报错也不会变成做空。
3. **滚动训练耗时**：CPU 单次重训约 3-5 分钟，30 个滚动窗口约 1-2 小时。
4. **核显/CPU**：DoubleEnsemble 基于 LightGBM，无需 GPU。
