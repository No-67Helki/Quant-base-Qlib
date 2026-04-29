# =============================================================================
# Alpha158HF: Alpha158 + 5 个高频近似因子
# -----------------------------------------------------------------------------
# 设计目标：
#   * 完全兼容 Alpha158 的 158 个因子（不删除任何原有特征）
#   * 在其上追加 5 个"高频"维度的因子，专门服务日内做T信号
#   * 当前实现使用日频 OHLCV 表达式（无需分钟数据即可跑通）
#   * 当 ~/.qlib/qlib_data/cn_data_1min 数据导入后，可在配置中切换 freq
#     或将下面的因子替换为基于 Resample/DayLast 的 1min 聚合版本
#
# 5 个因子（与对话约定一致）：
#   1. OVN_GAP    : 隔夜跳空 = open / 昨收 - 1
#   2. MOM_TAIL5  : 尾盘动量近似 = 最近 5 日 (close-open)/open 的均值
#                   （盘中力度持续性，分钟数据可换成最后 30min 收益）
#   3. RNG_OPEN   : 日内振幅 = (high - low) / open
#                   （分钟数据可换成开盘 30min 振幅）
#   4. INTRA_VOL5 : 日内波动率近似 = 5 日日收益率 std
#                   （分钟数据可换成单日分钟收益 std × √240）
#   5. PV_DIV5    : 量价背离 = 5 日 价格变动 与 成交量变动 的相关系数
#                   （分钟数据可换成 close / VWAP - 1）
# =============================================================================
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.data.loader import Alpha158DL


class Alpha158HF(Alpha158):
    """Alpha158 + 5 个高频近似因子。

    与原版 Alpha158 完全兼容，通过 ``get_feature_config`` 在原有特征列表末尾
    追加 5 列新特征，列名分别为 ``OVN_GAP``、``MOM_TAIL5``、``RNG_OPEN``、
    ``INTRA_VOL5``、``PV_DIV5``。
    """

    # 高频近似因子定义（日频表达式）
    HF_FIELDS = [
        # 1. 隔夜跳空：开盘相对昨收的相对涨跌
        "$open / Ref($close, 1) - 1",
        # 2. 尾盘动量近似：5 日盘中收益均值
        "Mean(($close - $open) / $open, 5)",
        # 3. 日内振幅：当日 (高 - 低) / 开盘
        "($high - $low) / $open",
        # 4. 日内波动率近似：5 日日收益率 std
        "Std($close / Ref($close, 1) - 1, 5)",
        # 5. 量价背离：5 日 价格变动 vs 成交量变动 的相关系数（取负方向更直观，这里保留原相关系数）
        "Corr($close / Ref($close, 1) - 1, "
        "Log($volume / Ref($volume, 1) + 1e-12), 5)",
    ]
    HF_NAMES = ["OVN_GAP", "MOM_TAIL5", "RNG_OPEN", "INTRA_VOL5", "PV_DIV5"]

    def get_feature_config(self):
        # 复用 Alpha158 默认配置，得到 158 个原始因子
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        fields, names = Alpha158DL.get_feature_config(conf)
        # 追加 5 个高频因子
        fields = list(fields) + list(self.HF_FIELDS)
        names = list(names) + list(self.HF_NAMES)
        return fields, names
