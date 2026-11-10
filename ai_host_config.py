#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""《数字媒体程序设计》第02讲：AI主机配置与容量估算（标准库版本）。

课程任务：Strata Qwen3.8 Flash 128B INT4（32K上下文）
        + MiniMax H3 pruned INT8 视频生成。

本脚本用于核对模型权重、KV cache预算、整机报价与三方案性价比。
- 64GB / 12GB / KV 2–4GB：来自课程实验指南的计算值或设计参考。
- 主机价格：来自配套实验报告中列明的参考价/预算，不是实际采购凭证。
- tok/s及H3耗时：均是报告的情景假设，不是本机实测或性能保证。

运行：python ai_host_config.py
自检：python ai_host_config.py --self-test
导出：python ai_host_config.py --csv comparison.csv
仅使用Python标准库，无需pip安装依赖。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


PARAMETERS = 128_000_000_000  # 128B个参数；课程采用总参数量计权重
BYTES_PER_DECIMAL_GB = 1_000_000_000
BYTES_PER_GIB = 1024 ** 3
BASE_CONTEXT = 32_768
# 实验指南给出的32K、单请求KV预算，真实值取决于网络结构、KV精度等
BASE_KV_MIN_GB = 2.0
BASE_KV_MAX_GB = 4.0
# 报告采用的框架/计算图/工作缓冲估算区间（非固定软件要求）
RUNTIME_MIN_GB = 4.0
RUNTIME_MAX_GB = 8.0
H3_480P_PRUNED_INT8_VRAM_GB = 12.0  # 课程设计参考值

# 配套报告表2：2026-11-10参考人民币价格。实际成交价会随时间变化。
MAIN_BUILD_PARTS = {
    "RTX PRO 4500 Blackwell 32GB ECC": 29_999,
    "AMD Ryzen 9 9900X": 2_999,
    "ASUS ProArt X870E-CREATOR WIFI": 3_499,
    "DDR5-6000 128GB (64GB x 2)": 12_299,
    "2TB PCIe 4.0 NVMe SSD": 1_299,
    "1200W ATX 3.1 Gold PSU": 1_399,
    "Corsair 5000D AIRFLOW": 899,
    "AM5 dual-tower CPU cooler": 499,
    "120mm PWM case fans": 199,
}
UNIFIED_MINI_PC_REFERENCE_CNY = 21_999
SECOND_GPU_REFERENCE_CNY = MAIN_BUILD_PARTS["RTX PRO 4500 Blackwell 32GB ECC"]
EXTRA_DUAL_GPU_COOLING_CNY = 200


@dataclass(frozen=True)
class Plan:
    label: str
    memory: str
    cost_cny: int
    llm_tok_s_low: float
    llm_tok_s_mid: float
    llm_tok_s_high: float
    h3_480p: str
    h3_768p: str
    comment: str

    @property
    def tok_per_s_per_10000_cny(self) -> float:
        """每花一万元所对应的假设出词速度，非实际基准分数。"""
        return self.llm_tok_s_mid / (self.cost_cny / 10_000)


def weight_size_gb(bits_per_parameter: int, params: int = PARAMETERS) -> float:
    """理想纯权重体积(十进制GB)，不含量化元数据及运行开销。"""
    return params * bits_per_parameter / 8 / BYTES_PER_DECIMAL_GB


def gb_to_gib(gb: float) -> float:
    return gb * BYTES_PER_DECIMAL_GB / BYTES_PER_GIB


def required_memory_range_gb(context: int, parallel_requests: int) -> tuple[float, float]:
    """按课程32K KV参考区间线性缩放，作为初步预算；非精确KV测量。"""
    if context <= 0 or parallel_requests <= 0:
        raise ValueError("上下文和并发路数均须大于0")
    scale = context / BASE_CONTEXT
    weight = weight_size_gb(4)
    low = weight + BASE_KV_MIN_GB * scale * parallel_requests + RUNTIME_MIN_GB
    high = weight + BASE_KV_MAX_GB * scale * parallel_requests + RUNTIME_MAX_GB
    return low, high


def plans() -> list[Plan]:
    a_cost = sum(MAIN_BUILD_PARTS.values())
    c_cost = a_cost + SECOND_GPU_REFERENCE_CNY + EXTRA_DUAL_GPU_COOLING_CNY
    return [
        Plan(
            label="A 单卡分层（双任务推荐）",
            memory="32GB独显 + 128GB DDR5",
            cost_cny=a_cost,
            llm_tok_s_low=1.5, llm_tok_s_mid=2.2, llm_tok_s_high=3.0,
            h3_480p="5–12 分钟", h3_768p="16–35 分钟",
            comment="32GB显存跑H3；128B INT4需GPU+系统内存分层，建议错峰运行",
        ),
        Plan(
            label="B 统一内存（LLM购置成本优）",
            memory="128GB统一内存（Ryzen AI Max+ 395）",
            cost_cny=UNIFIED_MINI_PC_REFERENCE_CNY,
            llm_tok_s_low=1.2, llm_tok_s_mid=2.0, llm_tok_s_high=2.8,
            h3_480p="15–45 分钟", h3_768p="兼容性/性能不保证",
            comment="主机成本低；需先确认H3在该平台的后端支持",
        ),
        Plan(
            label="C 双卡（并发与分片扩展）",
            memory="2x32GB独显 + 128GB DDR5",
            cost_cny=c_cost,
            llm_tok_s_low=2.5, llm_tok_s_mid=4.0, llm_tok_s_high=5.5,
            h3_480p="5–12 分钟", h3_768p="16–35 分钟",
            comment="两张32GB显卡不等于单块64GB显存；单视频不保证线性加速",
        ),
    ]


def verify() -> None:
    """校验程序计算与实验报告关键数字是否一致。"""
    assert weight_size_gb(16) == 256.0
    assert weight_size_gb(8) == 128.0
    assert weight_size_gb(4) == 64.0
    assert round(gb_to_gib(64), 2) == 59.60
    assert required_memory_range_gb(BASE_CONTEXT, 1) == (70.0, 76.0)
    assert sum(MAIN_BUILD_PARTS.values()) == 53_091
    a, b, c = plans()
    assert (a.cost_cny, b.cost_cny, c.cost_cny) == (53_091, 21_999, 83_290)
    assert [round(p.tok_per_s_per_10000_cny, 2) for p in (a, b, c)] == [0.41, 0.91, 0.48]


def write_csv(path: Path, options: list[Plan]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "方案", "显存和内存", "参考总价_元", "LLM出词下限_tok每秒",
            "LLM出词中位估算_tok每秒", "LLM出词上限_tok每秒", "每万元中位tok每秒",
            "H3_480p_5秒单条耗时", "H3_768p_5秒单条耗时", "备注",
        ])
        for plan in options:
            writer.writerow([
                plan.label, plan.memory, plan.cost_cny,
                plan.llm_tok_s_low, plan.llm_tok_s_mid, plan.llm_tok_s_high,
                f"{plan.tok_per_s_per_10000_cny:.2f}", plan.h3_480p,
                plan.h3_768p, plan.comment,
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description="第02讲 AI主机配置与容量估算")
    parser.add_argument("--context", type=int, default=BASE_CONTEXT,
                        help="上下文窗口token数，默认32768")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="同时运行的推理请求数，默认1")
    parser.add_argument("--csv", type=Path,
                        help="可选：把三方案对照导出为UTF-8 CSV文件")
    parser.add_argument("--self-test", action="store_true",
                        help="执行公式/预算/价格一致性断言")
    args = parser.parse_args()

    if args.self_test:
        verify()
        print("[校验通过] 8组容量、价格和性价比一致性检查全部通过。\n")

    print("《数字媒体程序设计》第02讲｜AI主机配置与容量估算")
    print("=" * 56)
    print("1. 128B模型理论纯权重大小（十进制GB）")
    for bits, name in ((16, "FP16"), (8, "INT8"), (4, "INT4")):
        size = weight_size_gb(bits)
        print(f"   {name:5s}: {size:6.1f} GB = {gb_to_gib(size):6.2f} GiB")
    print("   注意：实际量化文件含额外元数据，不等于纯权重大小。")

    low, high = required_memory_range_gb(args.context, args.concurrency)
    scale = args.context / BASE_CONTEXT
    kv_low = BASE_KV_MIN_GB * scale * args.concurrency
    kv_high = BASE_KV_MAX_GB * scale * args.concurrency
    print("\n2. INT4运行总内存预算（课程近似模型）")
    print(f"   上下文={args.context:,} tokens，并发={args.concurrency}路")
    print(f"   纯权重64GB + KV {kv_low:.1f}~{kv_high:.1f}GB + 运行开销4~8GB")
    print(f"   合计约 {low:.1f}~{high:.1f}GB（未包括全部OS余量）")
    print(f"   H3 pruned INT8 480p 单任务显存参考：约{H3_480P_PRUNED_INT8_VRAM_GB:g}GB")

    a_cost = sum(MAIN_BUILD_PARTS.values())
    print("\n3. A方案（32GB独显+128GB DDR5）零件预算")
    for part, cost in MAIN_BUILD_PARTS.items():
        print(f"   {part:<41s} ¥{cost:>7,}")
    print(f"   合计：¥{a_cost:,}")
    print(f"   GPU预算占比：{100 * SECOND_GPU_REFERENCE_CNY / a_cost:.1f}%")
    print(f"   内存预算占比：{100 * MAIN_BUILD_PARTS['DDR5-6000 128GB (64GB x 2)'] / a_cost:.1f}%")
    print("   PCIe 5.0理论单方向：x16约63 GB/s；双卡x8/x8各约31.5 GB/s。")
    print("   兼容性复核：主板插槽、双卡间距、显卡长度、线材和机箱风道。")

    options = plans()
    print("\n4. 三方案对比（人民币；速度/耗时仅为报告情景估算）")
    print(f"   {'方案':<24s} {'参考价':>10s} {'tok/s中位':>10s} {'tok/s/万元':>12s}")
    for p in options:
        print(f"   {p.label:<24s} ¥{p.cost_cny:>8,} {p.llm_tok_s_mid:>10.1f} {p.tok_per_s_per_10000_cny:>12.2f}")
        print(f"      H3 480p: {p.h3_480p}; 768p: {p.h3_768p}")
    print("\n结论：A优先兼顾CUDA视频与分层文本；B更偏低成本私有文本；")
    print("      C侧重并发/多卡分片。不能用这些估算速度冒充实际测速。")

    if args.csv:
        write_csv(args.csv, options)
        print(f"\nCSV已导出：{args.csv}")


if __name__ == "__main__":
    main()
