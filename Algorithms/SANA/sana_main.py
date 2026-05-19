#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量调用 SANA 2.2 运行网络对齐，并将结果汇总为 Excel 文件。
"""

import os
import re
import time
import subprocess
from pathlib import Path

import pandas as pd

# ============================================================
# 可手动修改的参数
# ============================================================
RUNTIME_MINUTES = 10                         # SANA 运行时间（分钟）
SANA_EXECUTABLE = "./sana2.2"               # 可执行程序路径
NETWORKS_DIR = "networks"                   # 网络文件根目录
SANA_OUT_FILE = "sana.out"                  # SANA 默认输出文件
RESULT_XLSX = "sana_batch_results(ec).xlsx"     # 最终 Excel 输出路径

# ============================================================
# 物种与配对定义
# ============================================================
name_map = {
    "AT": "AThaliana",
    "CE": "CElegans",
    "DM": "DMelanogaster",
    "MM": "MMusculus",
    "RN": "RNorvegicus",
    "SP": "SPombe",
}

pairs = [
    ("AT", "DM"),
    ("CE", "AT"),
    ("CE", "DM"),
    ("CE", "MM"),
    ("MM", "AT"),
    ("MM", "DM"),
    ("RN", "AT"),
    ("RN", "CE"),
    ("RN", "DM"),
    ("RN", "MM"),
    ("RN", "SP"),
    ("SP", "AT"),
    ("SP", "CE"),
    ("SP", "DM"),
    ("SP", "MM"),
]


# ============================================================
# 指标解析
# ============================================================
def parse_metrics_from_text(text):
    """
    从给定文本（SANA 的 stdout 或 sana.out 内容）中解析关键指标。
    返回 dict: nc, ec, ics, s3, time（单位：秒，若能解析到）。
    解析不到的字段为 None。
    """
    metrics = {"nc": None, "ec": None, "ics": None, "s3": None, "time": None}
    if not text:
        return metrics

    # 数值模式（支持整数、小数、科学计数法、负数）
    num = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

    # 这些指标在 SANA 输出中通常以 "metric: value" 的形式出现在 Scores: 段落
    # 例如: "ec: 0.623066"。有时同一 key 会重复多次，取第一次有效值即可。
    patterns = {
        "nc":  re.compile(rf"^\s*nc\s*:\s*({num})\s*$", re.MULTILINE),
        "ec":  re.compile(rf"^\s*ec\s*:\s*({num})\s*$", re.MULTILINE),
        "ics": re.compile(rf"^\s*ics\s*:\s*({num})\s*$", re.MULTILINE),
        "s3":  re.compile(rf"^\s*s3\s*:\s*({num})\s*$", re.MULTILINE),
    }

    for key, pat in patterns.items():
        matches = pat.findall(text)
        for m in matches:
            try:
                val = float(m)
                # 跳过 nan / -nan
                if val != val:
                    continue
                metrics[key] = val
                break
            except ValueError:
                continue

    # 运行时间：优先解析 "Actual execution time = 288.712s"
    time_patterns = [
        re.compile(rf"Actual execution time\s*=\s*({num})\s*s", re.IGNORECASE),
        re.compile(rf"Execution time\s*:\s*({num})\s*s", re.IGNORECASE),
        re.compile(rf"^\s*runtime\s*:\s*({num})\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(rf"elapsed\s*time\s*[:=]\s*({num})", re.IGNORECASE),
    ]
    for pat in time_patterns:
        m = pat.search(text)
        if m:
            try:
                metrics["time"] = float(m.group(1))
                break
            except ValueError:
                pass

    return metrics


def merge_metrics(primary, fallback):
    """若 primary 中某指标为 None，则用 fallback 的值补。"""
    merged = dict(primary)
    for k, v in merged.items():
        if v is None:
            merged[k] = fallback.get(k)
    return merged


# ============================================================
# 运行 SANA
# ============================================================
def run_sana_on_pair(sp1, sp2, runtime_minutes):
    """
    针对一个物种配对运行 SANA，返回结果 dict。
    """
    full1 = name_map[sp1]
    full2 = name_map[sp2]

    path1 = os.path.join(NETWORKS_DIR, full1, f"{full1}.gw")
    path2 = os.path.join(NETWORKS_DIR, full2, f"{full2}.gw")

    cmd = [
        SANA_EXECUTABLE,
        "-fg1", path1,
        "-fg2", path2,
        "-ec", "1",#修改目标函数的命令
        "-t", str(runtime_minutes),
        "-tolerance", "0",
    ]

    result = {
        "species1": sp1,
        "species2": sp2,
        "network1": full1,
        "network2": full2,
        "runtime_minutes": runtime_minutes,
        "nc": None,
        "ec": None,
        "ics": None,
        "s3": None,
        "elapsed_seconds": None,
        "return_code": None,
    }

    # 清理上一轮可能遗留的 sana.out，避免误读
    try:
        if Path(SANA_OUT_FILE).exists():
            Path(SANA_OUT_FILE).unlink()
    except OSError:
        pass

    print(f"[RUN] {sp1} -> {sp2}  cmd: {' '.join(cmd)}")

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        stdout_text = proc.stdout or ""
        return_code = proc.returncode
    except FileNotFoundError as e:
        print(f"[ERROR] 无法运行 SANA: {e}")
        result["return_code"] = -1
        result["elapsed_seconds"] = time.time() - t0
        return result
    except Exception as e:
        print(f"[ERROR] 运行 SANA 时发生异常: {e}")
        result["return_code"] = -1
        result["elapsed_seconds"] = time.time() - t0
        return result

    wall_elapsed = time.time() - t0
    result["return_code"] = return_code

    # 先从 stdout 解析
    m_stdout = parse_metrics_from_text(stdout_text)

    # 再从 sana.out 解析（作为补充）
    m_file = {"nc": None, "ec": None, "ics": None, "s3": None, "time": None}
    try:
        if Path(SANA_OUT_FILE).exists():
            with open(SANA_OUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
                file_text = f.read()
            m_file = parse_metrics_from_text(file_text)
    except Exception as e:
        print(f"[WARN] 读取 {SANA_OUT_FILE} 失败: {e}")

    merged = merge_metrics(m_stdout, m_file)

    result["nc"] = merged["nc"]
    result["ec"] = merged["ec"]
    result["ics"] = merged["ics"]
    result["s3"] = merged["s3"]

    # 运行时间：优先使用 SANA 自己报告的，否则用 wall clock
    if merged["time"] is not None:
        result["elapsed_seconds"] = merged["time"]
    else:
        result["elapsed_seconds"] = wall_elapsed

    print(f"[DONE] {sp1}->{sp2}  rc={return_code}  "
          f"ec={result['ec']}  ics={result['ics']}  s3={result['s3']}  "
          f"nc={result['nc']}  elapsed={result['elapsed_seconds']}")

    return result


# ============================================================
# 主流程
# ============================================================
def main():
    rows = []
    for sp1, sp2 in pairs:
        try:
            row = run_sana_on_pair(sp1, sp2, RUNTIME_MINUTES)
        except Exception as e:
            # 保底：即使单次运行抛异常，也不中断整体流程
            print(f"[ERROR] pair ({sp1},{sp2}) 发生未捕获异常: {e}")
            row = {
                "species1": sp1,
                "species2": sp2,
                "network1": name_map[sp1],
                "network2": name_map[sp2],
                "runtime_minutes": RUNTIME_MINUTES,
                "nc": None,
                "ec": None,
                "ics": None,
                "s3": None,
                "elapsed_seconds": None,
                "return_code": -1,
            }
        rows.append(row)

        # 每完成一个 pair 就覆盖写一次 Excel，防止中途崩溃丢失结果
        df_tmp = pd.DataFrame(rows, columns=[
            "species1", "species2", "network1", "network2",
            "runtime_minutes", "nc", "ec", "ics", "s3",
            "elapsed_seconds", "return_code",
        ])
        try:
            df_tmp.to_excel(RESULT_XLSX, index=False)
        except Exception as e:
            print(f"[WARN] 中间写入 Excel 失败: {e}")

    # 最终输出
    df = pd.DataFrame(rows, columns=[
        "species1", "species2", "network1", "network2",
        "runtime_minutes", "nc", "ec", "ics", "s3",
        "elapsed_seconds", "return_code",
    ])
    df.to_excel(RESULT_XLSX, index=False)
    print(f"[OK] 结果已写入: {RESULT_XLSX}")


if __name__ == "__main__":
    main()
