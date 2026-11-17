"""Gilg, Hanski & Sittler 2003 のレミング周期モデルを自前で書き直したもの。

式と数値の出典: github.com/fbarraquand/lemmingCycles_ODEmodels
  predation/GilgEtAl2003/original_model（GPL-3.0）。数値は事実として使い、コードは訳していない。
単位: 密度は 1ha あたり、時間は年（10/22 を 0 とする）。

使い方: python sim.py  → CRITERIA.md の4条件を回して results.json と表を出す
"""
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Params:
    # 1年の暦（10/22 = 0）
    t_return_owl_fox: float = 191 / 365   # 5/1 フクロウ・キツネが戻る
    t_return_skua: float = 226 / 365      # 6/5 トウゾクカモメが戻る
    t_snowmelt: float = 236 / 365         # 6/15 雪解け・子ギツネ誕生
    t_owl_hatch: float = 246 / 365        # 6/25 フクロウのひな
    t_stoat_born: float = 252 / 365       # 7/1 オコジョの子
    t_skua_born: float = 265 / 365        # 7/14 カモメのひな
    t_skua_leaves: float = 297 / 365      # 8/15
    t_owl_leaves: float = 343 / 365       # 9/30
    t_fox_leaves: float = 1.0             # 10/22
    # レミング
    r_w: float = 5.0    # 冬（雪の下）の増加率
    r_s: float = 0.8    # 夏の増加率
    K: float = float('inf')  # 餌の上限（検証4で追加・論文には無い）
    # オコジョ
    v: float = 4.0      # 出産で (1+v) 倍
    c: float = 1000.0   # 1頭が1年に食べる上限
    D: float = 0.08     # 半飽和密度（= 死亡率が切り替わる密度）
    d_l: float = 0.1    # 餌が多いときの死亡率
    d_h: float = 4.0    # 餌が少ないときの死亡率
    b: float = 25.0     # 切り替わりの鋭さ
    # ジェネラリスト（機能の応答）
    W_o: float = 4.7 * 365; D_o: float = 1.08
    W_l: float = 4.4 * 365; D_l: float = 2.2
    W_f: float = 3.8 * 365; D_f: float = 0.13
    # 成鳥・成獣の数（数の応答）
    b_o: float = 0.00366; Y_o: float = 2.86
    P_l: float = 0.02
    b_f: float = 0.0008; Y_f: float = 11.0
    # 子の数
    b1_o: float = 0.011; Y1_o: float = 4.0
    b1_l: float = 0.016; Y1_l: float = 6.0
    b1_f: float = 0.0028; Y1_f: float = 5.3


def _growth_curve(age_days, k, mid):
    return 1.0 / (1.0 + math.exp(-k * (age_days - mid)))


def predators(t, N, p: Params, nprime, age_mode="in_year"):
    """その時刻の天敵の密度（成体と子）: フクロウ・トウゾクカモメ・キツネ。"""
    tm = t % 1.0
    Np = N if tm < p.t_snowmelt else nprime

    def age(t0):
        return 365.0 * ((tm if age_mode == "in_year" else t) - t0)

    P_o = p.b_o * (Np - 2) / (p.Y_o + Np - 4) if Np > 2 and p.t_return_owl_fox < tm < p.t_owl_leaves else 0.0
    P_s = p.P_l if p.t_return_skua < tm < p.t_skua_leaves else 0.0
    P_f = p.b_f * Np**2 / (p.Y_f**2 + Np**2) if Np > 2 and p.t_return_owl_fox < tm < p.t_fox_leaves else 0.0

    Y_o = (p.b1_o * (Np - 2) / (p.Y1_o + Np - 4) * _growth_curve(age(p.t_owl_hatch), 0.36, 9.0)
           if P_o > 1e-7 and p.t_owl_hatch < tm < p.t_owl_leaves else 0.0)
    Y_s = (p.b1_l * Np**2 / (Np**2 + p.Y1_l**2) * _growth_curve(age(p.t_skua_born), 0.464, 4.55)
           if P_s > 1e-7 and p.t_skua_born < tm < p.t_skua_leaves else 0.0)
    Y_f = (p.b1_f * Np**2 / (Np**2 + p.Y1_f**2) * _growth_curve(age(p.t_snowmelt), 0.36, 9.0)
           if P_f > 1e-7 and p.t_snowmelt < tm < p.t_fox_leaves else 0.0)
    return P_o, Y_o, P_s, Y_s, P_f, Y_f


def rhs(t, y, p: Params, nprime, age_mode):
    """age_mode: "in_year" = その年の中の日齢 / "absolute" = 元コードと同じ通算時間の日齢"""
    N, P = max(y[0], 0.0), max(y[1], 0.0)
    winter = (t % 1.0) < p.t_snowmelt
    r = p.r_w if winter else p.r_s
    P_o, Y_o, P_s, Y_s, P_f, Y_f = predators(t, N, p, nprime, age_mode)
    gen = ((P_o + Y_o) * p.W_o * N**2 / (N**2 + p.D_o**2)
           + (P_s + Y_s) * p.W_l * N**4 / (N**4 + p.D_l**4)
           + (P_f + Y_f) * p.W_f * N**2 / (N**2 + p.D_f**2))
    stoat = p.c * P * N**2 / (N**2 + p.D**2)
    good = 0.5 + math.atan(p.b * (N - p.D)) / math.pi
    return [r * N * (1 - N / p.K) - gen - stoat, -P * (p.d_h - good * (p.d_h - p.d_l))]


def simulate(p: Params, years: int, stoats=True, age_mode="in_year", y0=(0.1, 0.001), rw_series=None):
    """雪解けの日（6/15）のレミング密度とオコジョ密度を年ごとに返す。rw_series は年ごとの冬の増加率（揺らぎ用）。"""
    cuts = sorted({p.t_return_owl_fox, p.t_return_skua, p.t_snowmelt, p.t_owl_hatch, p.t_stoat_born,
                   p.t_skua_born, p.t_skua_leaves, p.t_owl_leaves})
    N, P = y0[0], (y0[1] if stoats else 0.0)
    snow_N, snow_P = [], []
    base = p
    for year in range(years):
        if rw_series is not None:
            p = replace(base, r_w=float(rw_series[year]))
        edges = [year] + [year + c for c in cuts] + [year + 1]
        nprime = N
        for a, b in zip(edges[:-1], edges[1:]):
            sol = solve_ivp(rhs, (a, b), [N, P], args=(p, nprime, age_mode), method="LSODA",
                            rtol=1e-8, atol=1e-12)
            N, P = max(sol.y[0, -1], 0.0), max(sol.y[1, -1], 0.0)
            if not np.isfinite(N) or N > 1e6:  # 上限（環境収容力）の無いモデルなので、天敵が抑えないと発散する
                return np.array(snow_N + [np.inf]), np.array(snow_P + [P])
            if abs(b - (year + p.t_snowmelt)) < 1e-12:
                nprime = N
                snow_N.append(N); snow_P.append(P)
            if abs(b - (year + p.t_stoat_born)) < 1e-12:
                P *= 1 + p.v
    return np.array(snow_N), np.array(snow_P)


def measure(annual):
    peaks = [i for i in range(1, len(annual) - 1) if annual[i] > annual[i - 1] and annual[i] >= annual[i + 1]]
    gaps = np.diff(peaks)
    period = float(np.median(gaps)) if len(gaps) else float("nan")
    ratio = float(annual.max() / annual.min()) if annual.min() > 0 else float("inf")
    return period, ratio, [int(g) for g in gaps]


def verdict(period, ratio):
    if ratio < 10:
        return "再現しない"
    return "再現した" if 3.5 <= period <= 4.5 else "周期はあるが別物"


BURN, KEEP = 50, 200
CONDITIONS = {
    "1_基本(r_w=5)": dict(p=Params()),
    "2_論文本文(r_w=4)": dict(p=replace(Params(), r_w=4.0)),
    "3_オコジョなし": dict(p=Params(), stoats=False),
    "4_日齢は元コード式": dict(p=Params(), age_mode="absolute"),
}


def main():
    out = {}
    for name, cfg in CONDITIONS.items():
        n, s = simulate(cfg["p"], BURN + KEEP, stoats=cfg.get("stoats", True), age_mode=cfg.get("age_mode", "in_year"))
        if not np.isfinite(n[-1]):
            out[name] = dict(verdict="発散", diverged_year=len(n) - 1, first20=[float(x) for x in n[:20]])
            print(f"{name}: {len(n) - 1}年目に1haあたり100万匹を超えて発散 → 判定の3区分の外（発散）")
            print("   雪解けの密度:", " ".join(f"{x:.3g}" for x in n[:12]))
            continue
        n, s = n[BURN:], s[BURN:]
        period, ratio, gaps = measure(n)
        gap_hist = {g: gaps.count(g) for g in sorted(set(gaps))}
        out[name] = dict(period_median=period, ratio=ratio, gap_counts=gap_hist, verdict=verdict(period, ratio),
                         N_min=float(n.min()), N_max=float(n.max()), first20=[round(float(x), 4) for x in n[:20]],
                         stoat_first20=[round(float(x), 5) for x in s[:20]])
        print(f"{name}: 周期 {period} 年  振れ幅 {ratio:.3g} 倍  間隔の内訳 {gap_hist} → {out[name]['verdict']}")
        print("   雪解けの密度（最初の12年）:", " ".join(f"{x:.3g}" for x in n[:12]))
    Path(__file__).with_name("results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
