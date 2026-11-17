"""3D（ボクセル）用に、日ごとの密度を JSON に書き出す。

使い方: python export_daily.py [--K 50] [--years 8] [--out data/trajectory_base.json]
最初の50年を捨て、続く years 年を1日刻みで出す。年の数え始めは10/22（論文の暦）。
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import sim


def daily(p, burn, years):
    """simulate と同じ区切り・同じ出産の扱いで回し、burn 年以降を1日刻みで返す。"""
    cuts = sorted({p.t_return_owl_fox, p.t_return_skua, p.t_snowmelt, p.t_owl_hatch, p.t_stoat_born,
                   p.t_skua_born, p.t_skua_leaves, p.t_owl_leaves})
    N, P = 0.1, 0.001
    rows = []
    for year in range(burn + years):
        edges = [year] + [year + c for c in cuts] + [year + 1]
        nprime = N
        for a, b in zip(edges[:-1], edges[1:]):
            sol = solve_ivp(sim.rhs, (a, b), [N, P], args=(p, nprime, "in_year"), method="LSODA",
                            rtol=1e-8, atol=1e-12, dense_output=year >= burn)
            if year >= burn:
                for d in range(365):
                    t = year + d / 365
                    if a <= t < b:
                        n, s = (max(float(v), 0.0) for v in sol.sol(t))
                        po, yo, ps, ys, pf, yf = sim.predators(t, n, p, nprime)
                        rows.append(dict(day=(year - burn) * 365 + d, year=year - burn, doy=d,
                                         snow=d / 365 < p.t_snowmelt, lemming=round(n, 5), stoat=round(s, 6),
                                         owl=round(po + yo, 6), skua=round(ps + ys, 6), fox=round(pf + yf, 6)))
            N, P = max(sol.y[0, -1], 0.0), max(sol.y[1, -1], 0.0)
            if abs(b - (year + p.t_snowmelt)) < 1e-12:
                nprime = N
            if abs(b - (year + p.t_stoat_born)) < 1e-12:
                P *= 1 + p.v
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=float, default=float("inf"))
    ap.add_argument("--burn", type=int, default=50)
    ap.add_argument("--years", type=int, default=8)
    ap.add_argument("--out", default="data/trajectory_base.json")
    a = ap.parse_args()
    p = replace(sim.Params(), K=a.K)
    rows = daily(p, a.burn, a.years)
    n_ref, _ = sim.simulate(p, a.burn + a.years)
    snow_rows = [r for r in rows if r["doy"] == 236]  # 6/15 = 236/365（区間の端ちょうど）
    print("雪解けの日（書き出し）:", [r["lemming"] for r in snow_rows])
    print("雪解けの日（simulate）:", [round(float(x), 5) for x in n_ref[a.burn:]])
    meta = dict(model="Gilg et al. 2003（自前実装 projects/lemming_cycle/sim.py）", K=a.K, burn_years=a.burn,
                units="1ha あたりの密度", calendar="day 0 = 10/22・doy 236 = 6/15 雪解け・doy 252 = 7/1 オコジョ出産",
                fields=dict(lemming="レミング", stoat="オコジョ", owl="シロフクロウ（成鳥＋ひな）",
                            skua="シロハラトウゾクカモメ（成鳥＋ひな）", fox="ホッキョクギツネ（成獣＋子）",
                            snow="雪の下の季節（雪解け前）"))
    out = Path(a.out) if Path(a.out).is_absolute() else Path(__file__).parent / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(meta=meta, days=rows), ensure_ascii=False), encoding="utf-8")
    print("書き出し:", out, len(rows), "日")


if __name__ == "__main__":
    main()
