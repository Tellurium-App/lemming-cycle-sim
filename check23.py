"""検証2（感度）と検証3（揺らぎ）。判定線は CRITERIA2.md。"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import sim

BURN = 50


def run(p, years, rw_series=None):
    n, _ = sim.simulate(p, years, rw_series=rw_series)
    return n


def shape(annual):
    """山から次の山までの間で、増えた年数と減った年数（間ごと）"""
    peaks = [i for i in range(1, len(annual) - 1) if annual[i] > annual[i - 1] and annual[i] >= annual[i + 1]]
    ups, downs = [], []
    for a, b in zip(peaks[:-1], peaks[1:]):
        d = np.diff(np.log(annual[a:b + 1]))
        ups.append(int((d > 0).sum())); downs.append(int((d <= 0).sum()))
    return ups, downs


def check2():
    base = sim.Params()
    grid = {"D": [0.10, 0.12], "d_l": [0.0, 0.2], "d_h": [3.5, 4.5], "r_s": [0.0, 0.4],
            "r_w": [4.0, 6.0], "v": [3.0, 5.0], "c": [750.0, 1250.0]}
    rows = []
    for k, vals in grid.items():
        for v in vals:
            p = replace(base, **{k: v})
            if k == "D":
                pass  # D は死亡率の切り替え密度も兼ねる（元コードと同じ）
            n = run(p, BURN + 200)
            if not np.isfinite(n[-1]):
                rows.append(dict(param=k, value=v, verdict="発散")); continue
            n = n[BURN:]
            period, ratio, gaps = sim.measure(n)
            rows.append(dict(param=k, value=v, period=period, ratio=round(ratio, 1),
                             gaps={g: gaps.count(g) for g in sorted(set(gaps))}, verdict=sim.verdict(period, ratio)))
            print(f"  {k}={v}: 周期 {period} 振れ幅 {ratio:.3g} {rows[-1]['gaps']} → {rows[-1]['verdict']}", flush=True)
    ok = sum(r["verdict"] == "再現した" for r in rows)
    frac = ok / len(rows)
    v = "頑健" if frac >= 0.8 else "敏感" if frac < 0.5 else "保留"
    print(f"検証2: {ok}/{len(rows)} = {frac:.0%} → {v}")
    return dict(rows=rows, ok=ok, n=len(rows), verdict=v)


def check3():
    base = sim.Params()
    out = {}
    for sigma in (0.1, 0.2, 0.3):
        gaps_all, ups_all, downs_all, diverged = [], [], [], 0
        for seed in range(10):
            rng = np.random.default_rng(seed)
            years = BURN + 150
            rw = base.r_w * np.exp(sigma * rng.standard_normal(years))
            n = run(base, years, rw_series=rw)
            if not np.isfinite(n[-1]):
                diverged += 1; continue
            n = n[BURN:]
            _, _, gaps = sim.measure(n)
            ups, downs = shape(n)
            gaps_all += gaps; ups_all += ups; downs_all += downs
        med = float(np.median(gaps_all))
        out[sigma] = dict(period_median=med, gap_counts={g: gaps_all.count(g) for g in sorted(set(gaps_all))},
                          up_median=float(np.median(ups_all)), down_median=float(np.median(downs_all)),
                          up_down_counts={f"増{u}減{d}": sum(1 for a, b in zip(ups_all, downs_all) if (a, b) == (u, d))
                                          for u, d in sorted(set(zip(ups_all, downs_all)))},
                          diverged=diverged)
        print(f"  σ={sigma}: 周期中央値 {med} 間隔 {out[sigma]['gap_counts']} 形 {out[sigma]['up_down_counts']} 発散 {diverged}", flush=True)
    keep = all(3.5 <= o["period_median"] <= 4.5 for o in out.values())
    real_shape = {s: o["down_median"] > o["up_median"] for s, o in out.items()}
    print(f"検証3: 4年が残る={keep} 形が本物寄り={real_shape}")
    return dict(by_sigma=out, four_years_kept=keep, shape_real_like=real_shape)


if __name__ == "__main__":
    print("検証2"); r2 = check2()
    print("検証3"); r3 = check3()
    Path(__file__).with_name("results23.json").write_text(
        json.dumps(dict(check2=r2, check3={k: v for k, v in r3.items()}), ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
