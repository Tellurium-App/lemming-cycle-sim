"""検証4: 餌の上限 K を足す。判定線は CRITERIA4.md。"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import sim
from check23 import shape

BURN, KEEP = 50, 150


def one(p, rw=None, stoats=True):
    n, _ = sim.simulate(p, BURN + KEEP, rw_series=rw, stoats=stoats)
    if not np.isfinite(n[-1]):
        return None
    return n[BURN:]


def judge_K(K, stoats=True):
    p = replace(sim.Params(), K=K)
    res = {}
    n = one(p, stoats=stoats)
    if n is not None:
        per, ratio, gaps = sim.measure(n)
        res["no_noise"] = dict(period=per, ratio=round(ratio, 1), gaps={g: gaps.count(g) for g in sorted(set(gaps))},
                               first12=[round(float(x), 3) for x in n[:12]])
    else:
        res["no_noise"] = "発散"
    div, meds, sig = 0, {}, {}
    for sigma in (0.1, 0.2, 0.3):
        gaps_all, ups, downs = [], [], []
        for seed in range(10):
            rng = np.random.default_rng(seed)
            rw = p.r_w * np.exp(sigma * rng.standard_normal(BURN + KEEP))
            n = one(p, rw, stoats)
            if n is None:
                div += 1; continue
            _, _, g = sim.measure(n); u, d = shape(n)
            gaps_all += g; ups += u; downs += d
        med = float(np.median(gaps_all)) if gaps_all else float("nan")
        sig[sigma] = dict(period_median=med, gaps={g: gaps_all.count(g) for g in sorted(set(gaps_all))},
                          up_median=float(np.median(ups)) if ups else None,
                          down_median=float(np.median(downs)) if downs else None,
                          real_shape=(bool(np.median(downs) > np.median(ups)) if ups else None))
    if div:
        v = "止まらない"
    elif all(3.5 <= s["period_median"] <= 4.5 for s in sig.values()):
        v = "4年が残る"
    else:
        v = "周期が変わる"
    res.update(diverged=div, by_sigma=sig, verdict=v)
    return res


if __name__ == "__main__":
    out = {}
    for K in (10, 20, 50, 100):
        r = judge_K(K); out[f"K={K}"] = r
        print(f"K={K}: {r['verdict']} 発散{r['diverged']} 揺らぎなし={r['no_noise'] if isinstance(r['no_noise'], str) else (r['no_noise']['period'], r['no_noise']['gaps'])}", flush=True)
        for s, d in r["by_sigma"].items():
            print(f"   σ={s}: 中央値{d['period_median']} 間隔{d['gaps']} 増{d['up_median']}減{d['down_median']} 本物寄り={d['real_shape']}", flush=True)
    ok = [int(k.split("=")[1]) for k, r in out.items() if r["verdict"] == "4年が残る"]
    print("4年が残るK:", ok)
    if ok:
        Kmax = max(ok); r = judge_K(Kmax, stoats=False); out[f"K={Kmax}_オコジョなし"] = r
        print(f"対照 K={Kmax} オコジョなし: {r['verdict']} 揺らぎなし={r['no_noise'] if isinstance(r['no_noise'], str) else (r['no_noise']['period'], r['no_noise']['gaps'])}")
        for s, d in r["by_sigma"].items():
            print(f"   σ={s}: 中央値{d['period_median']} 間隔{d['gaps']}")
    Path(__file__).with_name("results4.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
