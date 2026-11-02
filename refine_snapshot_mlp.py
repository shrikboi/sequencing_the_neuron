"""refine_snapshot_mlp.py -- exact recovery of a (leaky) ReLU MLP from a saved committee snapshot.

    python refine_snapshot_mlp.py <pop snapshot .pt> [--out out.pt]

Same refiner as `run.py --fast --design-refine` (method._mlp_refine_layer, design sampling, layer by layer, rows
scale-matched to the guess, closed-form output layer), run on the best member of a saved population.  One addition:
a unit the refiner cannot solve from the best member's row is retried from the committee's other copies -- rows of
any member that match none of the layer's refined units, clustered across members, most-supported first (the
best-output member can hold a wrong copy of a unit that the other members learned).  The black box is queried by
forward evaluation only; teacher parameters are read only for the printed error scores.
"""
import argparse, copy, time, torch
from nets import MLP
from method import param_errors, _mlp_refine_layer
from align import align_clone_to, scale_normalize_

ap = argparse.ArgumentParser()
ap.add_argument("snapshot"); ap.add_argument("--out", default=""); ap.add_argument("--device", default="cuda")
ap.add_argument("--head-points", type=int, default=20000)
ap.add_argument("--unit-report", action="store_true")
ap.add_argument("--resume", default="", help="saved output of an earlier run: take its layers < --start-layer as refined")
ap.add_argument("--start-layer", type=int, default=1, help="1-indexed hidden layer to start refining at (with --resume)")
ap.add_argument("--gf-lines", type=int, default=20)
ap.add_argument("--track-step", type=float, default=2.0); ap.add_argument("--track-need-mult", type=int, default=2)
ap.add_argument("--stop-after", type=int, default=0)
ap.add_argument("--local-refine", action="store_true", help="bounded independent eight-query kink brackets before tracking/design")
ap.add_argument("--local-rounds", type=int, default=24)
ap.add_argument("--fallback-rounds", type=int, default=16, help="design rounds per fallback when --local-refine is enabled")
ap.add_argument("--prefix-check", action="store_true",
                help="before each layer l >= 2: probe a few of its units with the design sampler; if many candidate rows put an "
                     "outsized weight on the same previous-layer coordinate j, unit j of the prefix is wrong (one wrong rare unit "
                     "otherwise makes every row of the next layer fail the angle gate); re-solve it guess-free and re-check")
ap.add_argument("--prefix-check-probe", type=int, default=24, help="--prefix-check: units probed per layer")
ap.add_argument("--prefix-check-dom", type=float, default=0.3, help="--prefix-check: |w_j|/|w| above which coordinate j votes")
ap.add_argument("--dup-check", action="store_true",
                help="after each layer: no two solved rows may coincide (a rare unit refined onto a copy of another unit passes "
                     "the second-track check and leaves a null direction e_a - e_b in every next-layer fit); the copy that moved "
                     "farthest from its own committee guess is re-solved (committee orphan rows, then guess-free)")
ap.add_argument("--dup-angle-gate", type=float, default=20.0,
                help="--dup-check: angle gate (deg) for re-solving a duplicated unit from committee orphan rows (the row must "
                     "still pass the fit/validation checks and differ from every other row); 12 rejected the exact row of a "
                     "rare unit whose best committee copy was 12.9 deg off")
ap.add_argument("--dup-orphan-tries", type=int, default=30, help="--dup-check: committee rows tried per re-solved unit")
ap.add_argument("--orphan-gauge", choices=["guess", "net"], default="guess",
                help="committee orphan rows for unsolved units: 'guess' = members aligned to the original guess member (default, "
                     "as before); 'net' = members' rows mapped onto the REFINED prefix by activation matching (a broken unit in "
                     "the guess member scrambles the 'guess' alignment; see member_rows_netgauge)")
ap.add_argument("--design-from", type=int, default=0,
                help="1-indexed hidden layer from which every row is solved with the design sampler (as layer 1) instead of "
                     "surface tracking: ~2x precision loss per layer instead of ~10-20x, at more queries (0 = off, tracking)")
ap.add_argument("--design-need", type=int, default=0,
                help="--design-from layers: kink points per unit for the design fit (0 = method.py's default max(400, 2*Din)); "
                     "the sampler raises it to its own minimum max(Din+48, 7/6 (Din+16))")
ap.add_argument("--reanchor", action="store_true",
                help="before refining layer l >= 2: rewrite the ORIGINAL guess member's layer-l rows over the refined layer-(l-1) "
                     "units, matched by activation on random inputs (no queries). Without it, layer l's guess still weights the "
                     "guess member's own layer-(l-1) units, which no longer match where rows were replaced (committee orphans, "
                     "duplicates): 13 replaced L4 units left the L5 guess 15 deg off (0.12 deg after re-anchoring)")
ap.add_argument("--design-unc-gate", type=float, default=0.0,
                help="--design-from layers: accept a design fit with uncertainty below this (sampler default 1e-5; 0 = default). "
                     "Correct deep solves sit at ~the prefix error, wrong locks at ~1e-2")
ap.add_argument("--member", default="consensus", help="guess member: 'consensus' (teacher-free vote) or an index")
a = ap.parse_args(); dev = a.device
d = torch.load(a.snapshot, map_location=dev, weights_only=False)
dims, act = d["dims"], d["act"]
T = MLP(dims, act=act).to(dev); T.load_state_dict(d["teacher_state"]); T = T.double()
for p in T.parameters():
    p.requires_grad_(False)
mems = []
for s in d["pop_states"]:
    m = MLP(dims, act=act).to(dev); m.load_state_dict(s); mems.append(m.double())
X = torch.randn(4096, dims[0], device=dev, dtype=torch.float64) * 0.5
# guess member, teacher-free: the one whose hidden rows the rest of the committee reproduces most often (cos > 0.999).
# The lowest-output-error member is NOT a good choice: on the 9-layer run it was the worst member in layers 4-8
# (independently trained members that agree on a row have both found the teacher's row).
def _rows(n, l):
    R = torch.cat([n.layers[l].weight, n.layers[l].bias[:, None]], 1); return R / R.norm(dim=1, keepdim=True)
score = []
for i, ref in enumerate(mems):
    r0 = MLP(dims, act=act).to(dev).double(); r0.load_state_dict(ref.state_dict()); scale_normalize_(r0)
    agree = 0
    for k, m in enumerate(mems):
        if k != i:
            _, r = align_clone_to(m, r0)
            agree += sum(int(((_rows(r, l) * _rows(r0, l)).sum(1) > 0.999).sum()) for l in range(len(dims) - 2))
    score.append(agree)
bi = max(range(len(mems)), key=lambda i: score[i]) if a.member == "consensus" else int(a.member)
print(f"[refine] committee row agreement per member {score}", flush=True)
nq = 0
net = mems[bi]
for p in net.parameters():
    p.requires_grad_(False)
print(f"[refine] guess = member {bi} of {len(mems)} | dims {dims} | start max_eps {param_errors(net, T)['max_eps']:.2e}", flush=True)

guess0 = copy.deepcopy(net)        # --dup-check: the guess member's original rows
anchored = {}                      # --reanchor: layer -> (W, b) guess rows over the refined prefix

# committee copies in a common (scale-normalised) gauge, for the fallback guesses
with torch.no_grad():
    nz = MLP(dims, act=act).to(dev).double(); nz.load_state_dict(net.state_dict()); scale_normalize_(nz)
    col_s, s_prev = [], torch.ones(dims[0], device=dev, dtype=torch.float64)
    for l in range(len(dims) - 2):                          # column scales nz applies to layer l's inputs
        col_s.append(s_prev); s_prev = (net.layers[l].weight * s_prev[None]).norm(dim=1)
    alt = [align_clone_to(m, nz)[1] for k, m in enumerate(mems) if k != bi]


def rows(n, l):
    R = torch.cat([n.layers[l].weight, n.layers[l].bias[:, None]], 1)
    return R / R.norm(dim=1, keepdim=True)


def orphans(l, Wr, br, ok, cos=0.99):
    """[(support, row in nz gauge)]: member rows matching no refined row of layer l, clustered, most members first"""
    Rr = torch.cat([Wr[ok] * col_s[l][None], br[ok][:, None]], 1)
    Rr = Rr / Rr.norm(dim=1, keepdim=True)
    pool = []
    for k, m in enumerate([nz] + alt):
        R = rows(m, l)
        mc = (R @ Rr.T).abs().amax(1) if len(Rr) else torch.zeros(len(R), device=R.device, dtype=R.dtype)
        pool += [(r, k) for r in R[mc < cos]]
    out, used = [], [False] * len(pool)
    for i, (ri, _) in enumerate(pool):
        if used[i]:
            continue
        g = [k for k, (rk, _) in enumerate(pool) if not used[k] and float(rk @ ri) > cos]
        for k in g:
            used[k] = True
        support=len({pool[k][1] for k in g})
        group_rows=torch.stack([pool[k][0] for k in g])
        centrality=(group_rows@group_rows.T).sum(1)
        # Try actual committee guesses: lower-median aggregation of two rows
        # takes the coordinatewise minimum and can destroy their closeness.
        for rank in centrality.argsort(descending=True)[:3].tolist():
            out.append((support,group_rows[rank]))
    return sorted(out, key=lambda t: -t[0])


def distinct_candidate(j, w, b, Wr, br, ok, tol=1e-5):
    """Do not count the same recovered hyperplane twice (sign invariant)."""
    others=ok.clone();others[j]=False
    if not bool(others.any()):return True
    r=torch.cat([w.flatten(), b.reshape(1)]).double();r=r/r.norm()
    rows=torch.cat([Wr[others],br[others,None]],1).double()
    rows=rows/rows.norm(dim=1,keepdim=True)
    distance=torch.minimum((rows-r).norm(dim=1),(rows+r).norm(dim=1))
    return bool((distance>tol).all())


def install(l, idx, Wr, br):
    """scale-match refined unit rows to the guess (as run.py --fast)"""
    Lf = net.layers[l]; gw = Lf.weight[idx]; gb = Lf.bias[idx]
    uw = Wr[idx].double(); ub = br[idx].double()
    proj = (gw * uw).sum(1) + gb * ub; gn = (gw.pow(2).sum(1) + gb.pow(2)).sqrt()
    cs = torch.where(proj > 1e-6 * gn, proj, gn).clamp_min(1e-8)
    Lf.weight[idx] = cs[:, None] * uw; Lf.bias[idx] = cs * ub


def polish(l, chans=None):
    """light kink-point polish of layer l's rows (kink_solve.polish_layer); installs rows that moved < 1e-4,
    returns {unit: moved}"""
    global nq
    import copy, kink_solve
    from verify_layer1 import _Oracle
    orc = _Oracle(copy.deepcopy(T).eval())
    chans = list(range(net.layers[l].weight.shape[0])) if chans is None else chans
    out = kink_solve.polish_layer(orc, net, l, chans, torch.Generator(device=dev).manual_seed(55 + l), max_rows=20000)
    nq += orc.n
    moved = {}
    with torch.no_grad():
        for j, (w, b, _) in out.items():
            v = torch.cat([w.double().flatten(), torch.as_tensor(b, device=dev, dtype=torch.float64).flatten()]); v = v / v.norm()
            cur = torch.cat([net.layers[l].weight[j], net.layers[l].bias[j:j + 1]]); cur = cur / cur.norm()
            if float(v @ cur) < 0:
                v = -v
            dv = float((v - cur).norm())
            if dv < 1e-4:
                moved[j] = dv
                install(l, torch.tensor([j], device=dev), v[None, :-1], v[-1:])
    print(f"[refine] L{l + 1} polish: {len(moved)}/{len(chans)} rows, max move "
          f"{max(moved.values()) if moved else 0:.1e}, {orc.n:,} queries", flush=True)
    return moved


def prefix_probe(l):
    """--prefix-check: design-sample a few units of layer l (0-indexed), return (suspect previous-layer units, queries)."""
    global nq
    from peel.informative_kinks import recover_layer as rl_design
    probe = torch.randperm(dims[l + 1], generator=torch.Generator().manual_seed(31 + l))[:a.prefix_check_probe].tolist()
    cand = {}
    _, _, _, q = rl_design(T, net, l, dev, only_channels=probe, gen=torch.Generator(device=dev).manual_seed(4242 + l),
                           angle_gate=12., verbose=False, candidate_rows=cand)
    nq += q
    votes = torch.zeros(dims[l], dtype=torch.long)
    doms = []
    for c, (w, b) in cand.items():
        w = w.double().cpu(); f = w.abs() / w.norm(); doms.append(float(f.max()))
        votes += (f > a.prefix_check_dom).long()
    need = max(3, (len(cand) + 1) // 2)
    bad = (votes >= need).nonzero().flatten().tolist()
    top = votes.topk(3)
    print(f"[prefix-check] L{l + 1}: probed {len(probe)} units ({len(cand)} fitted, {q:,} q) | median max|w_j|/|w| "
          f"{sorted(doms)[len(doms) // 2] if doms else float('nan'):.3f} | top votes "
          f"{[(int(i), int(v)) for v, i in zip(top.values, top.indices)]} (need {need}) -> suspect L{l} units {bad}", flush=True)
    return bad


def prefix_repair(l, j):
    """re-solve unit j of layer l (0-indexed) guess-free (all other units of layers <= l treated as exact)."""
    global nq
    import guessfree
    known = [i for i in range(dims[l + 1]) if i != j]
    rows_gf, q, info = guessfree.recover_frontier_rows(T, net, l, known, lines=a.gf_lines, device=dev, seed=97 + l, stop_at=1)
    nq += q
    print(f"[prefix-check] L{l + 1} unit {j}: guess-free scan {len(rows_gf)} candidate rows, {q:,} q", flush=True)
    Wl = net.layers[l].weight.detach().clone().double(); bl = net.layers[l].bias.detach().clone().double()
    okl = torch.ones(dims[l + 1], dtype=torch.bool, device=dev); okl[j] = False
    g0 = (net.layers[l].weight[j].clone(), net.layers[l].bias[j].clone())
    mag = float(torch.cat([g0[0], g0[1][None]]).norm())
    for r in rows_gf:
        with torch.no_grad():
            r = r.double().to(dev); net.layers[l].weight[j] = mag * r[:-1]; net.layers[l].bias[j] = mag * r[-1]
        W2, b2, ok2, q = design_refine(l, [j]); nq += q
        if bool(ok2[j]) and distinct_candidate(j, W2[j], b2[j], Wl, bl, okl):
            with torch.no_grad():
                install(l, torch.tensor([j], device=dev), W2.double(), b2.double())
            print(f"[prefix-check] L{l + 1} unit {j}: re-solved from a guess-free row", flush=True)
            return True
    with torch.no_grad():
        net.layers[l].weight[j] = g0[0]; net.layers[l].bias[j] = g0[1]
    print(f"[prefix-check] L{l + 1} unit {j}: NOT repaired (row left as it was)", flush=True)
    return False


def member_rows_netgauge(l, n_pts=8192):
    """every member's layer-l rows written over the REFINED net's layer-l inputs (teacher-free): each member's
    layer-(l-1) unit is matched to the refined unit whose activation it reproduces (max correlation over random
    inputs) and its column rescaled (leaky ReLU: a positively scaled pre-activation gives a scaled activation).
    Aligning members to the original guess member instead scrambles rows when that member has a broken unit."""
    X = torch.randn(n_pts, dims[0], device=dev, dtype=torch.float64,
                    generator=torch.Generator(device=dev).manual_seed(77 + l)) * 0.5
    def h(n):
        H = X
        for i in range(l):
            H = n.act(n.layers[i](H))
        return H
    out = []
    Hn = h(net); Cn = Hn - Hn.mean(0); Cn = Cn / Cn.norm(dim=0).clamp_min(1e-300)
    for k_, m in enumerate(mems):
        m = guess0 if k_ == bi else m
        W = m.layers[l].weight.double(); B = m.layers[l].bias.double()
        if l > 0:
            Hm = h(m); Cm = Hm - Hm.mean(0); Cm = Cm / Cm.norm(dim=0).clamp_min(1e-300)
            kk = (Cm.T @ Cn).argmax(1)
            sc = (Hm * Hn[:, kk]).sum(0) / (Hn[:, kk] ** 2).sum(0).clamp_min(1e-300)
            Wn = torch.zeros(W.shape[0], dims[l], device=dev, dtype=torch.float64)
            Wn.index_add_(1, kk, W * sc[None])
            W = Wn
        Rm = torch.cat([W, B[:, None]], 1); out.append((Rm / Rm.norm(dim=1, keepdim=True), k_))
    return out


def orphans_net(l, ok, cos=0.98):
    """committee rows (refined-net gauge) matching no solved row, clustered across members, most-supported first;
    within equal support, the rows farthest from every solved row first"""
    S = torch.cat([net.layers[l].weight, net.layers[l].bias[:, None]], 1).double()[ok]
    S = S / S.norm(dim=1, keepdim=True)
    pool = []
    for Rm, k_ in member_rows_netgauge(l):
        mc = (Rm @ S.T).abs().amax(1) if len(S) else torch.zeros(len(Rm), device=Rm.device, dtype=Rm.dtype)
        pool += [(Rm[j], k_, float(mc[j])) for j in (mc < cos).nonzero().flatten().tolist()]
    out, used = [], [False] * len(pool)
    for i, (ri, _, _) in enumerate(pool):
        if used[i]:
            continue
        g = [k for k, (rk, _, _) in enumerate(pool) if not used[k] and float(rk @ ri) > cos]
        for k in g:
            used[k] = True
        support = len({pool[k][1] for k in g})
        cen = max(g, key=lambda k: sum(float(pool[k][0] @ pool[q][0]) for q in g))
        out.append((support, pool[cen][2], pool[cen][0]))
    return [(sup, r) for sup, _, r in sorted(out, key=lambda t: (-t[0], t[1]))]


def reanchor(l, n_pts=8192):
    """--reanchor: net.layers[l] <- guess0's layer-l rows written over net's (refined) layer-(l-1) units (no queries)"""
    X = torch.randn(n_pts, dims[0], device=dev, dtype=torch.float64,
                    generator=torch.Generator(device=dev).manual_seed(91 + l)) * 0.5
    def h(n):
        H = X
        for i in range(l):
            H = n.act(n.layers[i](H))
        return H
    Hn, Hg = h(net), h(guess0)
    zs = lambda H: (H - H.mean(0)) / (H - H.mean(0)).norm(dim=0).clamp_min(1e-300)
    C = zs(Hg).T @ zs(Hn)
    corr, kk = C.max(1)
    sc = (Hg * Hn[:, kk]).sum(0) / (Hn[:, kk] ** 2).sum(0).clamp_min(1e-300)
    W = torch.zeros(dims[l + 1], dims[l], device=dev, dtype=torch.float64)
    W.index_add_(1, kk, guess0.layers[l].weight.double() * sc[None])
    # columns the guess member cannot supply: refined units it has no unit for, or two (a duplicate standing in for
    # a missing unit -> its outgoing weights compensate on the copy). Fill them from the other members (each mapped
    # onto the refined units the same way), median over members of each guess row's best-matching member row.
    cnt = torch.bincount(kk, minlength=dims[l])
    S = (cnt != 1).nonzero().flatten()
    if len(S):
        keep = (cnt == 1)
        Gk = W[:, keep] / W[:, keep].norm(dim=1, keepdim=True).clamp_min(1e-300)
        vals, ok_cols = [], []
        for k_, m in enumerate(mems):
            if k_ == bi:
                continue
            Hm = h(m); cm, km = (zs(Hm).T @ zs(Hn)).max(1)
            okc = torch.bincount(km, minlength=dims[l])[S] == 1      # columns of S this member supplies cleanly
            if not bool(okc.any()):
                continue
            sm = (Hm * Hn[:, km]).sum(0) / (Hn[:, km] ** 2).sum(0).clamp_min(1e-300)
            Wm = torch.zeros(m.layers[l].weight.shape[0], dims[l], device=dev, dtype=torch.float64)
            Wm.index_add_(1, km, m.layers[l].weight.double() * sm[None])
            Mk = Wm[:, keep]; c_ = Gk @ (Mk / Mk.norm(dim=1, keepdim=True).clamp_min(1e-300)).T
            jj = c_.abs().argmax(1)
            a_ = (W[:, keep] * Mk[jj]).sum(1) / (Mk[jj] ** 2).sum(1).clamp_min(1e-300)   # scale (and sign) onto the guess row
            v_ = Wm[jj][:, S] * a_[:, None]
            vals.append(torch.where(okc[None, :], v_, torch.full_like(v_, float("nan")))); ok_cols.append(okc)
        if vals:
            med = torch.stack(vals).nanmedian(0).values
            have = torch.stack(ok_cols).any(0)
            W[:, S[have]] = med[:, have]
        print(f"[reanchor] L{l + 1}: {len(S)} prefix units the guess member cannot supply ({int((cnt == 0).sum())} missing, "
              f"{int((cnt > 1).sum())} duplicated) -> columns filled from other members (per column: {[int(c) for c in torch.stack(ok_cols).sum(0)] if vals else []} members)", flush=True)
    old = torch.cat([net.layers[l].weight, net.layers[l].bias[:, None]], 1).double()
    new = torch.cat([W, guess0.layers[l].bias.double()[:, None]], 1)
    cosd = torch.rad2deg(torch.acos(((old * new).sum(1) / (old.norm(dim=1) * new.norm(dim=1))).abs().clamp(max=1.)))
    anchored[l] = (W.clone(), guess0.layers[l].bias.detach().double().clone())
    with torch.no_grad():
        net.layers[l].weight.copy_(W.to(net.layers[l].weight.dtype)); net.layers[l].bias.copy_(guess0.layers[l].bias)
    print(f"[reanchor] L{l + 1}: guess rows rewritten over the refined L{l} units (match corr min {float(corr.min()):.6f}, "
          f"{dims[l] - len(set(kk.tolist()))} refined units unmatched) | rows moved median {float(cosd.median()):.2f} deg, "
          f"max {float(cosd.max()):.2f} deg", flush=True)


def dup_pairs(l, ok, tol=1e-6):
    """solved rows of layer l that coincide (signed cos of unit [w|b] > 1 - tol; no queries)"""
    R = torch.cat([net.layers[l].weight, net.layers[l].bias[:, None]], 1).double()
    R = R / R.norm(dim=1, keepdim=True)
    C = R @ R.T; C.fill_diagonal_(-2.)
    idx = ok.nonzero().flatten()
    Cs = C[idx][:, idx]
    return [(int(idx[i]), int(idx[k])) for i, k in (Cs > 1 - tol).nonzero().tolist() if i < k]


def dup_fix(l, Wr, br, ok):
    """--dup-check: unmark one copy of every duplicated row and re-solve it; updates Wr/br/ok in place"""
    global nq
    for _round in range(2):
        pairs = dup_pairs(l, ok)
        if not pairs:
            return
        GW, GB = anchored.get(l, (guess0.layers[l].weight.double(), guess0.layers[l].bias.double()))
        G = torch.cat([GW, GB[:, None]], 1).double()
        G = G / G.norm(dim=1, keepdim=True)
        R = torch.cat([net.layers[l].weight, net.layers[l].bias[:, None]], 1).double()
        R = R / R.norm(dim=1, keepdim=True)
        stay = (R * G).sum(1)                                   # cos(refined row, its own committee guess)
        drop = sorted({i if float(stay[i]) < float(stay[k]) else k for i, k in pairs})
        print(f"[dup-check] L{l + 1}: duplicate rows {pairs} (cos to own guess "
              f"{[(i, round(float(stay[i]), 4)) for p_ in pairs for i in p_]}) -> re-solving {drop}", flush=True)
        for k in drop:
            ok[k] = False
            with torch.no_grad():
                net.layers[l].weight[k] = GW[k].to(net.layers[l].weight.dtype); net.layers[l].bias[k] = GB[k].to(net.layers[l].bias.dtype)
        for k in drop:
            g0 = (net.layers[l].weight[k].clone(), net.layers[l].bias[k].clone())
            solved = False
            cands_ = orphans_net(l, ok, cos=0.98)
            print(f"[dup-check]   unit {k}: {len(cands_)} committee rows (refined-net gauge), trying {min(len(cands_), a.dup_orphan_tries)}", flush=True)
            for sup, r in cands_[:a.dup_orphan_tries]:
                with torch.no_grad():
                    w_ = r[:-1]; b_ = r[-1]
                    mag = float(torch.cat([g0[0], g0[1][None]]).norm()) / float(torch.cat([w_, b_[None]]).norm())
                    net.layers[l].weight[k] = mag * w_; net.layers[l].bias[k] = mag * b_
                W2, b2, ok2, q = design_refine(l, [k], gate=a.dup_angle_gate); nq += q
                # same tolerance as dup_pairs (cos > 1 - 1e-6 -> distance 1.4e-3): at depth two solves of ONE unit differ
                # by ~1e-4, so the default 1e-5 let a re-solved copy through
                if bool(ok2[k]) and distinct_candidate(k, W2[k], b2[k], Wr, br, ok, tol=2e-3):
                    Wr[k] = W2[k].double(); br[k] = b2[k].double(); ok[k] = True; solved = True
                    with torch.no_grad():
                        install(l, torch.tensor([k], device=dev), Wr, br)
                    print(f"[dup-check]   unit {k}: re-solved from a committee orphan row", flush=True)
                    break
            if not solved:
                with torch.no_grad():
                    net.layers[l].weight[k] = g0[0]; net.layers[l].bias[k] = g0[1]
                if l > 0 and prefix_repair(l, k):
                    v = torch.cat([net.layers[l].weight[k], net.layers[l].bias[k:k + 1]]).double(); v = v / v.norm()
                    Wr[k] = v[:-1]; br[k] = v[-1]; ok[k] = True; solved = True
            if not solved:
                print(f"[dup-check]   unit {k}: unsolved (layer stays incomplete)", flush=True)
    if dup_pairs(l, ok):
        print(f"[dup-check] L{l + 1}: duplicates remain {dup_pairs(l, ok)}", flush=True)


completed_layers=a.start_layer-1
TRK = dict(track_step=a.track_step, track_need=None)
t0 = time.time()
if a.resume:
    _r = torch.load(a.resume, map_location=dev, weights_only=False)
    with torch.no_grad():
        for l in range(a.start_layer - 1):
            net.layers[l].weight.copy_(_r["state_dict"][f"layers.{l}.weight"]); net.layers[l].bias.copy_(_r["state_dict"][f"layers.{l}.bias"])
    nq += _r["queries"]
    e = param_errors(net, T)["max_eps_per_matrix"]
    print(f"[refine] resumed L1-L{a.start_layer - 1} from {a.resume} ({_r['queries']:,} queries) | "
          f"max eps {' '.join(f'{v:.1e}' for v in e[:2 * (a.start_layer - 1)])}", flush=True)
def design_refine(layer, channels, gate=12.):
    if a.design_unc_gate and a.design_from and layer + 1 >= a.design_from and not a.local_refine:
        from peel.informative_kinks import recover_layer as rl_design
        din = net.layers[layer].weight.shape[1]
        W_, b_, m_, q_ = rl_design(T, net, layer, dev, only_channels=channels, gen=torch.Generator(device=dev).manual_seed(1234 + layer),
                                   angle_gate=gate, need=max(400, 2 * din), pool=max(768, 2 * din), verbose=True, unc_gate=a.design_unc_gate)
        _n = (W_.square().sum(1) + b_.square()).sqrt().clamp_min(1e-30)
        W_[m_] = W_[m_] / _n[m_, None]; b_[m_] = b_[m_] / _n[m_]
        return W_, b_, m_, q_
    if not a.local_refine:
        return _mlp_refine_layer(T,net,layer,dev,act,only_channels=channels,angle_gate=gate,design=True)
    from peel.informative_kinks import recover_layer
    return recover_layer(T,net,layer,dev,only_channels=channels,angle_gate=12.,
                         max_rounds=a.fallback_rounds,failure_dir=None)

for l in range(a.start_layer - 1, len(dims) - 2):
    TRK["track_need"] = a.track_need_mult * dims[l]
    tl = time.time()
    if a.reanchor and l >= 1:
        reanchor(l)
    if a.prefix_check and l >= 2:
        for _round in range(2):
            bad = prefix_probe(l)
            if not bad:
                break
            for j in bad:
                prefix_repair(l - 1, j)
            e = param_errors(net, T)["max_eps_per_matrix"]                 # scoring only
            print(f"[prefix-check] after repair: L{l} max eps W {e[2 * (l - 1)]:.2e} b {e[2 * (l - 1) + 1]:.2e} (eval only)", flush=True)
    # layer 1: design sampler (wide input); deeper: cheap surface tracking (~1.3M q/layer), design only for the
    # units tracking leaves unsolved
    use_design = l == 0 or (a.design_from and l + 1 >= a.design_from)
    if use_design and l > 0 and (a.design_need or a.design_unc_gate):
        # same call as method._mlp_refine_layer(design=True) -> kink_solve.recover_layer(sampling="design"), with need set
        from peel.informative_kinks import recover_layer as rl_design
        din = net.layers[l].weight.shape[1]
        print(f"  [design-refine] L{l + 1}: information-directed kink sampling, need {a.design_need or max(400, 2 * din)}, "
              f"uncertainty gate {a.design_unc_gate or 1e-5:g}", flush=True)
        Wr, br, ok, q = rl_design(T, net, l, dev, gen=torch.Generator(device=dev).manual_seed(1234 + l), angle_gate=12.0,
                                  need=a.design_need or max(400, 2 * din), pool=max(768, 2 * din), verbose=True,
                                  **({"unc_gate": a.design_unc_gate} if a.design_unc_gate else {}))
        _n = (Wr.square().sum(1) + br.square()).sqrt().clamp_min(1e-30)
        Wr[ok] = Wr[ok] / _n[ok, None]; br[ok] = br[ok] / _n[ok]
    elif use_design:
        Wr, br, ok, q = _mlp_refine_layer(T, net, l, dev, act, angle_gate=12.0, design=True)
    elif a.local_refine:
        from peel.local_kink_refine import recover_layer
        archive={}
        Wr,br,ok,q=recover_layer(T,net,l,rounds=a.local_rounds,archive=archive)
        if a.out:torch.save(archive,a.out+f".L{l+1}.points.pt")
    else:
        # tracking with spread-out points (step 2, 2*Din points): exact-prefix L3 test 3.5e-10 -> 1.6e-11 max error,
        # and the better-conditioned fit amplifies the prefix's error far less (defaults: L2 3e-9 -> L3 1.5e-6)
        import kink_solve
        Wr, br, ok, q = kink_solve.recover_layer(T, net, l, dev, gen=torch.Generator(device=dev).manual_seed(1234 + l),
                                                 angle_gate=12.0, sampling="track", verbose=False, **TRK)
    nq += q
    Wr, br = Wr.double(), br.double()
    # Later candidate isolation must see the rows we actually recovered.
    with torch.no_grad():install(l,ok.nonzero().flatten(),Wr,br)
    left = (~ok).nonzero().flatten().tolist()
    if left and l > 0 and not use_design:                 # (design already ran on every row; same seed, same result)
        W2, b2, ok2, q = design_refine(l,left); nq += q
        for j in left:
            if bool(ok2[j]) and distinct_candidate(j,W2[j],b2[j],Wr,br,ok):
                Wr[j] = W2[j].double(); br[j] = b2[j].double(); ok[j] = True
        print(f"[refine] L{l + 1}: track left {len(left)}, design fallback solved {sum(bool(ok2[j]) for j in left)}", flush=True)
        left = (~ok).nonzero().flatten().tolist()
    if a.out:
        torch.save(dict(dims=dims,act=act,state_dict=net.state_dict(),queries=nq,
                        frontier=l,refined_weight=Wr,refined_bias=br,refined_mask=ok),
                   a.out + f".L{l+1}.partial.pt")
    if left:
        # cluster at cos 0.98 and keep single-member rows too: a unit only 3 members learned, at cos 0.984-0.997,
        # formed three singleton clusters at 0.99 and was filtered out by a support >= 2 rule (L6, 9-layer run)
        oc = (orphans(l, Wr, br, ok, cos=0.98)[:3 * len(left) + 3] if a.orphan_gauge == "guess"
              else orphans_net(l, ok, cos=0.98)[:max(a.dup_orphan_tries, 3 * len(left) + 3)])
        cands = [r for sup, r in oc]
        print(f"[refine] L{l + 1}: {len(left)} unsolved -> {len(cands)} committee orphan rows, support {[sup for sup, _ in oc]}", flush=True)
        for j in left:
            g0 = (net.layers[l].weight[j].clone(), net.layers[l].bias[j].clone())
            for r in cands:
                with torch.no_grad():
                    w_ = (r[:-1] / col_s[l]) if a.orphan_gauge == "guess" else r[:-1]; b_ = r[-1]
                    mag = float(torch.cat([g0[0], g0[1][None]]).norm()) / float(torch.cat([w_, b_[None]]).norm())
                    net.layers[l].weight[j] = mag * w_; net.layers[l].bias[j] = mag * b_
                W2, b2, ok2, q = design_refine(l,[j]); nq += q
                if bool(ok2[j]) and distinct_candidate(j,W2[j],b2[j],Wr,br,ok):
                    Wr[j] = W2[j].double(); br[j] = b2[j].double(); ok[j] = True
                    print(f"[refine]   unit {j}: solved from a committee orphan row", flush=True)
                    break
            else:
                with torch.no_grad():
                    net.layers[l].weight[j] = g0[0]; net.layers[l].bias[j] = g0[1]
                print(f"[refine]   unit {j}: no committee row works", flush=True)
        left = (~ok).nonzero().flatten().tolist()
    if left and l > 0:
        # guess-free: every other unit of layers <= l is exact, so a black-box kink they do not explain belongs to
        # an unsolved unit (or a deeper layer; the global kink test rejects those).  Its row (prefix-Jacobian
        # mapback, coarse at depth) is the guess for an exact design solve.
        import guessfree
        # Exclusion must use the recovered rows, not stale committee guesses.
        gf_net=copy.deepcopy(net)
        with torch.no_grad():
            idx=ok.nonzero().flatten()
            gf_net.layers[l].weight[idx]=Wr[idx]
            gf_net.layers[l].bias[idx]=br[idx]
        gf_teacher=T
        if a.local_refine and dims[0]>dims[1]:
            from peel.input_subspace import project_first_rowspace
            gf_teacher,gf_net,_=project_first_rowspace(T,gf_net)
            print(f"[refine] guess-free input subspace: {dims[0]} -> {dims[1]} dimensions",flush=True)
        rows_gf, q, info = guessfree.recover_frontier_rows(gf_teacher, gf_net, l, ok.nonzero().flatten().tolist(), lines=a.gf_lines,
                                                           device=dev, seed=11 + l, stop_at=len(left))
        nq += q
        print(f"[refine] L{l + 1}: guess-free scan {len(rows_gf)} candidate rows, {q:,} queries, clusters {info['clusters'][:8]}", flush=True)
        for j in left:
            g0 = (net.layers[l].weight[j].clone(), net.layers[l].bias[j].clone())
            for r in rows_gf:
                with torch.no_grad():
                    r = r.double(); mag = float(torch.cat([g0[0], g0[1][None]]).norm())
                    net.layers[l].weight[j] = mag * r[:-1]; net.layers[l].bias[j] = mag * r[-1]
                W2, b2, ok2, q = design_refine(l,[j]); nq += q
                if bool(ok2[j]) and distinct_candidate(j,W2[j],b2[j],Wr,br,ok):
                    Wr[j] = W2[j].double(); br[j] = b2[j].double(); ok[j] = True
                    print(f"[refine]   unit {j}: solved from a guess-free row", flush=True)
                    break
            else:
                with torch.no_grad():
                    net.layers[l].weight[j] = g0[0]; net.layers[l].bias[j] = g0[1]
                print(f"[refine]   unit {j}: unsolved", flush=True)
    with torch.no_grad():
        install(l, ok.nonzero().flatten(), Wr, br)
    if l > 0 and bool(ok.all()) and not a.local_refine and not use_design:
        # exactness check: the track gate (12 deg) can pass a row 1e-2 off (L3 unit, 9-layer run). Re-solve every
        # row with the light polish (fresh kink points bracketed at ~1e-5 around the row, 8 queries each); a row
        # that polishes to within 1e-4 takes the polished value, one that cannot be polished goes to design.
        import kink_solve
        W3, b3, m3, q = kink_solve.recover_layer(T, net, l, dev, gen=torch.Generator(device=dev).manual_seed(777 + l),
                                                 angle_gate=12.0, sampling="track", verbose=False, **TRK); nq += q
        v1 = torch.cat([Wr, br[:, None]], 1); v1 = v1 / v1.norm(dim=1, keepdim=True)
        v3 = torch.cat([W3.double(), b3.double()[:, None]], 1); v3 = v3 / v3.norm(dim=1, keepdim=True)
        v3 = v3 * torch.sign((v1 * v3).sum(1, keepdim=True))
        gap = (v1 - v3).abs().amax(1)
        # tracking's own noise grows with depth (two-pass median gap: L2 1e-9, L3 5e-7), so flag OUTLIERS only:
        # rows the second pass failed, or whose gap exceeds both 1e-7 and 50x the layer's median gap
        thr = max(1e-7, 50 * float(gap[m3].median())) if bool(m3.any()) else 1e-7
        bad = [j for j in range(ok.numel()) if not bool(m3[j]) or float(gap[j]) > thr]
        print(f"[refine] L{l + 1}: second independent track ({q:,} q): {ok.numel() - len(bad)}/{ok.numel()} rows agree "
              f"to {thr:.0e} (median gap {float(gap[m3].median()):.1e})", flush=True)
        if bad:
            W2, b2, ok2, q = design_refine(l,bad); nq += q
            with torch.no_grad():
                idx = torch.tensor([j for j in bad if bool(ok2[j])], device=dev, dtype=torch.long)
                if len(idx):
                    install(l, idx, W2.double(), b2.double())
            ok[[j for j in bad if not bool(ok2[j])]] = False
            print(f"[refine] L{l + 1}: track disagreement {len(bad)} rows -> design solved {int(len(idx))}", flush=True)
    if a.dup_check:
        dup_fix(l, Wr, br, ok)
    if a.unit_report:                                      # scoring only: worst units of this layer vs the teacher
        t_, r_ = align_clone_to(net, T)
        ew = (t_.layers[l].weight - r_.layers[l].weight).abs().amax(1); eb = (t_.layers[l].bias - r_.layers[l].bias).abs()
        print(f"[refine]   worst units (aligned idx): W {[(int(i), f'{float(ew[i]):.1e}') for i in ew.topk(3).indices]} "
              f"b {[(int(i), f'{float(eb[i]):.1e}') for i in eb.topk(3).indices]}", flush=True)
    e = param_errors(net, T)["max_eps_per_matrix"]
    print(f"[refine] L{l + 1}: {int(ok.sum())}/{ok.numel()} rows | {nq:,} queries so far | {time.time() - tl:.0f}s | "
          f"L{l + 1} max eps W {e[2 * l]:.2e} b {e[2 * l + 1]:.2e}", flush=True)
    if bool(ok.all()):completed_layers=l+1
    if a.out:
        torch.save(dict(dims=dims,act=act,state_dict=net.state_dict(),queries=nq,
                        completed_layers=completed_layers,frontier=l,refined_mask=ok),
                   a.out+f".L{l+1}.pt")
    if a.stop_after and l + 1 >= a.stop_after:
        break
    if not bool(ok.all()):
        print("[refine] incomplete layer: stopping (an unsolved unit poisons every layer above)", flush=True)
        break
if completed_layers == len(dims)-2:
    with torch.no_grad():                                        # output layer: closed-form least squares
        Xh = torch.randn(a.head_points, dims[0], device=dev, dtype=torch.float64); Yh = T(Xh); nq += len(Xh)
        H = Xh
        for lay in net.layers[:-1]:
            H = net.act(lay(H))
        sol = torch.linalg.lstsq(torch.cat([H, torch.ones(len(H), 1, device=dev, dtype=H.dtype)], 1), Yh).solution
        net.layers[-1].weight.copy_(sol[:-1].T); net.layers[-1].bias.copy_(sol[-1])
else:
    print(f"[refine] INCOMPLETE: {completed_layers}/{len(dims)-2} hidden layers accepted; output fit deferred",flush=True)

e = param_errors(net, T)
print(f"[refine] FINAL max_eps {e['max_eps']:.3e} mean {sum(e['mean_eps_per_matrix']) / len(e['mean_eps_per_matrix']):.3e} | "
      f"per-matrix max {' '.join(f'{v:.1e}' for v in e['max_eps_per_matrix'])}", flush=True)
print(f"[refine] {nq:,} black-box queries (refine + head), {time.time() - t0:.0f}s", flush=True)
if a.out:
    torch.save({"dims": dims, "act": act, "state_dict": net.state_dict(), "queries": nq, "completed_layers": completed_layers,
                "complete": completed_layers == len(dims)-2}, a.out)
    print("[refine] saved", a.out, flush=True)
