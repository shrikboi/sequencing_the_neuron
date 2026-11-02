"""--tail-from: committee on the TAIL of an MLP whose first K layers have been recovered (run.py).

Students see h = prefix(x) through the RECOVERED layers 1..K (attacker-known, in the recovery's gauge); the committee
mines its queries in raw x (query_map). The black box is the REAL teacher queried at the raw x that produced each h:
query_map records (hash(h) -> x) for every no-grad call, and TailBlackBox.forward looks each h row up and queries
the full teacher at its x (counted). An h that query_map did not produce raises -- it never falls back to anything.

TailBlackBox is an MLP of the tail dims whose PARAMETERS are the teacher's tail written in the recovery's gauge
(layer K+1's columns permuted/rescaled by activation matching of the two prefixes). They are read ONLY by
evaluation (param_errors, snapshots' teacher_state for diagnostics); forward() never uses them."""
import torch
from nets import MLP


class _Recorder:
    def __init__(self, dim, device, max_rows):
        g = torch.Generator().manual_seed(20261102)
        self.R = (torch.randint(1, 2 ** 62, (dim,), generator=g, dtype=torch.int64) | 1).to(device)
        self.max_rows = max_rows
        self.chunks = []          # list of (keys int64 [n] on device, raw [n, d] on cpu)
        self.persist = []
        self.n_rows = 0

    def key(self, h):
        return (h.detach().float().contiguous().view(torch.int32).long() * self.R[None, :]).sum(1)

    def add(self, h, raw, persist=False):
        item = (self.key(h), raw.detach().float().cpu())
        (self.persist if persist else self.chunks).append(item)
        if not persist:
            self.n_rows += len(raw)
            while self.n_rows > self.max_rows and len(self.chunks) > 1:
                self.n_rows -= len(self.chunks.pop(0)[1])

    def lookup(self, h):
        items = self.persist + self.chunks
        keys = torch.cat([k for k, _ in items])
        raws = torch.cat([r for _, r in items])
        sk, order = keys.sort()
        q = self.key(h)
        pos = torch.searchsorted(sk, q).clamp(max=len(sk) - 1)
        hit = sk[pos] == q
        if not bool(hit.all()):
            raise RuntimeError(f"tail black box: {int((~hit).sum())}/{len(h)} queries were not produced by the query map "
                               "(refusing to answer them: the real black box needs their raw inputs)")
        return raws[order[pos].cpu()].to(h.device)


class TailBlackBox(MLP):
    def __init__(self, full_teacher, rec, tail_dims, act):
        super().__init__(tail_dims, act=act)
        self._ext = [full_teacher, rec]     # plain list: NOT submodules, so clone()/state_dict() carry only the eval tail
        self.n_queries = 0

    def forward(self, h):
        full, rec = self._ext
        x = rec.lookup(h)
        self.n_queries += len(x)
        with torch.no_grad():
            y = torch.cat([full(x[i:i + 65536].to(next(full.parameters()).dtype)) for i in range(0, len(x), 65536)])
        return y.to(h.dtype)


def setup(full_teacher, recovered, K, tail_dims, act, device, q, eval_n=2000, seed=0):
    """-> (query_map, black box, eval points); recovered = MLP with the full dims whose layers < K are the recovery"""
    d0 = recovered.layers[0].weight.shape[1]
    layers = [recovered.layers[i] for i in range(K)]
    W = [L.weight.detach().float().to(device) for L in layers]; B = [L.bias.detach().float().to(device) for L in layers]
    actf = recovered.act
    rec = _Recorder(tail_dims[0], device, max_rows=3 * q + 100000)

    def query_map(raw):
        h = raw
        for w, b in zip(W, B):
            h = actf(torch.nn.functional.linear(h, w, b))
        if not torch.is_grad_enabled():
            rec.add(h, raw)
        return h

    bb = TailBlackBox(full_teacher, rec, tail_dims, act).to(device)
    import atexit
    atexit.register(lambda: print(f"[tail] black-box rows answered by the real teacher: {bb.n_queries:,} "
                                  f"(training queries + {eval_n} eval points per evaluation)", flush=True))
    # evaluation-only parameters: teacher's tail in the recovery's gauge
    with torch.no_grad():
        X = torch.randn(16384, d0, device=device, dtype=torch.float64, generator=torch.Generator(device=device).manual_seed(7)) * 0.5
        Hn, Ht = X, X
        for i in range(K):
            Hn = recovered.act(recovered.layers[i].to(torch.float64)(Hn))
            Ht = full_teacher.act(torch.nn.functional.linear(Ht, full_teacher.layers[i].weight.double(), full_teacher.layers[i].bias.double()))
        zs = lambda H: (H - H.mean(0)) / (H - H.mean(0)).norm(dim=0).clamp_min(1e-300)
        C = zs(Hn).T @ zs(Ht)
        corr, perm = C.max(1)
        s = (Hn * Ht[:, perm]).sum(0) / (Ht[:, perm] ** 2).sum(0).clamp_min(1e-300)
        for j in range(len(tail_dims) - 1):
            Lt = full_teacher.layers[K + j]
            w = Lt.weight.double().clone(); b = Lt.bias.double().clone()
            if j == 0:
                w = w[:, perm] / s[None, :]
            bb.layers[j].weight.copy_(w.to(bb.layers[j].weight.dtype)); bb.layers[j].bias.copy_(b.to(bb.layers[j].bias.dtype))
        print(f"[tail] eval-only gauge map of the recovered prefix to the teacher's: corr min {float(corr.min()):.8f}, "
              f"bijective {len(set(perm.tolist())) == len(perm)}, scale range {float(s.min()):.3g}..{float(s.max()):.3g}", flush=True)
        raw = torch.randn(eval_n, d0, device=device, generator=torch.Generator(device=device).manual_seed(seed + 99)) * 0.5
        h = query_map(raw)
        rec.chunks.pop(); rec.n_rows -= len(raw)
        rec.add(h, raw, persist=True)
    return query_map, bb, h
