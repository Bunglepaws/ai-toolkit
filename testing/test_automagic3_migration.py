import sys
import torch

sys.path.insert(0, "/mnt/c/Data/git/AIToolkitWSL/ai-toolkit")
from toolkit.optimizers.automagic3 import Automagic3

torch.manual_seed(0)

# --- build a fresh optimizer over dummy LoRA-like params (bf16, like the real run)
p1 = torch.nn.Parameter(torch.randn(8, 4, dtype=torch.bfloat16))
p2 = torch.nn.Parameter(torch.randn(4, dtype=torch.bfloat16))
opt = Automagic3([p1, p2], lr=1e-4, weight_decay=1e-4, fused=False)

# --- fabricate an ORIGINAL-v3 style state dict: per-row lr railed at 1e-2 with a
# frozen row at 1e-8, pol_hist bool history, factored second moment, no dir_ema
old_state = {
    "state": {
        0: {
            "step": 9986,
            "lr": torch.tensor([1e-2, 1e-2, 1e-8, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2]),
            "pol_hist": [torch.zeros(8, 4, dtype=torch.bool) for _ in range(3)],
            "exp_avg_sq_row": torch.rand(8, dtype=torch.bfloat16),
            "exp_avg_sq_col": torch.rand(4, dtype=torch.bfloat16),
        },
        1: {
            "step": 9986,
            "lr": torch.full((4,), 1e-2),
            "pol_hist": [torch.zeros(4, dtype=torch.bool) for _ in range(3)],
            "exp_avg_sq": torch.rand(4, dtype=torch.bfloat16),
        },
    },
    "param_groups": [
        {
            "lr": 1e-4,
            "min_lr": 1e-8,
            "max_lr": 1e-2,
            "lr_bump_rate": 0.1,
            "beta2": 0.999,
            "eps": 1e-30,
            "clip_threshold": 1.0,
            "weight_decay": 1e-4,
            "polarity_history_count": 3,
            "agreement_floor": 0.6875,
            "initial_lr": 1e-4,
            "differentiable": False,
            "params": [0, 1],
        }
    ],
}

opt.load_state_dict(old_state)

for i, p in enumerate([p1, p2]):
    st = opt.state[p]
    assert isinstance(st["lr"], torch.Tensor) and st["lr"].dim() == 0, f"p{i} lr not scalar: {st['lr'].shape}"
    assert st["lr"].dtype == torch.float32
    assert abs(float(st["lr"]) - 1e-4) < 1e-9, f"p{i} lr not reset to group lr: {float(st['lr'])}"
    assert st["dir_ema"].dim() == 0 and float(st["dir_ema"]) == 0.0
    assert st.get("prev_sign") is None
    assert "pol_hist" not in st, "pol_hist not purged"
    print(f"p{i}: lr={float(st['lr']):.1e} dir_ema={float(st['dir_ema'])} OK, second moment kept:",
          "exp_avg_sq_row" in st or "exp_avg_sq" in st)

assert abs(opt._avg_lr - 1e-4) < 1e-10, f"_avg_lr wrong: {opt._avg_lr}"
print(f"_avg_lr = {opt._avg_lr:.1e} OK")

# --- run a few steps to make sure stepping works and lr stays sane
for step in range(5):
    for p in (p1, p2):
        p.grad = torch.randn_like(p)
    opt.step()
lrs = opt.get_learning_rates()
print("after 5 steps, reported lr:", ["%.2e" % l for l in lrs])
assert all(l < 1e-3 for l in lrs), "lr exploded after migration"

# --- new-format roundtrip: save/load keeps scalar lr value
sd = opt.state_dict()
opt2 = Automagic3([p1, p2], lr=1e-4, weight_decay=1e-4, fused=False)
opt2.load_state_dict(sd)
st = opt2.state[p1]
assert st["lr"].dim() == 0 and st["lr"].dtype == torch.float32
# parent load_state_dict casts state to param dtype (bf16) -> ~0.4% precision
rel = abs(float(st["lr"]) - float(opt.state[p1]["lr"])) / float(opt.state[p1]["lr"])
assert rel < 1e-2, f"new-format scalar lr not preserved (rel err {rel})"
print("new-format roundtrip: lr preserved at %.3e" % float(st["lr"]))
print("ALL MIGRATION TESTS PASSED")
