"""Independent positive MFPT network with hard Dirichlet boundary.

Predict tau directly for stable zero-rate boundary handling; u=Gamma*tau is
available explicitly. No actor, actor optimizer, or reward is imported.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PoissonMFPTNetwork(nn.Module):
    def __init__(self, feature_dim, hidden=64, time_scale_s=1.0):
        super().__init__()
        if feature_dim <= 0 or hidden <= 0 or not np.isfinite(time_scale_s) or time_scale_s <= 0:
            raise ValueError("invalid model dimensions/time scale")
        self.feature_dim, self.hidden = feature_dim, hidden
        self.register_buffer("time_scale_s", torch.tensor(float(time_scale_s), dtype=torch.float64))
        self.register_buffer("mean", torch.zeros(feature_dim, dtype=torch.float64))
        self.register_buffer("scale", torch.ones(feature_dim, dtype=torch.float64))
        self.layers = nn.Sequential(nn.Linear(feature_dim, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)).double()
        # Full-state linear path helps represent site potentials in stiff dilute
        # diffusion without forcing tiny differences through tanh bottlenecks.
        self.skip = nn.Linear(feature_dim, 1).double()
        nn.init.zeros_(self.skip.weight); nn.init.zeros_(self.skip.bias)

    def forward(self, x, absorbing):
        x = torch.as_tensor(x, dtype=torch.float64, device=self.mean.device)
        absorbing = torch.as_tensor(absorbing, dtype=torch.bool, device=x.device)
        if x.shape[:-1] != absorbing.shape or x.shape[-1] != self.feature_dim:
            raise ValueError("feature/boundary shape mismatch")
        normalized = (x - self.mean) / self.scale
        tau = F.softplus((self.layers(normalized) + self.skip(normalized)).squeeze(-1)) * self.time_scale_s
        return torch.where(absorbing, torch.zeros_like(tau), tau)

    def dimensionless_u(self, x, absorbing, total_rate):
        return self(x, absorbing) * torch.as_tensor(total_rate, dtype=torch.float64, device=self.mean.device)


def backward_residual(model, data):
    tau = model(data["x"], data["absorbing"])
    next_tau = model(data["successor_x"], data["successor_absorbing"])
    residual = (data["rates"] * (tau[:, None] - next_tau)).sum(-1) - 1
    return residual[~data["absorbing"]]


def load_dataset(path):
    with np.load(path, allow_pickle=False) as raw:
        metadata = json.loads(str(raw["metadata"]))
        data = {key: torch.as_tensor(raw[key], dtype=torch.bool if "absorbing" in key else torch.float64)
                for key in ("x", "absorbing", "rates", "successor_x", "successor_absorbing")}
        reference = {key: raw[key].copy() for key in ("reference_mean_s", "reference_se_s", "reference_censored") if key in raw}
    x, rates, sx = data["x"], data["rates"], data["successor_x"]
    if x.ndim != 2 or rates.ndim != 2 or sx.shape != (*rates.shape, x.shape[-1]):
        raise ValueError("dataset shape mismatch")
    if data["absorbing"].shape != x.shape[:1] or data["successor_absorbing"].shape != rates.shape:
        raise ValueError("dataset boundary shape mismatch")
    if not all(bool(torch.isfinite(data[k]).all()) for k in ("x", "rates", "successor_x")) or bool((rates < 0).any()):
        raise ValueError("invalid features/rates")
    interior = ~data["absorbing"]
    if not bool(interior.any()) or bool((rates.sum(-1)[interior] <= 0).any()):
        raise ValueError("dataset needs live nonabsorbing states")
    boundary_present = bool(data["absorbing"].any()) or bool(
        ((rates > 0) & data["successor_absorbing"]).any())
    if not boundary_present:
        raise ValueError("no observed absorbing boundary: an unanchored Poisson fit cannot be certified")
    if metadata.get("data_kind") != "raw_kmc_simulation":
        raise ValueError("calibration accepts traced raw KMC data; ideal/schematic data are excluded")
    return data, metadata, reference


@torch.no_grad()
def validate(model, data, references):
    residual = backward_residual(model, data)
    result = {"residual_rmse": float(residual.square().mean().sqrt()),
              "residual_bias": float(residual.mean()), "mfpt_reference_validated": False,
              "policy_physical_time_validated": False}
    if references:
        predicted = model(data["x"], data["absorbing"]).cpu().numpy()
        interior = (~data["absorbing"]).cpu().numpy()
        mean = np.asarray(references["reference_mean_s"])
        se = np.asarray(references["reference_se_s"])
        censored = np.asarray(references["reference_censored"])
        valid = interior & np.isfinite(mean) & (mean > 0) & np.isfinite(se) & (se >= 0) & (censored == 0)
        complete = bool(valid[interior].all())
        if valid.any():
            error = np.abs(predicted[valid] - mean[valid]) / mean[valid]
            uncertainty = se[valid] / mean[valid]
            result.update(mfpt_relative_error_mean=float(error.mean()),
                          mfpt_relative_error_max=float(error.max()),
                          reference_relative_se_max=float(uncertainty.max()))
            # Conservative preregistered engineering gates, not NMI guarantees.
            result["mfpt_reference_validated"] = bool(
                complete and result["residual_rmse"] <= .05 and np.max(error) <= .10
                and np.max(uncertainty) <= .05)
    return result


def fit(train_path, validation_path, output_path, *, hidden=64, iterations=300, seed=1, device="cpu"):
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    train, train_meta, train_references = load_dataset(train_path)
    validation, val_meta, references = load_dataset(validation_path)
    if train_meta["domain"] != val_meta["domain"]:
        raise ValueError("training and validation calibration domains differ")
    if train_meta["initial_seed"] == val_meta["initial_seed"]:
        raise ValueError("validation must use an independent initial-state/trajectory seed")
    if train["x"].shape[-1] != validation["x"].shape[-1]:
        raise ValueError("feature dimensions differ")
    torch.manual_seed(seed)
    gamma = train["rates"].sum(-1)[~train["absorbing"]]
    time_scale = float(1 / gamma.median())
    if train_references:
        means = train_references["reference_mean_s"]
        se = train_references["reference_se_s"]
        good = (np.isfinite(means) & (means > 0) & np.isfinite(se) & (se >= 0)
                & (train_references["reference_censored"] == 0)
                & (se / np.maximum(means, np.finfo(float).tiny) <= .05))
        if good.any():
            # Fast-flicker residence times can be orders of magnitude below the
            # MFPT. Scaling by reliable TRAINING MFPTs avoids enormous outputs.
            time_scale = float(np.median(means[good]))
    model = PoissonMFPTNetwork(train["x"].shape[-1], hidden, time_scale)
    x = train["x"]
    model.mean.copy_(x.mean(0)); model.scale.copy_(x.std(0, unbiased=False).clamp_min(1))
    model.to(device)
    train = {k: v.to(device) for k, v in train.items()}
    validation = {k: v.to(device) for k, v in validation.items()}
    teacher = None
    teacher_mask = None
    if train_references:
        means = train_references["reference_mean_s"]
        errors = train_references["reference_se_s"]
        censored = train_references["reference_censored"]
        mask = (np.isfinite(means) & (means > 0) & np.isfinite(errors) & (errors >= 0)
                & (censored == 0) & (errors / np.maximum(means, np.finfo(float).tiny) <= .05))
        teacher_mask = torch.as_tensor(mask, dtype=torch.bool, device=device) & ~train["absorbing"]
        if bool(teacher_mask.any()):
            teacher = torch.as_tensor(means, dtype=torch.float64, device=device)[teacher_mask]
            warm_optimizer = torch.optim.LBFGS(model.parameters(), lr=.8, max_iter=iterations,
                tolerance_grad=1e-12, tolerance_change=1e-16, line_search_fn="strong_wolfe")
            def warm_closure():
                warm_optimizer.zero_grad()
                predicted = model(train["x"], train["absorbing"])[teacher_mask]
                loss = (torch.log(predicted) - torch.log(teacher)).square().mean()
                loss.backward()
                return loss
            warm_optimizer.step(warm_closure)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=.8, max_iter=iterations,
                                  tolerance_grad=1e-10, tolerance_change=1e-14,
                                  line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = backward_residual(model, train).square().mean()
        if teacher is not None:
            predicted = model(train["x"], train["absorbing"])[teacher_mask]
            loss = loss + (torch.log(predicted) - torch.log(teacher)).square().mean()
        if not torch.isfinite(loss):
            raise ArithmeticError("nonfinite Poisson optimization loss")
        loss.backward()
        return loss
    optimizer.step(closure)
    model.eval()
    report = validate(model, validation, references)
    report["training_residual_rmse"] = float(backward_residual(model, train).detach().square().mean().sqrt())
    report["independent_mfpt_teacher_used"] = teacher is not None
    report["claim"] = "MFPT potential only; signed progress is not a physical clock"
    hashes = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
              for name, path in (("train", train_path), ("validation", validation_path))}
    payload = {"schema": 1, "kind": "independent_mfpt_potential", "feature_dim": model.feature_dim,
               "hidden": hidden, "time_scale_s": float(model.time_scale_s),
               "state_dict": model.state_dict(), "domain": train_meta["domain"],
               "validation": report, "dataset_sha256": hashes,
               "train_initial_seed": train_meta["initial_seed"], "validation_initial_seed": val_meta["initial_seed"]}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    output_path.with_suffix(".validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return model, report


def load_potential(path, domain, *, require_validated=True):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != 1 or payload.get("kind") != "independent_mfpt_potential" or payload["domain"] != domain:
        raise ValueError("checkpoint kind/domain/target mismatch")
    if require_validated and not payload["validation"].get("mfpt_reference_validated", False):
        raise RuntimeError("held-out passive MFPT validation has not passed")
    model = PoissonMFPTNetwork(payload["feature_dim"], payload["hidden"], payload["time_scale_s"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload["validation"]
