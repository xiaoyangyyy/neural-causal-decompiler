"""Seeded bivariate SCMs. Labels encode an explicit benchmark convention."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import hashlib
import json
import numpy as np
from .io import save_json, read_json

XY, YX, INDEPENDENT, UNKNOWN = range(4)
LABELS = ["X_to_Y", "Y_to_X", "independent", "undetermined"]
FAMILIES = ("linear_gaussian", "linear_nongaussian", "anm", "multiplicative", "heteroskedastic", "independent")
FUNCTIONS = ("square", "cubic", "sin", "tanh")
SPLITS = ("train", "dev", "extraction", "refinement", "alignment_fit", "alignment_test",
          "test_id", "test_function", "test_noise", "test_scale", "test_intervention")

def mechanism(name, x):
    if name == "linear": return x
    if name == "square": return x * x
    if name == "cubic": return x ** 3 / 3
    if name == "sin": return np.sin(2 * x)
    if name == "tanh": return np.tanh(x)
    if name == "cos": return np.cos(2 * x)
    if name == "piecewise": return np.where(x < 0, .4 * x, 1.8 * x)
    if name == "sigmoid": return 1 / (1 + np.exp(-np.clip(2 * x, -40, 40)))
    raise ValueError(f"Unknown mechanism: {name}")

def noise(rng, name, n):
    if name == "gaussian": return rng.normal(size=n)
    if name == "laplace": return rng.laplace(size=n) / np.sqrt(2)
    if name == "uniform": return rng.uniform(-np.sqrt(3), np.sqrt(3), n)
    if name == "student": return rng.standard_t(5, n) / np.sqrt(5 / 3)
    if name == "mixture": return (rng.normal(size=n) + rng.choice([-2., 2.], n)) / np.sqrt(5)
    raise ValueError(f"Unknown noise: {name}")

@dataclass(frozen=True)
class World:
    seed: int
    split: str
    family: str
    direction: int
    function: str
    cause_noise: str
    effect_noise: str
    coefficient: float
    noise_scale: float
    scale_x: float = 1.
    scale_y: float = 1.
    intervention: bool = False
    n: int = 128

    @property
    def label(self):
        if self.family == "independent": return INDEPENDENT
        if self.family == "linear_gaussian": return UNKNOWN
        return self.direction

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:24]

    def ast(self):
        cause, effect = ("X", "Y") if self.direction == XY else ("Y", "X")
        f = {"op": self.function, "arg": {"var": cause}}
        signal = {"op": "mul", "args": [{"const": self.coefficient}, f]}
        eps = {"op": "mul", "args": [{"const": self.noise_scale}, {"noise": "U_effect"}]}
        if self.family == "independent":
            rhs = {"noise": "U_effect"}
        elif self.family == "multiplicative":
            rhs = {"op": "mul", "args": [
                {"op": "add", "args": [{"const": .5}, {"op": "abs", "arg": signal}]},
                {"op": "exp", "arg": eps}]}
        elif self.family == "heteroskedastic":
            rhs = {"op": "add", "args": [signal, {"op": "mul", "args": [
                {"op": "add", "args": [{"const": .3}, {"op": "abs", "arg": {"var": cause}}]}, eps]}]}
        else:
            rhs = {"op": "add", "args": [signal, eps]}
        return {"internal_variables": {cause: {"noise": "U_cause"}, effect: rhs},
                "observed_scaling": {"X": self.scale_x, "Y": self.scale_y},
                "note": "Equations use internal coordinates before observed scaling."}

    def sample(self, intervention=None):
        intervene = self.intervention if intervention is None else intervention
        rng = np.random.default_rng(self.seed)
        x = noise(rng, self.cause_noise, self.n)
        u = noise(rng, self.effect_noise, self.n)
        if intervene:
            # Stochastic intervention on the root; no incoming edge is cut.
            irng = np.random.default_rng(self.seed ^ 0x6A09E667)
            x = irng.normal(.75, 1.7, self.n) if self.family == "linear_gaussian" else irng.uniform(-2.5, 2.5, self.n)
        signal = self.coefficient * mechanism(self.function, x)
        eps = self.noise_scale * u
        if self.family == "independent": y = u
        elif self.family == "multiplicative": y = (.5 + np.abs(signal)) * np.exp(np.clip(eps, -20, 20))
        elif self.family == "heteroskedastic": y = signal + (.3 + np.abs(x)) * eps
        else: y = signal + eps
        d = np.column_stack((x, y) if self.direction == XY else (y, x))
        d *= [self.scale_x, self.scale_y]
        if not np.isfinite(d).all(): raise ValueError("Non-finite SCM sample")
        return d.astype(np.float32)

    def metadata(self):
        return {**asdict(self), "world_id": self.identity, "label": self.label,
                "label_name": LABELS[self.label], "generating_direction": LABELS[self.direction],
                "ast": self.ast(), "identification_scope": "benchmark_family_assumptions"}

def generate_worlds(split, count, seed=42, n=128):
    if split not in SPLITS: raise ValueError(f"Unknown split {split}")
    if count < 1 or n < 16: raise ValueError("Positive count and n >= 16 required")
    # Stable split namespace, independent of requested sizes/order.
    code = int.from_bytes(hashlib.sha256(split.encode()).digest()[:4], "little")
    rng = np.random.default_rng(np.random.SeedSequence([seed, code]))
    worlds = []
    for i in range(count):
        family = FAMILIES[i % len(FAMILIES)]
        func = "linear" if family.startswith("linear") or family == "independent" else str(rng.choice(FUNCTIONS))
        if split == "test_function" and family in ("anm", "multiplicative", "heteroskedastic"):
            func = str(rng.choice(["cos", "piecewise", "sigmoid"]))
        cause = "gaussian" if family == "linear_gaussian" else str(rng.choice(["gaussian", "laplace", "uniform"]))
        effect = "gaussian" if family == "linear_gaussian" else str(rng.choice(["gaussian", "laplace"]))
        if family == "linear_nongaussian": effect = "laplace"
        # Keep the linear-Gaussian negative control Gaussian in all splits.
        if split == "test_noise" and family != "linear_gaussian":
            cause, effect = "student", "mixture"
        scales = np.exp(rng.uniform(-2.3, 2.3, 2)) if split == "test_scale" else np.ones(2)
        worlds.append(World(seed=int(rng.integers(0, 2**63 - 1)), split=split, family=family,
            direction=int(rng.integers(0, 2)), function=func, cause_noise=cause, effect_noise=effect,
            coefficient=float(rng.choice([-1, 1]) * rng.uniform(.5, 1.8)),
            noise_scale=float(rng.uniform(.2, .9)), scale_x=float(scales[0]), scale_y=float(scales[1]),
            intervention=split == "test_intervention", n=n))
    return worlds

def save_dataset(directory, worlds):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "samples.npz",
        data=np.stack([w.sample() for w in worlds]),
        observational=np.stack([w.sample(False) for w in worlds]),
        interventions=np.stack([w.sample(True) for w in worlds]),
        labels=np.array([w.label for w in worlds], dtype=np.int64))
    save_json(directory / "worlds.json", [w.metadata() for w in worlds])

def load_worlds(directory):
    return [World(**{k: v for k, v in m.items() if k in World.__dataclass_fields__})
            for m in read_json(Path(directory) / "worlds.json")]
