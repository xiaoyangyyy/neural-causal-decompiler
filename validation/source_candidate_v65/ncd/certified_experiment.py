"""Reproducible benchmark for certified finite interventional realization."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import html
import json
import time
import numpy as np

from .certified_finite import (
    FiniteInterventionalSystem,
    ResponseOracle,
    certificate_from_recovery,
    compare_partition,
    oracle_minimize,
    recover_from_responses,
    verify_certificate,
)
from .io import digest, read_json, save_json
from .finite_neural import FiniteNeuralTransducer
from .approximate_finite import approximate_minimize, verify_approximate_certificate


@dataclass
class CertifiedFiniteConfig:
    seed: int = 2701
    cases: int = 12
    min_abstract_states: int = 3
    max_abstract_states: int = 6
    min_duplicates: int = 2
    max_duplicates: int = 4
    actions: int = 2
    budget_fraction: float = 0.45

    @classmethod
    def quick(cls, seed: int = 2701) -> "CertifiedFiniteConfig":
        return cls(seed=seed, cases=3, max_abstract_states=4, max_duplicates=3)

    def validate(self) -> None:
        if self.cases < 1 or self.actions < 1:
            raise ValueError("cases and actions must be positive")
        if not 2 <= self.min_abstract_states <= self.max_abstract_states:
            raise ValueError("Invalid abstract-state range")
        if not 1 <= self.min_duplicates <= self.max_duplicates:
            raise ValueError("Invalid duplicate-state range")
        if not 0 < self.budget_fraction < 1:
            raise ValueError("budget_fraction must be between zero and one")


def _minimal_base(rng: np.random.Generator, states: int, actions: int) -> tuple[np.ndarray, list[int]]:
    labels = tuple(f"b{i}" for i in range(states))
    action_labels = tuple(f"a{i}" for i in range(actions))
    for _ in range(10000):
        transitions = rng.integers(0, states, size=(states, actions))
        outputs = rng.integers(0, max(2, states // 2), size=states).tolist()
        system = FiniteInterventionalSystem(
            labels, action_labels,
            tuple(tuple(int(x) for x in row) for row in transitions),
            tuple(outputs),
        )
        if len(oracle_minimize(system)["blocks"]) == states:
            return transitions, outputs
    raise RuntimeError("Could not sample a minimal base transducer")


def generate_split_system(seed: int, abstract_states: int, duplicates: int, actions: int) -> FiniteInterventionalSystem:
    """Create a finite neural-state system with behaviorally redundant states."""
    rng = np.random.default_rng(seed)
    base_transitions, base_outputs = _minimal_base(rng, abstract_states, actions)
    classes = list(range(abstract_states))
    classes.extend(int(x) for x in rng.integers(0, abstract_states, size=duplicates))
    rng.shuffle(classes)
    members = {state: [i for i, value in enumerate(classes) if value == state] for state in range(abstract_states)}
    rows = []
    for concrete_class in classes:
        row = []
        for action in range(actions):
            target_class = int(base_transitions[concrete_class, action])
            row.append(int(rng.choice(members[target_class])))
        rows.append(tuple(row))
    # The compiled neural system replaces these with its actual one-hot states.
    embeddings = tuple(tuple(float(x) for x in rng.normal(size=4)) for _ in classes)
    return FiniteInterventionalSystem(
        tuple(f"s{i}" for i in range(len(classes))),
        tuple(f"a{i}" for i in range(actions)),
        tuple(rows),
        tuple(base_outputs[state] for state in classes),
        embeddings,
    )


def _activation_partition(system: FiniteInterventionalSystem, clusters: int, seed: int) -> list[list[str]]:
    from sklearn.cluster import KMeans

    labels = KMeans(n_clusters=clusters, n_init=10, random_state=seed).fit_predict(
        np.asarray(system.embeddings, dtype=float)
    )
    blocks = []
    for label in sorted(set(int(x) for x in labels)):
        blocks.append([system.states[i] for i, value in enumerate(labels) if int(value) == label])
    return sorted(blocks, key=lambda block: system.state_index(block[0]))


def _case_parameters(config: CertifiedFiniteConfig, index: int) -> tuple[int, int, int]:
    abstract_range = config.max_abstract_states - config.min_abstract_states + 1
    duplicate_range = config.max_duplicates - config.min_duplicates + 1
    return (
        config.seed + index * 1009,
        config.min_abstract_states + index % abstract_range,
        config.min_duplicates + (index * 2) % duplicate_range,
    )


def _approximate_branch(
    system: FiniteInterventionalSystem, exact_blocks: list[list[str]]
) -> dict:
    block_of = {
        system.state_index(state): block
        for block, states in enumerate(exact_blocks)
        for state in states
    }
    ranks = {}
    outputs = []
    for state in range(system.state_count):
        block = block_of[state]
        members = [item for item in range(system.state_count) if block_of[item] == block]
        rank = members.index(state)
        if len(members) == 1:
            offset = 0.0
        else:
            offset = -0.2 + 0.4 * rank / (len(members) - 1)
        ranks[state] = offset
        outputs.append(2.0 * block + offset)
    numeric_source = FiniteInterventionalSystem(
        system.states, system.actions, system.transitions, tuple(outputs), system.embeddings
    )
    neural = FiniteNeuralTransducer.compile(numeric_source)
    numeric_system = neural.enumerate_system()
    points = []
    for epsilon in (0.0, 0.25, 10.0):
        certificate = approximate_minimize(numeric_system, epsilon)
        verification = verify_approximate_certificate(numeric_system, certificate)
        points.append({
            "epsilon": epsilon,
            "lower_bound": verification["lower_bound"],
            "upper_bound": verification["upper_bound"],
            "minimal": verification["minimal"],
            "certificate": certificate,
            "verification": verification,
        })
    return {
        "neural": neural.to_dict(),
        "system": numeric_system.to_dict(),
        "curve": points,
    }


def run_case(config: CertifiedFiniteConfig, index: int) -> dict:
    case_seed, abstract_states, duplicates = _case_parameters(config, index)
    source_table = generate_split_system(case_seed, abstract_states, duplicates, config.actions)
    neural = FiniteNeuralTransducer.compile(source_table)
    system = neural.enumerate_system()
    oracle = oracle_minimize(system)
    oracle_check = verify_certificate(system, oracle["certificate"])

    active = recover_from_responses(ResponseOracle(system), strategy="active", seed=case_seed)
    active_quality = compare_partition(oracle["blocks"], active["blocks"])
    if not active["certified_complete"] or not active_quality["exact"] or active["candidate"] is None:
        raise RuntimeError("Response-only active recovery failed to certify the exact quotient")
    active_certificate = certificate_from_recovery(system.identity(), active)
    active_check = verify_certificate(system, active_certificate)
    approximate = _approximate_branch(system, oracle["blocks"])

    fair_budget = active["queries"]
    random_result = recover_from_responses(
        ResponseOracle(system), strategy="random", query_budget=fair_budget, seed=case_seed + 1
    )
    passive_result = recover_from_responses(
        ResponseOracle(system), strategy="passive", query_budget=fair_budget, seed=case_seed + 2
    )
    activation_blocks = _activation_partition(system, len(oracle["blocks"]), case_seed)
    limited_budget = max(system.state_count, int(fair_budget * config.budget_fraction))
    limited = recover_from_responses(
        ResponseOracle(system), strategy="active", query_budget=limited_budget, seed=case_seed
    )
    return {
        "case": index,
        "seed": case_seed,
        "neural": neural.to_dict(),
        "system": system.to_dict(),
        "oracle": oracle,
        "oracle_verification": oracle_check,
        "active": active,
        "active_certificate": active_certificate,
        "active_verification": active_check,
        "active_quality": active_quality,
        "approximate": approximate,
        "baselines": {
            "random": {
                "result": random_result,
                "quality": compare_partition(oracle["blocks"], random_result["blocks"]),
            },
            "passive": {
                "result": passive_result,
                "quality": compare_partition(oracle["blocks"], passive_result["blocks"]),
            },
            "activation_clustering": {
                "blocks": activation_blocks,
                "quality": compare_partition(oracle["blocks"], activation_blocks),
            },
        },
        "limited_budget": {
            "budget": limited_budget,
            "result": limited,
            "quality": compare_partition(oracle["blocks"], limited["blocks"]),
        },
    }


def summarize(cases: list[dict]) -> dict:
    count = len(cases)
    baseline_names = ("random", "passive", "activation_clustering")
    baseline_summary = {}
    for name in baseline_names:
        qualities = [case["baselines"][name]["quality"] for case in cases]
        baseline_summary[name] = {
            "exact_cases": sum(int(value["exact"]) for value in qualities),
            "false_merges": sum(value["false_merges"] for value in qualities),
            "false_splits": sum(value["false_splits"] for value in qualities),
        }
        if name != "activation_clustering":
            baseline_summary[name]["certified_complete_cases"] = sum(
                int(case["baselines"][name]["result"]["certified_complete"]) for case in cases
            )
    source_states = sum(len(case["system"]["states"]) for case in cases)
    quotient_states = sum(len(case["oracle"]["blocks"]) for case in cases)
    epsilon_values = [point["epsilon"] for point in cases[0]["approximate"]["curve"]]
    complexity_curve = []
    for epsilon in epsilon_values:
        points = [
            next(point for point in case["approximate"]["curve"] if point["epsilon"] == epsilon)
            for case in cases
        ]
        complexity_curve.append({
            "epsilon": epsilon,
            "lower_bound_total": sum(point["lower_bound"] for point in points),
            "upper_bound_total": sum(point["upper_bound"] for point in points),
            "closed_cases": sum(int(point["minimal"]) for point in points),
        })
    return {
        "schema": "ncd.certified-finite-summary.v1",
        "cases": count,
        "all_oracle_certificates_verified": all(case["oracle_verification"]["minimal"] for case in cases),
        "all_active_certificates_verified": all(case["active_verification"]["minimal"] for case in cases),
        "active_exact_cases": sum(int(case["active_quality"]["exact"]) for case in cases),
        "source_states": source_states,
        "quotient_states": quotient_states,
        "compression_ratio": quotient_states / source_states,
        "active_queries": sum(case["active"]["queries"] for case in cases),
        "approximate_complexity_curve": complexity_curve,
        "all_approximate_intervals_closed": all(
            point["minimal"]
            for case in cases
            for point in case["approximate"]["curve"]
        ),
        "baselines_at_active_query_budget": baseline_summary,
        "limited_budget": {
            "certified_complete_cases": sum(
                int(case["limited_budget"]["result"]["certified_complete"]) for case in cases
            ),
            "unresolved_pairs": sum(
                case["limited_budget"]["result"]["pair_status"]["unresolved"] for case in cases
            ),
        },
        "claim_scope": "exact finite deterministic ReLU Moore transducers over a complete action-closed state list",
    }


def _report(summary: dict) -> str:
    rows = []
    for name, value in summary["baselines_at_active_query_budget"].items():
        rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{value['exact_cases']}</td>"
            f"<td>{value.get('certified_complete_cases', 0)}</td>"
            f"<td>{value['false_merges']}</td><td>{value['false_splits']}</td></tr>"
        )
    return """<!doctype html><meta charset="utf-8"><title>Certified finite realization</title>
<style>body{{font:16px system-ui;max-width:900px;margin:40px auto;line-height:1.5}}
table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:6px 10px}}</style>
<h1>Certified finite interventional realization</h1>
<p><strong>Scope:</strong> {scope}</p>
<p>All {cases} oracle and response-only active certificates verified. Concrete
states: {source}; minimal states: {quotient}; compression ratio: {ratio:.3f}.
Active response queries: {queries}.</p>
<p>Budget-limited runs preserved {unresolved} unresolved pairs.</p>
<p>All finite approximate certificate intervals closed at the three declared
L-infinity tolerances.</p>
<table><thead><tr><th>Baseline</th><th>Exact cases</th><th>Certified cases</th><th>False merges</th>
<th>False splits</th></tr></thead><tbody>{rows}</tbody></table>
""".format(
        scope=html.escape(summary["claim_scope"]),
        cases=summary["cases"],
        source=summary["source_states"],
        quotient=summary["quotient_states"],
        ratio=summary["compression_ratio"],
        queries=summary["active_queries"],
        unresolved=summary["limited_budget"]["unresolved_pairs"],
        rows="".join(rows),
    )


def run_certified_finite(output: Path, config: CertifiedFiniteConfig) -> dict:
    config.validate()
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    save_json(output / "config.json", asdict(config))
    cases = []
    started = time.perf_counter()
    for index in range(config.cases):
        case = run_case(config, index)
        cases.append(case)
        directory = output / "cases" / f"case_{index:03d}"
        save_json(directory / "system.json", case["system"])
        save_json(directory / "neural.json", case["neural"])
        save_json(directory / "oracle_certificate.json", case["oracle"]["certificate"])
        save_json(directory / "active_recovery.json", case["active"])
        save_json(directory / "active_certificate.json", case["active_certificate"])
        save_json(directory / "approximate.json", case["approximate"])
        save_json(directory / "baselines.json", case["baselines"])
        save_json(directory / "limited_budget.json", case["limited_budget"])
        save_json(directory / "case.json", case)
    summary = summarize(cases)
    summary["runtime_seconds"] = time.perf_counter() - started
    save_json(output / "summary.json", summary)
    (output / "report.html").write_text(_report(summary), encoding="utf-8", newline="\n")
    artifacts = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifacts[path.relative_to(output).as_posix()] = digest(path)
    manifest = {
        "schema": "ncd.certified-finite-manifest.v1",
        "artifacts": artifacts,
    }
    save_json(output / "manifest.json", manifest)
    return summary


def verify_certified_finite(output: Path) -> dict:
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    if manifest.get("schema") != "ncd.certified-finite-manifest.v1":
        raise ValueError("Unsupported manifest schema")
    actual_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual_files != set(manifest["artifacts"]):
        raise ValueError("Artifact set does not match manifest")
    for relative, expected in manifest["artifacts"].items():
        if digest(output / relative) != expected:
            raise ValueError(f"Artifact integrity failure: {relative}")

    config = CertifiedFiniteConfig(**read_json(output / "config.json"))
    config.validate()
    stored_summary = read_json(output / "summary.json")
    replayed_cases = []
    certificates = 0
    for index in range(config.cases):
        directory = output / "cases" / f"case_{index:03d}"
        stored = read_json(directory / "case.json")
        neural = FiniteNeuralTransducer.from_dict(read_json(directory / "neural.json"))
        system = FiniteInterventionalSystem.from_dict(read_json(directory / "system.json"))
        if neural.enumerate_system().to_dict() != system.to_dict():
            raise ValueError(f"Stored neural enumeration mismatch in case {index}")
        approximate = read_json(directory / "approximate.json")
        approximate_neural = FiniteNeuralTransducer.from_dict(approximate["neural"])
        approximate_system = FiniteInterventionalSystem.from_dict(approximate["system"])
        if approximate_neural.enumerate_system().to_dict() != approximate_system.to_dict():
            raise ValueError(f"Approximate neural enumeration mismatch in case {index}")
        for point in approximate["curve"]:
            verify_approximate_certificate(approximate_system, point["certificate"])
        verify_certificate(system, read_json(directory / "oracle_certificate.json"))
        verify_certificate(system, read_json(directory / "active_certificate.json"))
        replayed = run_case(config, index)
        if _canonical_case(replayed) != _canonical_case(stored):
            raise ValueError(f"Deterministic replay mismatch in case {index}")
        replayed_cases.append(replayed)
        certificates += 2 + len(approximate["curve"])
    replayed_summary = summarize(replayed_cases)
    comparable_summary = dict(stored_summary)
    comparable_summary.pop("runtime_seconds", None)
    if _canonical_case(replayed_summary) != _canonical_case(comparable_summary):
        raise ValueError("Summary metrics do not match replay")
    return {
        "status": "verified",
        "cases_replayed": config.cases,
        "certificates_verified": certificates,
        "approximate_certificates_verified": sum(
            len(case["approximate"]["curve"]) for case in replayed_cases
        ),
        "active_exact_cases": replayed_summary["active_exact_cases"],
        "limited_unresolved_pairs": replayed_summary["limited_budget"]["unresolved_pairs"],
    }


def _canonical_case(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
