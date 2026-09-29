"""Certified exact realizations of finite deterministic neural transducers."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from typing import Any, Iterable, Sequence

Word = tuple[str, ...]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _same_output(left: Any, right: Any) -> bool:
    return _canonical(left) == _canonical(right)


def decode_word(word: Sequence[str]) -> Word:
    if isinstance(word, (str, bytes)):
        raise ValueError("A word must be an action sequence")
    return tuple(str(action) for action in word)


@dataclass(frozen=True)
class FiniteInterventionalSystem:
    """Closed finite Moore system with resettable internal-state handles."""

    states: tuple[str, ...]
    actions: tuple[str, ...]
    transitions: tuple[tuple[int, ...], ...]
    outputs: tuple[Any, ...]
    embeddings: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        n = len(self.states)
        if n == 0 or not self.actions:
            raise ValueError("A finite system needs states and actions")
        if len(set(self.states)) != n or any(not state for state in self.states):
            raise ValueError("State labels must be unique and non-empty")
        if len(set(self.actions)) != len(self.actions) or any(not action for action in self.actions):
            raise ValueError("Action labels must be unique and non-empty")
        if len(self.transitions) != n or any(len(row) != len(self.actions) for row in self.transitions):
            raise ValueError("Transition table shape must be states by actions")
        if any(not isinstance(t, int) or t < 0 or t >= n for row in self.transitions for t in row):
            raise ValueError("Invalid transition target")
        if len(self.outputs) != n:
            raise ValueError("There must be one output per state")
        for output in self.outputs:
            _canonical(output)
        if self.embeddings:
            if len(self.embeddings) != n or not self.embeddings[0]:
                raise ValueError("Embeddings must have one non-empty vector per state")
            width = len(self.embeddings[0])
            if any(len(row) != width for row in self.embeddings):
                raise ValueError("Embedding rows must have a common width")

    @property
    def state_count(self) -> int:
        return len(self.states)

    def state_index(self, state: str | int) -> int:
        if isinstance(state, int):
            if 0 <= state < self.state_count:
                return state
            raise ValueError(f"Unknown state index {state}")
        try:
            return self.states.index(state)
        except ValueError as exc:
            raise ValueError(f"Unknown state {state!r}") from exc

    def action_index(self, action: str) -> int:
        try:
            return self.actions.index(action)
        except ValueError as exc:
            raise ValueError(f"Unknown action {action!r}") from exc

    def advance(self, state: str | int, word: Sequence[str]) -> int:
        current = self.state_index(state)
        for action in decode_word(word):
            current = self.transitions[current][self.action_index(action)]
        return current

    def response(self, state: str | int, word: Sequence[str]) -> Any:
        return self.outputs[self.advance(state, word)]

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema": "ncd.finite-interventional-system.v1",
            "states": list(self.states),
            "actions": list(self.actions),
            "transitions": [list(row) for row in self.transitions],
            "outputs": list(self.outputs),
        }
        if self.embeddings:
            value["embeddings"] = [list(row) for row in self.embeddings]
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FiniteInterventionalSystem":
        if value.get("schema") != "ncd.finite-interventional-system.v1":
            raise ValueError("Unsupported finite-system schema")
        return cls(
            tuple(str(x) for x in value["states"]),
            tuple(str(x) for x in value["actions"]),
            tuple(tuple(int(x) for x in row) for row in value["transitions"]),
            tuple(value["outputs"]),
            tuple(tuple(float(x) for x in row) for row in value.get("embeddings", ())),
        )

    def identity(self) -> str:
        return sha256(_canonical(self.to_dict()).encode("utf-8")).hexdigest()


class ResponseOracle:
    """Memoized response-only access with auditable query accounting."""

    def __init__(self, system: FiniteInterventionalSystem):
        self._system = system
        self._cache: dict[tuple[int, Word], Any] = {}
        self.calls = 0

    @property
    def states(self) -> tuple[str, ...]:
        return self._system.states

    @property
    def actions(self) -> tuple[str, ...]:
        return self._system.actions

    def query(self, state: str | int, word: Sequence[str]) -> Any:
        state_index = self._system.state_index(state)
        decoded = decode_word(word)
        if any(action not in self.actions for action in decoded):
            raise ValueError("Query contains an inadmissible action")
        key = (state_index, decoded)
        if key not in self._cache:
            self._cache[key] = self._system.response(state_index, decoded)
            self.calls += 1
        return self._cache[key]

    def transcript(self) -> list[dict[str, Any]]:
        return [
            {"state": self.states[state], "word": list(word), "response": response}
            for (state, word), response in sorted(
                self._cache.items(), key=lambda item: (len(item[0][1]), item[0][1], item[0][0])
            )
        ]


def words(actions: Sequence[str], max_length: int) -> Iterable[Word]:
    if max_length < 0:
        raise ValueError("max_length must be non-negative")
    yield ()
    for length in range(1, max_length + 1):
        yield from product(tuple(actions), repeat=length)


def _partition_from_signatures(signatures: Sequence[tuple[str, ...]]) -> list[list[int]]:
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for state, signature in enumerate(signatures):
        groups[signature].append(state)
    return sorted(groups.values(), key=lambda block: block[0])


def _canonicalize_partition(blocks: Iterable[Sequence[int]], state_count: int) -> tuple[list[list[int]], list[int]]:
    normalized = sorted((sorted(set(block)) for block in blocks if block), key=lambda block: block[0])
    if sorted(state for block in normalized for state in block) != list(range(state_count)):
        raise ValueError("Blocks must partition every state exactly once")
    mapping = [-1] * state_count
    for block_index, block in enumerate(normalized):
        for state in block:
            mapping[state] = block_index
    return normalized, mapping


def oracle_partition(system: FiniteInterventionalSystem) -> list[list[int]]:
    """Compute exact behavioral equivalence by Moore partition refinement."""
    output_groups: dict[str, list[int]] = defaultdict(list)
    for state, output in enumerate(system.outputs):
        output_groups[_canonical(output)].append(state)
    blocks, mapping = _canonicalize_partition(output_groups.values(), system.state_count)
    while True:
        groups: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
        for state in range(system.state_count):
            key = (mapping[state], tuple(mapping[target] for target in system.transitions[state]))
            groups[key].append(state)
        new_blocks, new_mapping = _canonicalize_partition(groups.values(), system.state_count)
        if new_mapping == mapping:
            return new_blocks
        blocks, mapping = new_blocks, new_mapping


def quotient_from_partition(
    system: FiniteInterventionalSystem, blocks: Sequence[Sequence[int]]
) -> tuple[FiniteInterventionalSystem, list[int]]:
    normalized, mapping = _canonicalize_partition(blocks, system.state_count)
    transitions: list[tuple[int, ...]] = []
    outputs: list[Any] = []
    for block in normalized:
        representative = block[0]
        output = system.outputs[representative]
        if any(not _same_output(system.outputs[state], output) for state in block):
            raise ValueError("A quotient block merges distinct outputs")
        rows = {tuple(mapping[target] for target in system.transitions[state]) for state in block}
        if len(rows) != 1:
            raise ValueError("A quotient block is not transition-congruent")
        outputs.append(output)
        transitions.append(next(iter(rows)))
    quotient = FiniteInterventionalSystem(
        tuple(f"q{i}" for i in range(len(normalized))),
        system.actions,
        tuple(transitions),
        tuple(outputs),
    )
    return quotient, mapping


def _shortest_witness(system: FiniteInterventionalSystem, left: int, right: int) -> Word:
    queue: deque[tuple[int, int, Word]] = deque([(left, right, ())])
    seen = {(left, right)}
    while queue:
        s, t, word = queue.popleft()
        if not _same_output(system.outputs[s], system.outputs[t]):
            return word
        for action_index, action in enumerate(system.actions):
            pair = (system.transitions[s][action_index], system.transitions[t][action_index])
            if pair not in seen:
                seen.add(pair)
                queue.append((pair[0], pair[1], word + (action,)))
    raise ValueError("Equivalent states do not have a distinguishing witness")


def make_certificate(
    system: FiniteInterventionalSystem,
    quotient: FiniteInterventionalSystem,
    encoding: Sequence[int],
    *,
    construction: str,
) -> dict[str, Any]:
    if len(encoding) != system.state_count:
        raise ValueError("Encoding must cover every concrete state")
    representatives: list[int] = []
    for abstract_state in range(quotient.state_count):
        members = [state for state, target in enumerate(encoding) if target == abstract_state]
        if not members:
            raise ValueError("Every abstract state must be used")
        representatives.append(members[0])
    witnesses = []
    for left in range(len(representatives)):
        for right in range(left + 1, len(representatives)):
            word = _shortest_witness(system, representatives[left], representatives[right])
            witnesses.append({
                "left": system.states[representatives[left]],
                "right": system.states[representatives[right]],
                "word": list(word),
                "left_response": system.response(representatives[left], word),
                "right_response": system.response(representatives[right], word),
            })
    return {
        "schema": "ncd.exact-realization-certificate.v1",
        "scope": {
            "states": "declared complete finite state set",
            "experiments": "all finite action words A*",
            "epsilon": 0,
        },
        "source_identity": system.identity(),
        "construction": construction,
        "candidate": quotient.to_dict(),
        "encoding": {state: int(encoding[i]) for i, state in enumerate(system.states)},
        "upper": {
            "bound": quotient.state_count,
            "proof": "exhaustive output preservation and labelled transition homomorphism",
        },
        "lower": {
            "bound": len(representatives),
            "proof": "pairwise distinguishing response witnesses",
            "representatives": [system.states[state] for state in representatives],
            "witnesses": witnesses,
        },
    }


def oracle_minimize(system: FiniteInterventionalSystem) -> dict[str, Any]:
    blocks = oracle_partition(system)
    quotient, encoding = quotient_from_partition(system, blocks)
    certificate = make_certificate(
        system, quotient, encoding, construction="oracle transition-table partition refinement"
    )
    return {
        "blocks": [[system.states[state] for state in block] for block in blocks],
        "certificate": certificate,
    }


def verify_certificate(system: FiniteInterventionalSystem, certificate: dict[str, Any]) -> dict[str, Any]:
    """Independently verify upper and lower exact-minimality certificates."""
    if certificate.get("schema") != "ncd.exact-realization-certificate.v1":
        raise ValueError("Unsupported certificate schema")
    if certificate.get("source_identity") != system.identity():
        raise ValueError("Certificate source identity mismatch")
    scope = certificate.get("scope", {})
    if scope.get("epsilon") != 0 or scope.get("experiments") != "all finite action words A*":
        raise ValueError("Certificate does not claim exact A* behavior")
    candidate = FiniteInterventionalSystem.from_dict(certificate["candidate"])
    if candidate.actions != system.actions:
        raise ValueError("Candidate changes action labels")
    raw_encoding = certificate.get("encoding", {})
    if set(raw_encoding) != set(system.states):
        raise ValueError("Encoding does not cover exactly the concrete states")
    encoding = [raw_encoding[state] for state in system.states]
    if any(not isinstance(value, int) or value < 0 or value >= candidate.state_count for value in encoding):
        raise ValueError("Encoding contains an invalid abstract state")
    if set(encoding) != set(range(candidate.state_count)):
        raise ValueError("Candidate contains unused abstract states")
    for state in range(system.state_count):
        abstract = encoding[state]
        if not _same_output(system.outputs[state], candidate.outputs[abstract]):
            raise ValueError("Upper certificate fails output preservation")
        for action_index in range(len(system.actions)):
            concrete_target = system.transitions[state][action_index]
            if encoding[concrete_target] != candidate.transitions[abstract][action_index]:
                raise ValueError("Upper certificate fails transition homomorphism")
    if certificate.get("upper", {}).get("bound") != candidate.state_count:
        raise ValueError("Incorrect upper bound")

    lower = certificate.get("lower", {})
    representatives = lower.get("representatives", [])
    if len(set(representatives)) != len(representatives):
        raise ValueError("Lower representatives must be unique")
    rep_indices = [system.state_index(state) for state in representatives]
    expected_pairs = {
        (min(a, b), max(a, b))
        for index, a in enumerate(rep_indices)
        for b in rep_indices[index + 1:]
    }
    seen_pairs: set[tuple[int, int]] = set()
    for witness in lower.get("witnesses", []):
        left = system.state_index(witness["left"])
        right = system.state_index(witness["right"])
        pair = (min(left, right), max(left, right))
        if pair not in expected_pairs or pair in seen_pairs:
            raise ValueError("Lower certificate has an unexpected or duplicate pair")
        word = decode_word(witness["word"])
        left_response = system.response(left, word)
        right_response = system.response(right, word)
        if _same_output(left_response, right_response):
            raise ValueError("Lower witness does not distinguish its states")
        if not _same_output(left_response, witness["left_response"]) or not _same_output(
            right_response, witness["right_response"]
        ):
            raise ValueError("Lower witness stores an incorrect response")
        seen_pairs.add(pair)
    if seen_pairs != expected_pairs:
        raise ValueError("Lower certificate is missing pairwise witnesses")
    if lower.get("bound") != len(representatives):
        raise ValueError("Incorrect lower bound")
    lower_bound = len(representatives)
    upper_bound = candidate.state_count
    if lower_bound > upper_bound:
        raise ValueError("Certificate lower bound exceeds its upper bound")
    return {
        "status": "verified",
        "source_states": system.state_count,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "minimal": lower_bound == upper_bound,
        "witnesses_verified": len(expected_pairs),
        "homomorphism_checks": system.state_count * len(system.actions),
    }


def _signatures(oracle: ResponseOracle, suffixes: Sequence[Word]) -> list[tuple[str, ...]]:
    return [
        tuple(_canonical(oracle.query(state, suffix)) for suffix in suffixes)
        for state in range(len(oracle.states))
    ]


def _pair_status(signatures: Sequence[tuple[str, ...]], certified: bool) -> dict[str, int]:
    separated = merged = unresolved = 0
    for left in range(len(signatures)):
        for right in range(left + 1, len(signatures)):
            if signatures[left] != signatures[right]:
                separated += 1
            elif certified:
                merged += 1
            else:
                unresolved += 1
    return {"separated": separated, "merged_certified": merged, "unresolved": unresolved}


def recover_from_responses(
    oracle: ResponseOracle,
    *,
    query_budget: int | None = None,
    strategy: str = "active",
    seed: int = 0,
) -> dict[str, Any]:
    """Recover classes using only R(s,w), preserving unresolved pairs."""
    import random

    if strategy not in {"active", "random", "passive"}:
        raise ValueError("strategy must be active, random, or passive")
    if query_budget is not None and query_budget < 0:
        raise ValueError("query_budget must be non-negative")
    n = len(oracle.states)

    def can_query(state: int, word: Word) -> bool:
        cached = (state, word) in oracle._cache
        return cached or query_budget is None or oracle.calls < query_budget

    def safe_query(state: int, word: Word) -> tuple[bool, Any]:
        if not can_query(state, word):
            return False, None
        return True, oracle.query(state, word)

    if strategy == "active":
        suffixes: list[Word] = []
        for state in range(n):
            ok, _ = safe_query(state, ())
            if not ok:
                break
        if all((state, ()) in oracle._cache for state in range(n)):
            suffixes.append(())
        certified = False
        candidate = None
        while suffixes:
            signatures = _signatures(oracle, suffixes)
            blocks = _partition_from_signatures(signatures)
            new_suffix = None
            derivative_rows: dict[tuple[int, int], tuple[str, ...]] = {}
            budget_hit = False
            for block_index, block in enumerate(blocks):
                for action_index, action in enumerate(oracle.actions):
                    rows = []
                    for state in block:
                        values = []
                        for suffix in suffixes:
                            ok, value = safe_query(state, (action,) + suffix)
                            if not ok:
                                budget_hit = True
                                break
                            values.append(_canonical(value))
                        if budget_hit:
                            break
                        row = tuple(values)
                        derivative_rows[(state, action_index)] = row
                        rows.append((state, row))
                    if budget_hit:
                        break
                    first_state, first_row = rows[0]
                    mismatch = next(((state, row) for state, row in rows[1:] if row != first_row), None)
                    if mismatch is not None:
                        state, row = mismatch
                        component = next(i for i, (a, b) in enumerate(zip(first_row, row)) if a != b)
                        new_suffix = (action,) + suffixes[component]
                        break
                if budget_hit or new_suffix is not None:
                    break
            if budget_hit:
                break
            if new_suffix is not None:
                if new_suffix not in suffixes:
                    suffixes.append(new_suffix)
                continue

            # Complete state enumeration makes closure checkable from response rows:
            # the true derivative is another declared state and must match a row.
            row_to_block = {signatures[block[0]]: index for index, block in enumerate(blocks)}
            transitions = []
            for block_index, block in enumerate(blocks):
                row = []
                for action_index in range(len(oracle.actions)):
                    target = row_to_block.get(derivative_rows[(block[0], action_index)])
                    if target is None:
                        raise ValueError("Declared state handles are not closed under actions")
                    row.append(target)
                transitions.append(tuple(row))
            outputs = tuple(oracle.query(block[0], ()) for block in blocks)
            quotient = FiniteInterventionalSystem(
                tuple(f"q{i}" for i in range(len(blocks))),
                oracle.actions,
                tuple(transitions),
                outputs,
            )
            encoding = [0] * n
            for block_index, block in enumerate(blocks):
                for state in block:
                    encoding[state] = block_index
            candidate = {
                "system": quotient.to_dict(),
                "encoding": {oracle.states[i]: encoding[i] for i in range(n)},
            }
            certified = True
            break
    else:
        rng = random.Random(seed)
        candidates = list(words(oracle.actions, max(0, n - 1)))
        if strategy == "random":
            tail = candidates[1:]
            rng.shuffle(tail)
            candidates = [()] + tail
        suffixes = []
        for word in candidates:
            missing = sum((state, word) not in oracle._cache for state in range(n))
            if query_budget is not None and oracle.calls + missing > query_budget:
                break
            for state in range(n):
                oracle.query(state, word)
            suffixes.append(word)
        signatures = _signatures(oracle, suffixes) if suffixes else [tuple() for _ in range(n)]
        blocks = _partition_from_signatures(signatures)
        certified = len(suffixes) == len(candidates)
        candidate = None

    signatures = _signatures(oracle, suffixes) if suffixes else [tuple() for _ in range(n)]
    blocks = _partition_from_signatures(signatures)
    return {
        "schema": "ncd.response-recovery.v1",
        "strategy": strategy,
        "state_bound": n,
        "assumptions": ["deterministic Moore responses", "complete action-closed state handle list"],
        "certified_complete": certified,
        "suffixes": [list(word) for word in suffixes],
        "blocks": [[oracle.states[state] for state in block] for block in blocks],
        "candidate": candidate,
        "queries": oracle.calls,
        "pair_status": _pair_status(signatures, certified),
        "transcript": oracle.transcript(),
    }


def certificate_from_recovery(source_identity: str, recovery: dict[str, Any]) -> dict[str, Any]:
    """Assemble a certificate from a response-only recovery transcript."""
    if not recovery.get("certified_complete") or recovery.get("candidate") is None:
        raise ValueError("An incomplete recovery cannot produce an exact certificate")
    candidate = recovery["candidate"]
    system_value = candidate["system"]
    encoding = candidate["encoding"]
    abstract_count = len(system_value["states"])
    representatives = []
    for abstract in range(abstract_count):
        members = sorted(state for state, target in encoding.items() if target == abstract)
        if not members:
            raise ValueError("Recovered candidate has an unused state")
        representatives.append(members[0])
    responses = {
        (entry["state"], tuple(entry["word"])): entry["response"]
        for entry in recovery["transcript"]
    }
    suffixes = [tuple(word) for word in recovery["suffixes"]]
    witnesses = []
    for index, left in enumerate(representatives):
        for right in representatives[index + 1:]:
            word = next(
                (
                    suffix
                    for suffix in suffixes
                    if (left, suffix) in responses
                    and (right, suffix) in responses
                    and not _same_output(responses[(left, suffix)], responses[(right, suffix)])
                ),
                None,
            )
            if word is None:
                raise ValueError("Recovered abstract states lack a response witness")
            witnesses.append({
                "left": left,
                "right": right,
                "word": list(word),
                "left_response": responses[(left, word)],
                "right_response": responses[(right, word)],
            })
    return {
        "schema": "ncd.exact-realization-certificate.v1",
        "scope": {
            "states": "declared complete finite state set",
            "experiments": "all finite action words A*",
            "epsilon": 0,
        },
        "source_identity": source_identity,
        "construction": "response-only closed and consistent observation table",
        "candidate": system_value,
        "encoding": encoding,
        "upper": {
            "bound": abstract_count,
            "proof": "candidate submitted for exhaustive output and labelled-transition checking",
        },
        "lower": {
            "bound": len(representatives),
            "proof": "pairwise response witnesses from the query transcript",
            "representatives": representatives,
            "witnesses": witnesses,
        },
    }


def partition_relation(blocks: Sequence[Sequence[str]]) -> set[tuple[str, str]]:
    return {(left, right) for block in blocks for left in block for right in block}


def compare_partition(reference: Sequence[Sequence[str]], candidate: Sequence[Sequence[str]]) -> dict[str, Any]:
    truth = partition_relation(reference)
    predicted = partition_relation(candidate)
    labels = sorted({state for block in reference for state in block})
    pairs = [(labels[i], labels[j]) for i in range(len(labels)) for j in range(i + 1, len(labels))]
    return {
        "exact": predicted == truth,
        "false_merges": sum((a, b) in predicted and (a, b) not in truth for a, b in pairs),
        "false_splits": sum((a, b) not in predicted and (a, b) in truth for a, b in pairs),
        "classes": len(candidate),
    }
