"""Typed scalar-expression / four-class decision-tree DSL. No eval or pickle."""
from dataclasses import dataclass
import numpy as np
from .statistics import FEATURES

@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple = ()
    feature: str = ""

    def __post_init__(self):
        arity = {"feature": 0, "sub": 2, "add": 2, "mul": 2, "abs": 1}
        if self.op not in arity or len(self.args) != arity[self.op]:
            raise ValueError("Invalid expression")
        if self.op == "feature" and self.feature not in FEATURES:
            raise ValueError("Unknown feature")
        if any(not isinstance(a, Expr) for a in self.args): raise ValueError("Scalar operands required")

    @property
    def complexity(self): return 1 + sum(a.complexity for a in self.args)

    def evaluate(self, features):
        if self.op == "feature": return features[:, FEATURES.index(self.feature)]
        a = [x.evaluate(features) for x in self.args]
        if self.op == "abs": return np.abs(a[0])
        if self.op == "sub": return a[0] - a[1]
        if self.op == "add": return a[0] + a[1]
        return a[0] * a[1]

    def to_dict(self):
        if self.op == "feature": return {"op": self.op, "feature": self.feature}
        return {"op": self.op, "args": [a.to_dict() for a in self.args]}

    @classmethod
    def from_dict(cls, d, depth=0):
        if depth > 16: raise ValueError("Expression too deep")
        return cls(d["op"], tuple(cls.from_dict(a, depth+1) for a in d.get("args", [])), d.get("feature", ""))

    def __str__(self):
        if self.op == "feature": return self.feature
        if self.op == "abs": return f"abs({self.args[0]})"
        return f"({self.args[0]} {dict(sub='-', add='+', mul='*')[self.op]} {self.args[1]})"

@dataclass(frozen=True)
class Program:
    label: int | None = None
    expr: Expr | None = None
    threshold: float = 0.
    left: "Program | None" = None
    right: "Program | None" = None

    def __post_init__(self):
        if self.label is not None:
            if self.label not in range(4) or any(x is not None for x in (self.expr,self.left,self.right)):
                raise ValueError("Invalid leaf")
        elif self.expr is None or self.left is None or self.right is None or not np.isfinite(self.threshold):
            raise ValueError("Invalid branch")

    @property
    def complexity(self):
        return 1 if self.label is not None else 2 + self.expr.complexity + self.left.complexity + self.right.complexity

    def predict(self, features):
        if self.label is not None: return np.full(len(features), self.label, dtype=np.int64)
        return np.where(self.expr.evaluate(features) < self.threshold, self.left.predict(features), self.right.predict(features))

    def leaves(self, path=()):
        if self.label is not None: return [(path, self)]
        return self.left.leaves(path+(0,)) + self.right.leaves(path+(1,))

    def mask(self, features, path):
        mask = np.ones(len(features), dtype=bool)
        node = self
        for side in path:
            decision = node.expr.evaluate(features) < node.threshold
            mask &= decision if side == 0 else ~decision
            node = node.left if side == 0 else node.right
        return mask

    def replace(self, path, subtree):
        if not path: return subtree
        if path[0] == 0: return Program(expr=self.expr, threshold=self.threshold, left=self.left.replace(path[1:], subtree), right=self.right)
        return Program(expr=self.expr, threshold=self.threshold, left=self.left, right=self.right.replace(path[1:], subtree))

    def to_dict(self):
        if self.label is not None: return {"label": self.label}
        return {"expr": self.expr.to_dict(), "threshold": self.threshold,
                "left": self.left.to_dict(), "right": self.right.to_dict()}

    @classmethod
    def from_dict(cls, d, depth=0):
        if depth > 16: raise ValueError("Program too deep")
        if "label" in d: return cls(label=int(d["label"]))
        return cls(expr=Expr.from_dict(d["expr"]), threshold=float(d["threshold"]),
                   left=cls.from_dict(d["left"], depth+1), right=cls.from_dict(d["right"], depth+1))

    def text(self, indent=0):
        prefix = "    " * indent
        if self.label is not None:
            from .worlds import LABELS
            return prefix + "return " + LABELS[self.label]
        return (prefix + f"if {self.expr} < {self.threshold:.6g}:\n" + self.left.text(indent+1)
                + "\n" + prefix + "else:\n" + self.right.text(indent+1))

    def used_features(self):
        def names(e):
            return {e.feature} if e.op == "feature" else set().union(*(names(a) for a in e.args))
        if self.label is not None: return set()
        return names(self.expr) | self.left.used_features() | self.right.used_features()
