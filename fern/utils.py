from typing import TypeVar

K = TypeVar("K")
V = TypeVar("V")


def invert_dict(d: dict[K, V]) -> dict[V, K]:
    assert len(set(d.keys())) == len(set(d.values())), (
        "Dictionary is expected to have unique values"
    )
    return dict((v, k) for k, v in d.items())
