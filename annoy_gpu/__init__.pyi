
from typing import Sized, overload
from typing_extensions import Literal, Protocol

class _Vector(Protocol, Sized):
    def __getitem__(self, i: int) -> float: ...

class AnnoyIndex:
    f: int
    def __init__(self, f: int, metric: Literal["angular", "euclidean", "manhattan", "hamming", "dot"]) -> None: ...

    def set_print_redirection(self, fd) -> None: ...

    def save_items(self) -> None: ...


    def fill_items(self, fn: str) -> None: ...

    def load_items(self, fn: str) -> Literal[True]: ...


    def load(self, fn: str, prefault: bool = ...) -> Literal[True]: ...
    def save(self, fn: str, prefault: bool = ...) -> Literal[True]: ...
    @overload
    def get_nns_by_item(self, i: int, n: int, search_k: int = ..., include_distances: Literal[False] = ...) -> list[int]: ...
    @overload
    def get_nns_by_item(
        self, i: int, n: int, search_k: int, include_distances: Literal[True]
    ) -> tuple[list[int], list[float]]: ...
    @overload
    def get_nns_by_item(
        self, i: int, n: int, search_k: int = ..., *, include_distances: Literal[True]
    ) -> tuple[list[int], list[float]]: ...
    @overload
    def get_nns_by_vector(
        self, vector: _Vector, n: int, search_k: int = ..., include_distances: Literal[False] = ...
    ) -> list[int]: ...
    @overload
    def get_nns_by_vector(
        self, vector: _Vector, n: int, search_k: int, include_distances: Literal[True]
    ) -> tuple[list[int], list[float]]: ...
    @overload
    def get_nns_by_vector(
        self, vector: _Vector, n: int, search_k: int = ..., *, include_distances: Literal[True]
    ) -> tuple[list[int], list[float]]: ...
    def get_item_vector(self, __i: int) -> list[float]: ...
    def add_item(self, i: int, vector: _Vector) -> None: ...
    def on_disk_build(self, fn: str) -> Literal[True]: ...
    def build(self, n_trees: int, n_jobs: int = ...) -> Literal[True]: ...
    def unbuild(self) -> Literal[True]: ...
    def unload(self) -> Literal[True]: ...
    def get_distance(self, __i: int, __j: int) -> float: ...
    def get_n_items(self) -> int: ...
    def get_n_trees(self) -> int: ...
    def verbose(self, __v: bool) -> Literal[True]: ...
    def set_seed(self, __s: int) -> None: ...
