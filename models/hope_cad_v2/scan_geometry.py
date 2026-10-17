"""Spatial routing for the frozen HOPE-CAD v2 design (four independent scans)."""

from dataclasses import dataclass

import torch


DIRECTIONS = ("row_forward", "row_reverse", "column_forward", "column_reverse")


@dataclass(frozen=True)
class ScanRoute:
    name: str
    permutation: torch.Tensor
    inverse: torch.Tensor
    chunk_size: int

    def scan(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reorder [..., N, channels] from spatial row-major into this route."""
        if tokens.shape[-2] != self.permutation.numel():
            raise ValueError("token count does not match scan geometry")
        return tokens.index_select(-2, self.permutation)

    def restore(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-2] != self.inverse.numel():
            raise ValueError("token count does not match scan geometry")
        return tokens.index_select(-2, self.inverse)


def scan_routes(height: int, width: int, *, device=None) -> tuple[ScanRoute, ...]:
    """Reverse the whole flattened sequence, not each individual row/column."""
    if any(type(n) is not int or n < 1 for n in (height, width)):
        raise ValueError("grid dimensions must be positive integers")
    grid = torch.arange(height * width, device=device).reshape(height, width)
    row, column = grid.flatten(), grid.t().flatten()
    permutations = (row, row.flip(0), column, column.flip(0))
    return tuple(
        ScanRoute(name, permutation, torch.argsort(permutation), chunk)
        for name, permutation, chunk in zip(
            DIRECTIONS, permutations, (width, width, height, height)
        )
    )


def chunk_spans(length: int, chunk_size: int) -> tuple[tuple[int, int], ...]:
    if any(type(n) is not int or n < 1 for n in (length, chunk_size)):
        raise ValueError("length and chunk_size must be positive integers")
    return tuple((start, min(start + chunk_size, length)) for start in range(0, length, chunk_size))
