from typing import List
import torch


class ErrorBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self._size = 0
        self._cursor = 0

        self._errors = torch.empty(capacity, dtype=torch.float32, device=self.device)

    def _add_single(self, err: torch.Tensor) -> None:
        self._errors[self._cursor] = err
        self._inc_cursor()

    def add(self, errors: List[torch.Tensor]):
        for _err in errors:
            self._add_single(_err)

    def get_all(self) -> torch.Tensor:
        return self._errors[: self._size]

    def get_min(self) -> torch.Tensor:
        return self.get_all().min()

    def get_max(self) -> torch.Tensor:
        return self.get_all().max()

    def _inc_cursor(self) -> None:
        # Increase size until max size is reached
        if self._size < self.capacity:
            self._size += 1
        # When cursor reaches end, restart at beginning, overwriting oldest entries first
        self._cursor = (self._cursor + 1) % self.capacity

    def reset(self) -> None:
        self._cursor = 0
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self.size
