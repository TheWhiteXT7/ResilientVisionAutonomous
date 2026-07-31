"""Abstract base dataset loader module."""

import abc
import logging
from typing import Any, Generic, Iterator, List, Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseDatasetLoader(abc.ABC, Sequence[T], Generic[T]):
    """Abstract base class for dataset loaders.

    Provides standard sequence interface including index access, iteration,
    length queries, and sample lookup by string ID.
    """

    @property
    @abc.abstractmethod
    def sample_ids(self) -> List[str]:
        """Return list of sample IDs in the dataset split."""
        pass

    @abc.abstractmethod
    def get_sample_by_id(self, sample_id: str) -> T:
        """Fetch a dataset sample by its string ID.

        Args:
            sample_id: Unique string identifier of the sample.

        Returns:
            The dataset sample object.
        """
        pass

    def __len__(self) -> int:
        """Return total number of samples in the dataset split."""
        return len(self.sample_ids)

    def __getitem__(self, idx: Any) -> Any:
        """Retrieve sample by integer index, slice, or string ID.

        Args:
            idx: Integer index, slice, or string sample ID.

        Returns:
            Dataset sample or list of dataset samples for slices.

        Raises:
            IndexError: If integer index is out of bounds.
            TypeError: If index type is invalid.
        """
        if isinstance(idx, str):
            return self.get_sample_by_id(idx)
        elif isinstance(idx, int):
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError(
                    f"Index {idx} out of range (length {len(self)})"
                )
            sample_id = self.sample_ids[idx]
            return self.get_sample_by_id(sample_id)
        elif isinstance(idx, slice):
            return [self.get_sample_by_id(sid) for sid in self.sample_ids[idx]]
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def __iter__(self) -> Iterator[T]:
        """Iterate over all samples in the dataset split."""
        for sample_id in self.sample_ids:
            yield self.get_sample_by_id(sample_id)
