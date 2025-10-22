"""
Thread-safe Circular Buffer for ROS2 message storage
"""
import threading
from typing import TypeVar, Generic, Optional, List
from collections import deque

T = TypeVar('T')


class CircularBuffer(Generic[T]):
    """
    Thread-safe circular buffer implementation.
    When the buffer is full, new items overwrite the oldest ones.
    """

    def __init__(self, capacity: int):
        """
        Initialize circular buffer with given capacity.

        Args:
            capacity: Maximum number of items the buffer can hold

        Raises:
            ValueError: If capacity is less than 1
        """
        if capacity < 1:
            raise ValueError("Capacity must be greater than 0")

        self._capacity = capacity
        self._buffer = [None] * capacity
        self._head = 0
        self._tail = 0
        self._size = 0
        self._is_full = False
        self._lock = threading.RLock()

    def push(self, item: T) -> None:
        """
        Add an item to the buffer.
        If buffer is full, overwrites the oldest item.

        Args:
            item: Item to add to the buffer
        """
        with self._lock:
            self._buffer[self._head] = item

            if self._is_full:
                self._tail = (self._tail + 1) % self._capacity

            self._head = (self._head + 1) % self._capacity

            if not self._is_full:
                self._size += 1

            self._is_full = (self._head == self._tail)

    def pop(self) -> Optional[T]:
        """
        Remove and return the oldest item from the buffer.

        Returns:
            The oldest item, or None if buffer is empty
        """
        with self._lock:
            if self._is_empty_unsafe():
                return None

            item = self._buffer[self._tail]
            self._buffer[self._tail] = None
            self._tail = (self._tail + 1) % self._capacity
            self._is_full = False
            self._size -= 1

            return item

    def get_latest(self) -> Optional[T]:
        """
        Get the most recent item without removing it.

        Returns:
            The most recent item, or None if buffer is empty
        """
        with self._lock:
            if self._is_empty_unsafe():
                return None

            latest_index = (self._head - 1) % self._capacity
            return self._buffer[latest_index]

    def get_oldest(self) -> Optional[T]:
        """
        Get the oldest item without removing it.

        Returns:
            The oldest item, or None if buffer is empty
        """
        with self._lock:
            if self._is_empty_unsafe():
                return None

            return self._buffer[self._tail]

    def at(self, index: int) -> Optional[T]:
        """
        Access item by index (0 = oldest, size-1 = newest).

        Args:
            index: Index of the item (0-based)

        Returns:
            The item at the specified index, or None if index is out of range
        """
        with self._lock:
            if index < 0 or index >= self._size:
                return None

            actual_index = (self._tail + index) % self._capacity
            return self._buffer[actual_index]

    def get_all(self) -> List[T]:
        """
        Get all items in the buffer as a list (oldest to newest).

        Returns:
            List of all items in the buffer
        """
        with self._lock:
            result = []
            for i in range(self._size):
                index = (self._tail + i) % self._capacity
                result.append(self._buffer[index])
            return result

    def clear(self) -> None:
        """Clear all items from the buffer."""
        with self._lock:
            self._buffer = [None] * self._capacity
            self._head = 0
            self._tail = 0
            self._size = 0
            self._is_full = False

    def size(self) -> int:
        """
        Get the current number of items in the buffer.

        Returns:
            Number of items currently stored
        """
        with self._lock:
            return self._size

    def capacity(self) -> int:
        """
        Get the maximum capacity of the buffer.

        Returns:
            Maximum number of items the buffer can hold
        """
        return self._capacity

    def is_empty(self) -> bool:
        """
        Check if the buffer is empty.

        Returns:
            True if buffer is empty, False otherwise
        """
        with self._lock:
            return self._is_empty_unsafe()

    def is_full(self) -> bool:
        """
        Check if the buffer is full.

        Returns:
            True if buffer is full, False otherwise
        """
        with self._lock:
            return self._is_full

    def _is_empty_unsafe(self) -> bool:
        """Internal method to check if buffer is empty (without lock)."""
        return not self._is_full and (self._head == self._tail)

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return self.size()

    def __bool__(self) -> bool:
        """Return True if buffer is not empty."""
        return not self.is_empty()

    def __repr__(self) -> str:
        """Return string representation of the buffer."""
        with self._lock:
            return f"CircularBuffer(capacity={self._capacity}, size={self._size})"


class CircularBufferDeque(Generic[T]):
    """
    Alternative implementation using collections.deque for simpler code.
    Slightly different behavior: uses maxlen parameter of deque.
    """

    def __init__(self, capacity: int):
        """
        Initialize circular buffer with given capacity.

        Args:
            capacity: Maximum number of items the buffer can hold

        Raises:
            ValueError: If capacity is less than 1
        """
        if capacity < 1:
            raise ValueError("Capacity must be greater than 0")

        self._capacity = capacity
        self._buffer = deque(maxlen=capacity)
        self._lock = threading.RLock()

    def push(self, item: T) -> None:
        """Add an item to the buffer."""
        with self._lock:
            self._buffer.append(item)

    def pop(self) -> Optional[T]:
        """Remove and return the oldest item."""
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return self._buffer.popleft()

    def get_latest(self) -> Optional[T]:
        """Get the most recent item without removing it."""
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return self._buffer[-1]

    def get_oldest(self) -> Optional[T]:
        """Get the oldest item without removing it."""
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return self._buffer[0]

    def at(self, index: int) -> Optional[T]:
        """Access item by index."""
        with self._lock:
            if index < 0 or index >= len(self._buffer):
                return None
            return self._buffer[index]

    def get_all(self) -> List[T]:
        """Get all items as a list."""
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        """Clear all items."""
        with self._lock:
            self._buffer.clear()

    def size(self) -> int:
        """Get current number of items."""
        with self._lock:
            return len(self._buffer)

    def capacity(self) -> int:
        """Get maximum capacity."""
        return self._capacity

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self._lock:
            return len(self._buffer) == 0

    def is_full(self) -> bool:
        """Check if buffer is full."""
        with self._lock:
            return len(self._buffer) == self._capacity

    def __len__(self) -> int:
        return self.size()

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __repr__(self) -> str:
        with self._lock:
            return f"CircularBufferDeque(capacity={self._capacity}, size={len(self._buffer)})"