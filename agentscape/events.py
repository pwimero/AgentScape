"""Complete, disk-backed event history with a bounded in-memory tail."""
from __future__ import annotations

import gzip
import json
import os
import shutil
import threading
from collections import deque
from collections.abc import Sequence
from itertools import islice
from pathlib import Path


class EventJournal(Sequence):
    def __init__(self, directory: Path, capacity=32):
        if capacity < 1:
            raise ValueError("Event buffer capacity must be positive.")
        self.path = directory / "events.jsonl"
        self.compressed_path = directory / "events.jsonl.gz"
        self._tail = deque(maxlen=capacity)
        self._count = 0
        self._sealed = False
        self._lock = threading.RLock()
        self.path.touch(mode=0o600, exist_ok=False)

    def __len__(self):
        with self._lock:
            return self._count

    def append(self, entry):
        encoded = json.dumps(entry, ensure_ascii=False) + "\n"
        with self._lock:
            if self._sealed:
                raise ValueError("Cannot append to an archived event journal.")
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(encoded)
            self._tail.append(entry)
            self._count += 1

    def __iter__(self):
        # Open under the lock, so archiving cannot remove the source between
        # selecting it and opening it. Yield outside the lock to allow writers.
        with self._lock:
            count = self._count
            if not count:
                return
            source = (gzip.open(self.compressed_path, "rt", encoding="utf-8") if self._sealed
                      else self.path.open(encoding="utf-8"))
        with source:
            for line in islice(source, count):
                yield json.loads(line)

    def __getitem__(self, key):
        with self._lock:
            count = self._count
            if isinstance(key, slice):
                start, stop, step = key.indices(count)
                if step < 0:
                    return list(self)[key]
                if start >= count - len(self._tail):
                    tail = list(self._tail)
                    return tail[start - (count - len(tail)):stop - (count - len(tail)):step]
                return list(islice(iter(self), start, stop, step))
            index = key if key >= 0 else count + key
            if not 0 <= index < count:
                raise IndexError("Event index out of range.")
            if index >= count - len(self._tail):
                return self._tail[index - (count - len(self._tail))]
            return next(islice(iter(self), index, index + 1))

    def archive(self):
        """Seal and gzip the complete journal; keep the original on failure."""
        with self._lock:
            if self._sealed:
                return
            if not self._count:
                self.path.touch(mode=0o600)
            temporary = self.compressed_path.with_suffix(".gz.tmp")
            try:
                descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "wb") as output:
                    with self.path.open("rb") as source, gzip.GzipFile(fileobj=output, mode="wb", compresslevel=1) as compressed:
                        shutil.copyfileobj(source, compressed, length=256 * 1024)
                temporary.replace(self.compressed_path)
                self._sealed = True
                self.path.unlink()
            finally:
                temporary.unlink(missing_ok=True)
