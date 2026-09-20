"""Client-side request pacing, for an endpoint that meters by the minute."""

from __future__ import annotations

import asyncio
import threading
import time


class RateLimiter:
    """Paces calls to at most `rpm` per minute. Not thread-safe; one event loop only."""

    def __init__(
        self,
        rpm: float,
        penalty_s: float = 20.0,
        max_penalty_s: float = 120.0,
        client_max_retries: int | None = None,
    ) -> None:
        if rpm <= 0:
            raise ValueError("rpm must be positive; use 0 for an unpaced run")
        self.rpm = float(rpm)
        self.client_max_retries = client_max_retries
        self.interval = 60.0 / self.rpm
        self.penalty_s = penalty_s
        self.max_penalty_s = max_penalty_s
        self._next: float | None = None
        # A floor on departure time that penalise() raises. Separate from `_next` because
        # `_next` is the issue cursor (claimed once, then advanced) while this is re-read by
        # every sleeping caller on each wake, which is what lets a penalty reach them.
        self._penalty_floor = 0.0
        self._cond = asyncio.Condition()
        self._wakes: set = set()
        self.n_acquired = 0
        self.n_penalties = 0
        self.total_wait_s = 0.0

    async def acquire(self) -> None:
        """Wait for this caller's departure slot."""
        loop = asyncio.get_running_loop()
        async with self._cond:
            slot = self._claim_locked(loop.time())
            self.n_acquired += 1
            entered = loop.time()
            while True:
                now = loop.time()
                if self._penalty_floor > slot:
                    slot = self._claim_locked(self._penalty_floor)
                    continue
                if now >= slot:
                    break
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=slot - now)
                except TimeoutError:
                    pass
            # Once, from the two endpoints. Accumulating per sleep double-counted any wait
            # a penalty interrupted and inflated mean_wait_s.
            self.total_wait_s += max(loop.time() - entered, 0.0)

    def _claim_locked(self, floor: float) -> float:
        """Next departure slot at or after `floor`; caller holds the condition's lock.
        A run that has been idle does not get to bank the idle time as burst."""
        if self._next is None or self._next < floor:
            self._next = floor
        slot = self._next
        self._next += self.interval
        return slot

    def penalise(self) -> None:
        """A 429 was seen. Hold back every caller, including those already sleeping,
        because the limit is on the key rather than on the call: the caller that drew the
        rejection is not the one at fault, and slowing only that one leaves the run at the
        same offered rate."""
        loop = asyncio.get_running_loop()
        now = loop.time()
        base = max(self._penalty_floor, now)
        self._penalty_floor = min(base + self.penalty_s, now + self.max_penalty_s)
        # Push the issue schedule too, so slots claimed after this also land late.
        if self._next is None or self._next < self._penalty_floor:
            self._next = self._penalty_floor
        self.n_penalties += 1
        self._notify_soon(loop)

    def _notify_soon(self, loop) -> None:
        async def _wake() -> None:
            async with self._cond:
                self._cond.notify_all()

        task = loop.create_task(_wake())
        self._wakes.add(task)
        task.add_done_callback(self._wakes.discard)

    def stats(self) -> dict:
        return {
            "rpm": self.rpm,
            "client_max_retries": self.client_max_retries,
            "calls_paced": self.n_acquired,
            "n_429_penalties": self.n_penalties,
            "mean_wait_s": (
                round(self.total_wait_s / self.n_acquired, 2)
                if self.n_acquired
                else 0.0
            ),
        }


_RATE_LIMIT_MARKERS = (
    "error code: 429",
    "status_code: 429",
    "status code: 429",
    "rpm_limited",
    "burst_limited",
    "too many requests",
)


def is_rate_limit_error(err: object) -> bool:
    """A 429 from any of the routes in use. Matched on the text because the
    exception is already stringified by the time the probes see it, and because a host
    often puts the limiter's name in the body rather than in the type."""
    s = str(err).lower()
    return any(m in s for m in _RATE_LIMIT_MARKERS)


class SyncRateLimiter:
    """The same pacing for a THREAD pool, because the retrieval benchmark has one."""

    def __init__(
        self,
        rpm: float,
        penalty_s: float = 20.0,
        max_penalty_s: float = 120.0,
        client_max_retries: int | None = None,
    ) -> None:
        if rpm <= 0:
            raise ValueError("rpm must be positive; use --rpm 0 for an unpaced run")
        self.rpm = float(rpm)
        self.interval = 60.0 / self.rpm
        self.penalty_s = penalty_s
        self.max_penalty_s = max_penalty_s
        self.client_max_retries = client_max_retries
        self._next: float | None = None
        self._penalty_floor = 0.0
        self._cond = threading.Condition()
        self.n_acquired = 0
        self.n_penalties = 0
        self.total_wait_s = 0.0

    def acquire(self) -> None:
        """The threading counterpart. See RateLimiter.acquire for why a condition rather
        than a sleep: the wait must be interruptible by a penalty AND the callers it wakes
        must re-claim spaced slots rather than all departing on the floor."""
        with self._cond:
            slot = self._claim_locked(time.monotonic())
            self.n_acquired += 1
            entered = time.monotonic()
            while True:
                now = time.monotonic()
                if self._penalty_floor > slot:
                    slot = self._claim_locked(self._penalty_floor)
                    continue
                if now >= slot:
                    break
                self._cond.wait(timeout=slot - now)
            self.total_wait_s += max(time.monotonic() - entered, 0.0)

    def _claim_locked(self, floor: float) -> float:
        if self._next is None or self._next < floor:
            self._next = floor
        slot = self._next
        self._next += self.interval
        return slot

    def penalise(self) -> None:
        """See RateLimiter.penalise. notify_all is what makes the waiters' sleep
        interruptible; without it the floor rises and nobody already waiting observes it."""
        with self._cond:
            now = time.monotonic()
            base = max(self._penalty_floor, now)
            self._penalty_floor = min(base + self.penalty_s, now + self.max_penalty_s)
            if self._next is None or self._next < self._penalty_floor:
                self._next = self._penalty_floor
            self.n_penalties += 1
            self._cond.notify_all()

    def stats(self) -> dict:
        return {
            "rpm": self.rpm,
            "client_max_retries": self.client_max_retries,
            "calls_paced": self.n_acquired,
            "n_429_penalties": self.n_penalties,
            "mean_wait_s": (
                round(self.total_wait_s / self.n_acquired, 2)
                if self.n_acquired
                else 0.0
            ),
        }
