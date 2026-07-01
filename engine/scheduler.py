"""Static, dynamic, and continuous batching schedulers over one shared pool."""

from __future__ import annotations

from collections import deque

from engine.batch import decode_step, prefill
from engine.blocks import BlockPool
from engine.generate import _eos_ids
from engine.model import LoadedModel
from engine.request import State


class Scheduler:
    """Common engine hooks; subclasses supply the admission policy."""

    def __init__(self, lm: LoadedModel, pool: BlockPool, max_batch: int):
        self.lm = lm
        self.pool = pool
        self.max_batch = max_batch
        self.eos_ids = _eos_ids(lm)

    def finished(self, req) -> bool:
        return req.last_token in self.eos_ids or len(req.generated) >= req.max_new

    def _admit(self, req, done):
        # Prefill (emits first token); route to running or done.
        prefill(self.lm, self.pool, req)
        if self.finished(req):
            req.state = State.DONE
            done.append(req)
            return False
        return True

    def _run_to_completion(self, batch, done, step):
        # Whole-batch execution: decode until every request finishes.
        running = [r for r in batch if self._admit(r, done)]
        while running:
            decode_step(self.lm, self.pool, running)
            step += 1
            done.extend(r for r in running if self.finished(r))
            running = [r for r in running if not self.finished(r)]
        return step


class StaticScheduler(Scheduler):
    def run(self, requests):
        ready = deque(sorted(requests, key=lambda r: r.arrival))
        done, step, batches = [], 0, 0
        while ready:
            # Wait for a full batch, unless every remaining request has arrived.
            while (len([r for r in ready if r.arrival <= step]) < self.max_batch
                   and any(r.arrival > step for r in ready)):
                step += 1
            batch = []
            while ready and len(batch) < self.max_batch and ready[0].arrival <= step:
                batch.append(ready.popleft())
            step = self._run_to_completion(batch, done, step)
            batches += 1
        return done, {"steps": step, "batches": batches}


class DynamicScheduler(Scheduler):
    def __init__(self, lm, pool, max_batch, timeout: int = 4):
        super().__init__(lm, pool, max_batch)
        self.timeout = timeout

    def run(self, requests):
        ready = deque(sorted(requests, key=lambda r: r.arrival))
        done, step, batches = [], 0, 0
        while ready:
            # Fire at a full batch OR when the oldest waiter times out.
            while True:
                arrived = [r for r in ready if r.arrival <= step]
                if len(arrived) >= self.max_batch:
                    break
                if arrived and step - min(r.arrival for r in arrived) >= self.timeout:
                    break
                if arrived and not any(r.arrival > step for r in ready):
                    break
                step += 1
            batch = []
            while ready and len(batch) < self.max_batch and ready[0].arrival <= step:
                batch.append(ready.popleft())
            step = self._run_to_completion(batch, done, step)
            batches += 1
        return done, {"steps": step, "batches": batches}


class ContinuousScheduler(Scheduler):
    def run(self, requests):
        ready = deque(sorted(requests, key=lambda r: r.arrival))
        running, done, step = [], [], 0
        while ready or running:
            # Top up the running set every step (token-level admission).
            while ready and len(running) < self.max_batch and ready[0].arrival <= step:
                req = ready.popleft()
                if self._admit(req, done):
                    running.append(req)
            if not running:
                step += 1  # nothing ready yet; advance to next arrival
                continue
            decode_step(self.lm, self.pool, running)
            step += 1
            done.extend(r for r in running if self.finished(r))
            running = [r for r in running if not self.finished(r)]
        return done, {"steps": step}


_SCHEDULERS = {
    "static": StaticScheduler,
    "dynamic": DynamicScheduler,
    "continuous": ContinuousScheduler,
}


def make_scheduler(lm: LoadedModel, pool: BlockPool, max_batch: int, name: str = None):
    # Default to the configured policy (continuous).
    name = name or lm.cfg.scheduler
    return _SCHEDULERS[name](lm, pool, max_batch)
