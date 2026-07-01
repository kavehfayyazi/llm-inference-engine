"""Static, dynamic, and continuous batching schedulers over one shared pool."""

from __future__ import annotations

from collections import deque

from engine.batch import decode_step, prefill, prefill_chunk
from engine.blocks import BlockPool
from engine.generate import _eos_ids
from engine.metrics import now
from engine.model import LoadedModel
from engine.request import State


class Scheduler:
    """Common engine hooks; subclasses supply the admission policy."""

    def __init__(self, lm: LoadedModel, pool: BlockPool, max_batch: int):
        self.lm = lm
        self.pool = pool
        self.max_batch = max_batch
        self.eos_ids = _eos_ids(lm)
        self.chunk = lm.cfg.prefill_chunk
        self._start = 0.0
        self._live_sum = 0     # actual cached tokens, summed over steps
        self._padded_sum = 0   # tokens a dense [B, max_len] layout would reserve
        self._paged_sum = 0    # tokens paging actually reserves (blocks * block_size)

    def finished(self, req) -> bool:
        return req.last_token in self.eos_ids or len(req.generated) >= req.max_new

    def _elapsed(self) -> float:
        return now(self.lm.device) - self._start

    def _record(self, running):
        # Snapshot batch KV footprint each step. A padded/pre-allocated batcher
        # reserves each slot's full max length (prompt + max_new) for the whole
        # run; paging allocates blocks on demand as the sequence grows.
        if not running:
            return
        self._live_sum += sum(r.kv.live_tokens() for r in running)
        self._padded_sum += sum(r.prompt_ids.shape[1] + r.max_new for r in running)
        self._paged_sum += sum(r.kv.reserved_tokens() for r in running)

    def _kv_metrics(self):
        def waste(reserved):
            return round(100 * (1 - self._live_sum / reserved), 1) if reserved else 0.0
        return {"pad_waste_pct": waste(self._padded_sum), "frag_pct": waste(self._paged_sum)}

    def _admit(self, req, done, step):
        # Prefill (emits first token); stamp first-token time/step; route.
        prefill(self.lm, self.pool, req)
        req.t_first = self._elapsed()
        req.s_first = step
        if self.finished(req):
            req.state = State.DONE
            req.t_finish = self._elapsed()
            req.s_finish = step
            req.kv.release()  # free blocks back to the shared pool
            done.append(req)
            return False
        return True

    def _drain_finished(self, running, done, step):
        # Stamp + move finished requests out of the running set.
        still = []
        for r in running:
            if self.finished(r):
                r.state = State.DONE
                r.t_finish = self._elapsed()
                r.s_finish = step
                r.kv.release()  # free blocks back to the shared pool
                done.append(r)
            else:
                still.append(r)
        return still

    def _run_to_completion(self, batch, done, step):
        # Whole-batch execution: decode until every request finishes.
        running = [r for r in batch if self._admit(r, done, step)]
        while running:
            self._record(running)
            decode_step(self.lm, self.pool, running)
            step += 1
            running = self._drain_finished(running, done, step)
        return step


class StaticScheduler(Scheduler):
    def run(self, requests):
        self._start = now(self.lm.device)
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
        return done, {"steps": step, "batches": batches, "wall_s": self._elapsed(), **self._kv_metrics()}


class DynamicScheduler(Scheduler):
    def __init__(self, lm, pool, max_batch, timeout: int = 4):
        super().__init__(lm, pool, max_batch)
        self.timeout = timeout

    def run(self, requests):
        self._start = now(self.lm.device)
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
        return done, {"steps": step, "batches": batches, "wall_s": self._elapsed(), **self._kv_metrics()}


class ContinuousScheduler(Scheduler):
    def run(self, requests):
        self._start = now(self.lm.device)
        ready = deque(sorted(requests, key=lambda r: r.arrival))
        prefilling, running, done, step = [], [], [], 0
        while ready or prefilling or running:
            # Admit up to capacity; new requests enter chunked prefill.
            while ready and (len(prefilling) + len(running)) < self.max_batch and ready[0].arrival <= step:
                req = ready.popleft()
                req.state = State.PREFILL
                prefilling.append(req)
            if not prefilling and not running:
                step += 1  # nothing ready yet; advance to next arrival
                continue

            # Advance each prefilling request by one prompt chunk this step.
            still = []
            for req in prefilling:
                if not prefill_chunk(self.lm, self.pool, req, self.chunk):
                    still.append(req)
                    continue
                req.t_first = self._elapsed()
                req.s_first = step
                if self.finished(req):
                    req.state = State.DONE
                    req.t_finish = self._elapsed()
                    req.s_finish = step
                    req.kv.release()
                    done.append(req)
                else:
                    running.append(req)
            prefilling = still

            if running:
                self._record(running)
                decode_step(self.lm, self.pool, running)
            step += 1
            running = self._drain_finished(running, done, step)
        return done, {"steps": step, "wall_s": self._elapsed(), **self._kv_metrics()}


_SCHEDULERS = {
    "static": StaticScheduler,
    "dynamic": DynamicScheduler,
    "continuous": ContinuousScheduler,
}


def make_scheduler(lm: LoadedModel, pool: BlockPool, max_batch: int, name: str = None):
    # Default to the configured policy (continuous).
    name = name or lm.cfg.scheduler
    return _SCHEDULERS[name](lm, pool, max_batch)
