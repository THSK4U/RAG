"""
Test module: scheduler.py
A simplified scheduler that decides which requests to process next.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class RequestStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass
class SequenceRequest:
    """Represents a single inference request."""

    request_id: str
    prompt: str
    max_tokens: int = 64
    status: RequestStatus = RequestStatus.WAITING
    tokens_generated: int = 0

    def is_finished(self) -> bool:
        return self.tokens_generated >= self.max_tokens

    def step(self) -> str:
        """Simulate one decoding step, returning a fake token."""
        self.tokens_generated += 1
        if self.is_finished():
            self.status = RequestStatus.FINISHED
        return f"<tok_{self.tokens_generated}>"


class Scheduler:
    """
    A simple FCFS (First Come First Served) scheduler.
    Maintains a waiting queue and a running pool.
    """

    def __init__(self, max_running: int = 4):
        self.max_running = max_running
        self.waiting: list[SequenceRequest] = []
        self.running: list[SequenceRequest] = []
        self.finished: list[SequenceRequest] = []

    def add_request(self, req: SequenceRequest) -> None:
        """Adds a new request to the waiting queue."""
        self.waiting.append(req)

    def schedule(self) -> list[SequenceRequest]:
        """
        Moves requests from waiting to running (up to max_running),
        returns the current batch of running requests.
        """
        available_slots = self.max_running - len(self.running)
        for _ in range(min(available_slots, len(self.waiting))):
            req = self.waiting.pop(0)
            req.status = RequestStatus.RUNNING
            self.running.append(req)
        return list(self.running)

    def step(self) -> dict[str, list[str]]:
        """
        Runs one decode step for every running request.
        Returns a mapping of request_id -> [new_token].
        Finished requests are moved to self.finished.
        """
        outputs: dict[str, list[str]] = {}
        still_running: list[SequenceRequest] = []

        for req in self.running:
            token = req.step()
            outputs[req.request_id] = [token]
            if req.is_finished():
                self.finished.append(req)
            else:
                still_running.append(req)

        self.running = still_running
        return outputs

    def is_done(self) -> bool:
        """Returns True when all requests have finished."""
        return len(self.waiting) == 0 and len(self.running) == 0


def run_scheduler_demo():
    """Simple demo: schedule and decode 3 requests."""
    scheduler = Scheduler(max_running=2)

    for i in range(3):
        scheduler.add_request(
            SequenceRequest(request_id=f"req-{i}", prompt=f"prompt {i}", max_tokens=3)
        )

    while not scheduler.is_done():
        batch = scheduler.schedule()
        outputs = scheduler.step()
        print(f"Batch: {[r.request_id for r in batch]} | Outputs: {outputs}")

    print(f"Finished {len(scheduler.finished)} requests.")
