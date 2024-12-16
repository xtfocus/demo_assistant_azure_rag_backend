"""
Simple implementation to make sure no new requests added until no background tasks are running

"""

from dataclasses import dataclass


@dataclass
class TaskCounter:
    active_tasks: int = 0

    def increment(self):
        self.active_tasks += 1

    def decrement(self):
        self.active_tasks -= 1

    @property
    def is_busy(self):
        return self.active_tasks > 0
