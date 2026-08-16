from enum import Enum
import threading
import time
from typing import Optional


class AvatarState(Enum):
    IDLE = "idle"
    TALKING = "talking"
    INTERRUPTED = "interrupted"


class StateManager:
    """
    Manages real-time avatar states, audio queues, and synchronization
    for smooth transitions between idle video loops and active speech generation.
    """

    def __init__(self):
        self._state = AvatarState.IDLE
        self._lock = threading.Lock()
        self._interrupted_flag = threading.Event()
        self._last_state_change = time.time()

    @property
    def state(self) -> AvatarState:
        with self._lock:
            return self._state

    def set_idle(self):
        with self._lock:
            self._state = AvatarState.IDLE
            self._last_state_change = time.time()

    def set_talking(self):
        with self._lock:
            self._state = AvatarState.TALKING
            self._interrupted_flag.clear()
            self._last_state_change = time.time()

    def trigger_interrupt(self):
        """
        Signals immediate cancellation of active speech (Barge-in).
        """
        with self._lock:
            self._state = AvatarState.INTERRUPTED
            self._interrupted_flag.set()
            self._last_state_change = time.time()

    def is_interrupted(self) -> bool:
        return self._interrupted_flag.is_set()

    def reset_interrupt(self):
        self._interrupted_flag.clear()
        with self._lock:
            self._state = AvatarState.IDLE
