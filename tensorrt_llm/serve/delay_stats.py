import time
from contextlib import contextmanager
from dataclasses import dataclass
from io import StringIO
from typing import Dict, List, Union

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import (ChatCompletionRequest,
                                                CompletionRequest)


@dataclass
class Delay:
    delay: float = 0
    req_count: int = 0
    last_log_time: float = 0
    label: str = ""

    def add(self, delay_label, last_timestamp, now):
        last_delay = now - last_timestamp
        self.delay += last_delay
        self.req_count += 1
        if now - self.last_log_time > 0.999999:
            avg_delay = self.delay / self.req_count if self.req_count > 0 else 0
            logger.warning(
                f"{delay_label} delay: {avg_delay}, count: {self.req_count}")
            self.reset()
            self.last_log_time = now

    def reset(self):
        self.delay = 0
        self.req_count = 0


delays: Dict[str, Delay] = {}


def log_delay(label: str, last_labels: List[str],
              request: Union[ChatCompletionRequest, CompletionRequest]):
    log_delay_ts(label, last_labels, request.timestamps)


def log_delay_ts(label: str, last_labels: List[str], req_ts: Dict[str, float]):
    if req_ts is None:
        return
    global delays
    now = time.time()
    for last_label in last_labels:
        delay_label = f"{label}-{last_label}"
        if last_label not in req_ts:
            #logger.warning(f"{last_label} not in timestamps, current label: {label}")
            continue
        if delay_label not in delays:
            delays[delay_label] = Delay(label=delay_label)
        delays[delay_label].add(delay_label, req_ts[last_label], now)
    req_ts[label] = time.time()


def log_delay_ts_once(label: str, last_labels: List[str], req_ts: Dict[str,
                                                                       float]):
    if req_ts is None or label in req_ts:
        return
    log_delay_ts(label, last_labels, req_ts)


@contextmanager
def scoped_delay(delay_label: str, request: Union[ChatCompletionRequest,
                                                  CompletionRequest]):
    global delays
    start_st = time.time()
    yield
    if delay_label not in delays:
        delays[delay_label] = Delay(label=delay_label)
    delays[delay_label].add(delay_label, start_st, time.time())


@contextmanager
def profile_and_print():
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()

    try:
        yield
    finally:
        profiler.disable()
    stream = StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats(
        pstats.SortKey.CUMULATIVE).print_stats(50)
    logger.warning(stream.getvalue())
