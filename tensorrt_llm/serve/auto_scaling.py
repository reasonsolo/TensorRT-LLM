import asyncio
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import List, Tuple

from tensorrt_llm.llmapi.disagg_utils import DisaggClusterConfig, ServerRole

# from tensorrt_llm.logger import logger
from .cluster_storage import (ClusterStorage, WatchEvent, WatchEventType,
                              key_time, logger)


@dataclass
class WorkerInfo:
    worker_id: str
    host: str
    port: int
    role: ServerRole
    status: str


@dataclass
class WorkerWatchEvent:
    key_prefixes: List[str]
    events: List[WatchEvent]


def get_worker_key_prefix(cluster_name: str):
    return f"/trtllm-disagg/{cluster_name}/workers"


def get_worker_key(name: str, role: ServerRole, worker_id: str = "") -> str:
    return f"{get_worker_key_prefix(name)}/{worker_id}"


class ClusterManager:

    def __init__(self, config: DisaggClusterConfig, storage: ClusterStorage):
        self._config = config
        self._cluster_storage = storage
        self._lock = asyncio.Lock()
        self._minimal_ctx_worker_num = config.minimal_instances.context_servers
        self._minimal_gen_worker_num = config.minimal_instances.generation_servers
        self._current_ctx_workers = {}
        self._current_gen_workers = {}
        self._watch_handle = None

    async def cluster_info(self) -> dict:
        async with self._lock:
            return {
                "current_workers": {
                    "context_servers": [
                        asdict(worker)
                        for worker in self._current_ctx_workers.values()
                    ],
                    "generation_servers": [
                        asdict(worker)
                        for worker in self._current_gen_workers.values()
                    ]
                },
                "minimal_instances": {
                    "context_servers": self._minimal_ctx_worker_num,
                    "generation_servers": self._minimal_gen_worker_num
                },
            }

    @property
    def current_ctx_worker_num(self):
        return len(self._current_ctx_workers)

    @property
    def current_gen_worker_num(self):
        return len(self._current_gen_workers)

    @property
    def worker_key_prefix(self):
        return get_worker_key_prefix(self._config.cluster_name)

    # returns future([new_workers], [inactive_workers])
    async def watch_workers(self) -> Tuple[List[str], List[str]]:
        self._watch_handle = await self._cluster_storage.watch(
            self.worker_key_prefix)

    async def unwatch_workers(self):
        await self._cluster_storage.unwatch([self.worker_key_prefix])
        self._watch_handle = None

    async def get_worker_events(
            self) -> List[Tuple[WorkerInfo, WatchEventType]]:
        events = await self._watch_handle.drain()
        worker_events = []
        for event in events:
            worker_info = self._parse_worker_info(event.storage_item.value)
            need_notify_event = await self._update_workers(
                worker_info, event.event_type)
            if need_notify_event:
                worker_events.append((worker_info, event.event_type))
        return worker_events

    def _log_cluster_status(self, worker_info: WorkerInfo, change_event: str):
        logger.info(
            f"Worker {worker_info.worker_id} becomes {change_event}, current context worker: {self.current_ctx_worker_num}/{self._minimal_ctx_worker_num}, current generation worker: {self.current_gen_worker_num}/{self._minimal_gen_worker_num}"
        )

    def _need_notify_event(self, current_workers: dict[str, WorkerInfo],
                           worker_info: WorkerInfo, event_type: WatchEventType):
        if event_type == WatchEventType.SET:
            if worker_info.worker_id in current_workers and current_workers[
                    worker_info.worker_id] == worker_info:
                return False
        elif event_type == WatchEventType.DELETE:
            if worker_info.worker_id not in current_workers:
                return False
        return True

    async def _update_workers(self, worker_info: WorkerInfo,
                              event_type: WatchEventType) -> bool:
        async with self._lock:
            curr_workers = None
            if worker_info.role == ServerRole.CONTEXT:
                curr_workers = self._current_ctx_workers
            elif worker_info.role == ServerRole.GENERATION:
                curr_workers = self._current_gen_workers
            else:
                raise ValueError(
                    f"Invalid worker role: {worker_info.role.name}")
            need_notify_event = self._need_notify_event(curr_workers,
                                                        worker_info, event_type)
            if need_notify_event:
                if event_type == WatchEventType.SET:
                    curr_workers[worker_info.worker_id] = worker_info
                elif event_type == WatchEventType.DELETE:
                    curr_workers.pop(worker_info.worker_id)
                self._log_cluster_status(
                    worker_info, "active/updated"
                    if event_type == WatchEventType.SET else "inactive")
            else:
                logger.debug(
                    f"Worker {worker_info.worker_id} is already in the cluster or unknown before, skipping {event_type.name} event"
                )
            return need_notify_event

    def _parse_worker_info(self, worker_info: str) -> WorkerInfo:
        worker_info = WorkerInfo(**json.loads(worker_info))
        worker_info.role = ServerRole(worker_info.role)
        return worker_info

    async def is_ready(self) -> bool:
        return self.current_ctx_worker_num >= self._minimal_ctx_worker_num and self.current_gen_worker_num >= self._minimal_gen_worker_num

    async def is_ready_with_router(self, router_ctx_worker_num: int,
                                   router_gen_worker_num: int) -> bool:
        return router_ctx_worker_num >= self._minimal_ctx_worker_num and router_gen_worker_num >= self._minimal_gen_worker_num


class ClusterWorker:

    def __init__(self, role: ServerRole, host: str, port: int,
                 config: DisaggClusterConfig, storage: ClusterStorage):
        self._role = role
        self._host = host
        self._port = port
        self._config = config
        self._cluster_storage = storage
        self._stop = False
        self._heartbeat_task = None
        self._last_heartbeat = 0
        self._worker_id = f"{role.name}-{host}:{port}-{int(time.time()*1000)}-{os.getpid()}-{random.randint(0, 1000):03}"

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def worker_info(self) -> WorkerInfo:
        return WorkerInfo(worker_id=self._worker_id,
                          role=self._role,
                          host=self._host,
                          port=self._port,
                          status="")

    @property
    def worker_key(self) -> str:
        return get_worker_key(self._config.cluster_name, self._role,
                              self._worker_id)

    async def register_worker(self, validator=None, retry_interval=5):
        if validator and not validator():
            logger.warning(
                f"Worker {self.worker_info.worker_id} is not valid, skipping registration"
            )
            return False
        worker_info = self.worker_info
        logger.debug(
            f"Worker {self.worker_info.worker_id} registering, {asdict(worker_info)}"
        )
        success = await self._cluster_storage.set(
            self.worker_key,
            json.dumps(asdict(worker_info)),
            ttl=self._config.inactive_timeout)
        if not success:
            if retry_interval > 0:
                logger.warning(
                    f"Worker {self.worker_info.worker_id} registration failed, retry in {retry_interval} seconds"
                )
                await asyncio.sleep(max(10, retry_interval))
                return await self.register_worker(validator, retry_interval + 1)
        else:
            logger.info(
                f"Worker {self.worker_info.worker_id} registration successful")
        self._last_heartbeat = key_time()
        if self._config.heartbeat_interval > 0 and self._config.heartbeat_interval < self._config.inactive_timeout:
            if not self._heartbeat_task:
                self._heartbeat_task = asyncio.create_task(
                    self._heartbeat(validator))
        else:
            logger.warning(
                f"Heartbeat interval {self._config.heartbeat_interval} is not positive or less than inactive timeout {self._config.inactive_timeout}, heartbeat is disabled"
            )
        return True

    async def deregister_worker(self):
        self._stop = True
        self._heartbeat_task.cancel()
        self._heartbeat_task = None
        success = await self._cluster_storage.delete(self.worker_key)
        if not success:
            logger.warning(
                f"Worker {self.worker_info.worker_id} deregistration failed")
        return success

    async def _heartbeat(self, validator=None):
        logger.info(f"Worker {self.worker_info.worker_id} heartbeat started")
        while not self._stop:
            remaining_time = self._config.heartbeat_interval - (
                key_time() - self._last_heartbeat)
            logger.info(
                f"Worker {self.worker_info.worker_id} heartbeat remaining time: {remaining_time}"
            )
            if remaining_time > 0:
                await asyncio.sleep(remaining_time)
            self._last_heartbeat = key_time()
            if validator and not validator():
                logger.warning(
                    f"Worker {self.worker_info.worker_id} is not valid, skipping heartbeat {key_time()}"
                )
                continue
            expire_res = await self._cluster_storage.expire(
                self.worker_key, self._config.inactive_timeout)
            if not expire_res:
                logger.warning(
                    f"Worker {self.worker_info.worker_id} heartbeat failed, re-registering {key_time()}"
                )
                await self.register_worker(validator)
            else:
                logger.debug(
                    f"Worker {self.worker_info.worker_id} heartbeat successful {key_time()}"
                )
