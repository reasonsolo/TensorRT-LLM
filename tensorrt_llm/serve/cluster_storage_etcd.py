import asyncio
from typing import Callable, List

import etcd3

from tensorrt_llm.serve.cluster_storage import (ClusterStorage, StorageItem,
                                                WatchEvent, WatchEventQueue,
                                                WatchEventType)


class Etcd3WatchEventQueue(WatchEventQueue):

    def __init__(self,
                 key_prefix: str,
                 cancel_event: Callable[[], None] = None):
        self.key_prefix = key_prefix
        self._cancel_event = cancel_event
        self.events = asyncio.Queue()

    def set_cancel_event(self, cancel_event: Callable[[], None]):
        self._cancel_event = cancel_event

    def __del__(self):
        if self._cancel_event:
            self._cancel_event()

    def add_event(self, watch_resp):
        for event in watch_resp.events:
            # Event type is even not in public interface of etcd3
            event_type = WatchEventType.SET if "Put" in event.__class__.__name__ else WatchEventType.DELETE
            self.events.put_nowait(
                WatchEvent(
                    StorageItem(key=event.key.decode('utf-8'),
                                value=event.value.decode('utf-8')), event_type))
        if self.events._loop:
            self.events._loop._write_to_self()


class Etcd3ClusterStorage(ClusterStorage):

    def __init__(self,
                 cluster_uri: str,
                 cluster_name: str,
                 one_single_lease: bool = False):
        # This will support multiple etcd servers like etcd1:2379,etcd2:2379
        cluster_uri = cluster_uri.replace("etcd://", "")
        host, port = cluster_uri.rsplit(":", 1)
        self._client = etcd3.client(host, port)
        self._leases = {}
        self._instance_lease = None
        self._watch_handles = {}
        self._one_single_lease = one_single_lease

    def __del__(self):
        self._client.close()

    def _get_lease(self, key: str, ttl: int = -1) -> etcd3.Lease:
        if ttl <= 0:
            return None
        if self._one_single_lease:
            return self._instance_lease
        if key not in self._leases:
            self._leases[key] = self._client.lease(ttl)
        return self._leases[key]

    async def set(self,
                  key: str,
                  value: str,
                  overwrite_if_exists: bool = False,
                  ttl: int = -1) -> bool:
        lease = self._get_lease(key, ttl)
        if not overwrite_if_exists:
            self._client.put_if_not_exists(key, value, lease=lease)
        else:
            self._client.put(key, value, lease=lease)
        return True

    async def get(self, key: str) -> str:
        data, meta = self._client.get(key)
        return data.decode('utf-8') if data else None

    async def delete(self, key: str) -> bool:
        self._client.delete(key)
        return True

    async def expire(self, key: str, ttl: int) -> bool:
        if ttl <= 0:
            raise ValueError(f"TTL must be greater than 0, got {ttl}")
        lease = self._get_lease(key, ttl)
        # TTL will be ignored since it can only be set when creating a lease
        self._client.refresh_lease(lease_id=lease.id)
        return True

    async def get_keys(self, key_prefix: str) -> List[str]:
        return [
            metadata.key.decode('utf-8')
            for _, metadata in self._client.get_prefix(key_prefix,
                                                       keys_only=True)
        ]

    async def watch(self, key_prefix: str) -> WatchEventQueue:
        if key_prefix in self._watch_handles:
            return self._watch_handles[key_prefix]
        watch_handle = Etcd3WatchEventQueue(key_prefix=key_prefix)
        watch_id = self._client.add_watch_prefix_callback(
            key_prefix, watch_handle.add_event)
        watch_handle.set_cancel_event(
            lambda: self._client.cancel_watch(watch_id))
        self._watch_handles[key_prefix] = watch_handle
        return watch_handle

    async def unwatch(self, key_prefix: str) -> None:
        self._watch_handles.pop(key_prefix)
