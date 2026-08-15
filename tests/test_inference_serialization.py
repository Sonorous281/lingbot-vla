from __future__ import annotations

import asyncio
import threading
import time

from deploy.websocket_policy_server import WebsocketPolicyServer


class _Policy:
    def __init__(self) -> None:
        self.active = 0
        self.peak_active = 0
        self.started = threading.Event()
        self._guard = threading.Lock()

    def infer(self, observation: dict[str, object]) -> dict[str, object]:
        with self._guard:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        self.started.set()
        try:
            time.sleep(0.05)
            if observation.get("fail"):
                raise RuntimeError("inference failed")
            return {"action": observation["value"]}
        finally:
            with self._guard:
                self.active -= 1


def test_inference_is_serialized_without_blocking_event_loop() -> None:
    async def run() -> None:
        policy = _Policy()
        server = WebsocketPolicyServer(policy)
        heartbeats = 0

        async def heartbeat() -> None:
            nonlocal heartbeats
            for _ in range(5):
                await asyncio.sleep(0.01)
                heartbeats += 1

        first = asyncio.create_task(server._infer({"value": 1}))
        await asyncio.to_thread(policy.started.wait, 1)
        second = asyncio.create_task(server._infer({"value": 2}))
        results = await asyncio.gather(first, second, heartbeat())

        assert results[:2] == [{"action": 1}, {"action": 2}]
        assert heartbeats == 5
        assert policy.peak_active == 1
        server._inference_executor.shutdown(wait=True)

    asyncio.run(run())


def test_inference_lock_recovers_after_policy_error() -> None:
    async def run() -> None:
        policy = _Policy()
        server = WebsocketPolicyServer(policy)

        try:
            await server._infer({"fail": True})
        except RuntimeError as error:
            assert str(error) == "inference failed"
        else:
            raise AssertionError("expected inference failure")

        assert await server._infer({"value": 3}) == {"action": 3}
        assert policy.peak_active == 1
        server._inference_executor.shutdown(wait=True)

    asyncio.run(run())
