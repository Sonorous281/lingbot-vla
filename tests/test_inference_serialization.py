from __future__ import annotations

import asyncio
import socket
import threading
import time

import websockets.asyncio.client

from deploy.msgpack_numpy import Packer, unpackb
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


def test_multiple_websocket_connections_share_serialized_inference() -> None:
    async def request(port: int, observation: dict[str, object]):
        packer = Packer()
        async with websockets.asyncio.client.connect(
            f"ws://127.0.0.1:{port}", compression=None, max_size=None
        ) as websocket:
            assert unpackb(await websocket.recv()) == {"test": True}
            await websocket.send(packer.pack(observation))
            response = await websocket.recv()
            return response if isinstance(response, str) else unpackb(response)

    async def health(port: int) -> None:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        await writer.drain()
        response = await asyncio.wait_for(reader.read(), timeout=0.1)
        writer.close()
        await writer.wait_closed()
        assert b"200 OK" in response

    async def run() -> None:
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()

        policy = _Policy()
        server = WebsocketPolicyServer(
            policy,
            host="127.0.0.1",
            port=port,
            metadata={"test": True},
        )
        server_task = asyncio.create_task(server.run())
        for _ in range(100):
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
            except OSError:
                await asyncio.sleep(0.01)
            else:
                writer.close()
                await writer.wait_closed()
                break
        else:
            raise AssertionError("websocket server did not start")

        first = asyncio.create_task(request(port, {"value": 1}))
        await asyncio.to_thread(policy.started.wait, 1)
        await health(port)
        second = asyncio.create_task(request(port, {"value": 2}))
        results = await asyncio.gather(first, second)

        assert {result["action"] for result in results} == {1, 2}
        assert policy.peak_active == 1

        failure = await request(port, {"fail": True})
        assert isinstance(failure, str)
        assert "inference failed" in failure
        recovery = await request(port, {"value": 3})
        assert recovery["action"] == 3
        assert recovery["server_timing"]["infer_ms"] > 0

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    asyncio.run(run())
