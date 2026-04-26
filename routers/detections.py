import asyncio
import copy
import json
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import shared_state


router = APIRouter()


def _safe_json_default(value):
    # Convert common non-JSON-native values (for example numpy scalars).
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


async def _detection_event_stream():
    last_serialized = None
    last_heartbeat = time.monotonic()

    try:
        while True:
            snapshot = copy.deepcopy(shared_state.data_structure)
            payload = {
                "success": True,
                "count": len(snapshot),
                "result": snapshot,
            }

            serialized = json.dumps(payload, default=_safe_json_default)
            if serialized != last_serialized:
                yield f"data: {serialized}\n\n"
                last_serialized = serialized

            # Keep-alive comment so idle proxies/connections do not close the stream.
            if time.monotonic() - last_heartbeat > 15:
                yield ": keep-alive\n\n"
                last_heartbeat = time.monotonic()

            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        return


@router.get("/detections")
async def get_detections():
    return {"success": True, "count": len(shared_state.data_structure), "result": shared_state.data_structure}


@router.get("/detections/stream")
async def stream_detections():
    return StreamingResponse(
        _detection_event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
