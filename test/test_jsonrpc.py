from __future__ import annotations

import asyncio
import json

import pytest

from nomad import jsonrpc
from nomad.jsonrpc import Dispatcher, JSONRPCError, ProtocolError


def _dispatch(handlers, message, **kwargs):
    if not isinstance(message, str):
        message = json.dumps(message)
    return Dispatcher(handlers, **kwargs).dispatch(message)


def test_result_may_be_any_json_value():
    handlers = {
        "list": lambda params: [1, 2],
        "none": lambda params: None,
        "echo": lambda params: params,
    }

    assert _dispatch(handlers, jsonrpc.make_request("list", request_id=1)) == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": [1, 2],
    }
    assert (
        _dispatch(handlers, jsonrpc.make_request("none", request_id=2))["result"]
        is None
    )
    assert _dispatch(
        handlers, jsonrpc.make_request("echo", {"a": 1}, request_id="x")
    ) == {"jsonrpc": "2.0", "id": "x", "result": {"a": 1}}


@pytest.mark.parametrize(
    ("message", "code"),
    [
        ("{not json", jsonrpc.PARSE_ERROR),
        ([], jsonrpc.INVALID_REQUEST),
        ({"jsonrpc": "1.0", "id": 1, "method": "m"}, jsonrpc.INVALID_REQUEST),
        ({"jsonrpc": "2.0", "id": 1, "method": 5}, jsonrpc.INVALID_REQUEST),
        ({"jsonrpc": "2.0", "id": True, "method": "m"}, jsonrpc.INVALID_REQUEST),
        (
            {"jsonrpc": "2.0", "id": 1, "method": "m", "params": 3},
            jsonrpc.INVALID_PARAMS,
        ),
    ],
    ids=["parse", "batch", "version", "method", "bool-id", "params"],
)
def test_malformed_requests_get_spec_errors_with_null_id(message, code):
    response = _dispatch({"m": lambda params: None}, message)

    assert response["id"] is None
    assert response["error"]["code"] == code


def test_unknown_method_and_positional_params_are_rejected():
    unknown = _dispatch({}, jsonrpc.make_request("missing", request_id=3))
    positional = _dispatch(
        {"m": lambda params: None},
        {"jsonrpc": "2.0", "id": 4, "method": "m", "params": [1]},
    )

    assert unknown["id"] == 3
    assert unknown["error"]["code"] == jsonrpc.METHOD_NOT_FOUND
    assert positional["error"]["code"] == jsonrpc.INVALID_PARAMS


def test_handler_exceptions_become_server_errors():
    seen: list[Exception] = []

    def fail(params):
        raise RuntimeError("boom")

    def reject(params):
        raise JSONRPCError("bad batch_size", code=jsonrpc.INVALID_PARAMS, data=1)

    handlers = {"fail": fail, "reject": reject}
    failed = _dispatch(
        handlers, jsonrpc.make_request("fail", request_id=5), on_error=seen.append
    )
    rejected = _dispatch(
        handlers, jsonrpc.make_request("reject", request_id=6), on_error=seen.append
    )

    assert failed["error"] == {"code": jsonrpc.SERVER_ERROR, "message": "boom"}
    assert rejected["error"] == {
        "code": jsonrpc.INVALID_PARAMS,
        "message": "bad batch_size",
        "data": 1,
    }
    assert [str(exc) for exc in seen] == ["boom"]


def test_notifications_run_without_a_response():
    calls: list[dict] = []
    notification = {"jsonrpc": "2.0", "method": "m", "params": {"a": 1}}

    assert _dispatch({"m": calls.append}, notification) is None
    assert calls == [{"a": 1}]


def test_async_dispatch_awaits_handlers():
    async def handler(params):
        return params["value"] * 2

    dispatcher = Dispatcher({"double": handler})
    message = json.dumps(jsonrpc.make_request("double", {"value": 4}, request_id=7))

    assert asyncio.run(dispatcher.dispatch_async(message))["result"] == 8
    assert dispatcher.dispatch(message)["error"]["code"] == jsonrpc.SERVER_ERROR


def test_parse_response_returns_result_or_raises_peer_error():
    assert jsonrpc.parse_response(jsonrpc.make_result(1, [1]), expected_id=1) == [1]

    error = JSONRPCError("nope", code=jsonrpc.METHOD_NOT_FOUND)
    with pytest.raises(JSONRPCError, match="nope") as raised:
        jsonrpc.parse_response(jsonrpc.make_error(1, error), expected_id=1)
    assert raised.value.code == jsonrpc.METHOD_NOT_FOUND

    with pytest.raises(JSONRPCError, match="Parse error"):
        jsonrpc.parse_response(
            jsonrpc.make_error(None, JSONRPCError("Parse error")), expected_id=1
        )


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"jsonrpc": "2.0", "id": 1},
        {"jsonrpc": "2.0", "id": 1, "result": 1, "error": {}},
        {"jsonrpc": "2.0", "id": 2, "result": 1},
        {"jsonrpc": "2.0", "id": 1, "error": "bad"},
        {"jsonrpc": "2.0", "id": 2, "error": {"code": 1, "message": "x"}},
    ],
    ids=["non-object", "empty", "both", "wrong-id", "bad-error", "error-wrong-id"],
)
def test_parse_response_rejects_malformed_responses(payload):
    with pytest.raises(ProtocolError):
        jsonrpc.parse_response(payload, expected_id=1)


def test_encode_response_falls_back_to_internal_error():
    response = jsonrpc.make_result(8, object())

    encoded = json.loads(jsonrpc.encode_response(response, jsonrpc.encode_line))

    assert encoded["id"] == 8
    assert encoded["error"]["code"] == jsonrpc.INTERNAL_ERROR


def test_encode_line_keeps_embedded_newlines_on_one_line():
    line = jsonrpc.encode_line(jsonrpc.make_result(1, "a\nb"))

    assert line.count("\n") == 1
    assert json.loads(line)["result"] == "a\nb"


def test_frames_round_trip_and_enforce_max_size():
    async def read(data: bytes, **kwargs):
        reader = asyncio.StreamReader()
        reader.feed_data(data)
        reader.feed_eof()
        return await jsonrpc.read_frame(reader, **kwargs)

    payload = jsonrpc.make_result(1, {"x": 1})
    frame = jsonrpc.encode_frame(payload)

    assert json.loads(asyncio.run(read(frame))) == payload
    assert asyncio.run(read(b"")) is None
    with pytest.raises(ConnectionError):
        asyncio.run(read(frame[:-1]))
    with pytest.raises(ProtocolError):
        asyncio.run(read(frame, max_size=4))
    with pytest.raises(ValueError, match="max size"):
        jsonrpc.encode_frame(payload, max_size=4)
