import socket

import pytest

from kodb.adapters.server import start_server


def test_start_server_refuses_a_port_already_in_use():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        url = f"http://127.0.0.1:{s.getsockname()[1]}"
        # before the fix, ready() answered for whatever server held the port and that server was measured
        with pytest.raises(SystemExit, match="already in use"):
            start_server("test", ["true"], url, lambda: True)
