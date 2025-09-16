from typing import Optional, Tuple

def split_host_port(hostport: str, default_port: Optional[int] = None) -> Tuple[str, int]:
    if ':' in hostport:
        host, port_str = hostport.rsplit(':', 1)
        try:
            port = int(port_str)
        except ValueError:
            raise ValueError(f"Invalid port number: {port_str}")
        assert port > 0 and port < 65536, "Port number must be between 1 and 65535"

    else:
        host = hostport
        if default_port is not None:
            port = default_port
        else:
            raise ValueError("Port number is required if not provided in hostport and no default_port is set.")

    return host, port

