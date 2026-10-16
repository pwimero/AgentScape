#!/bin/sh
set -eu
AGENTSCAPE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ ! -x "$AGENTSCAPE_ROOT/.venv/bin/python" ]; then
    printf 'Run ./setup.sh first.\n' >&2
    exit 1
fi
exec "$AGENTSCAPE_ROOT/.venv/bin/python" -m agentscape "$@"
