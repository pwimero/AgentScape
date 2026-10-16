#!/bin/sh
set -eu
AGENTSCAPE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$AGENTSCAPE_ROOT"
command -v python3 >/dev/null || { printf 'Install Python 3.11+ first.\n'; exit 1; }
command -v ollama >/dev/null || { printf 'Install Ollama first: https://ollama.com\n'; exit 1; }
python3 -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ required"'
if [ ! -d .venv ]; then python3 -m venv .venv; fi
.venv/bin/python -m pip install --disable-pip-version-check -e . -c requirements.lock
chmod +x agentscape.sh
mkdir -p "$HOME/.local/bin"
# Python quotes paths correctly, including spaces and shell metacharacters.
.venv/bin/python - "$AGENTSCAPE_ROOT" <<'PY'
import pathlib, shlex, sys
root = pathlib.Path(sys.argv[1])
launcher = pathlib.Path.home() / '.local/bin/agentscape'
if launcher.exists() and 'agentscape' not in launcher.read_text().lower():
    raise SystemExit(f'Refusing to replace unrelated launcher: {launcher}')
launcher.write_text('#!/bin/sh\nexec ' + shlex.quote(str(root / '.venv/bin/python')) + ' -m agentscape "$@"\n')
launcher.chmod(0o755)
shell_file = pathlib.Path.home() / ('.zshrc' if 'zsh' in __import__('os').environ.get('SHELL', '') else '.bashrc')
addition = '\n# Agentscape terminal launcher\nexport PATH="$HOME/.local/bin:$PATH"\n'
current = shell_file.read_text() if shell_file.exists() else ''
if '# Agentscape terminal launcher' not in current and '.local/bin' not in current:
    with shell_file.open('a') as stream:
        stream.write(addition)
print(f'Installed {launcher}')
PY
.venv/bin/python -m agentscape setup --quick
printf '\nReady. Open a new terminal and type: agentscape\nIn this terminal: export PATH="$HOME/.local/bin:$PATH"\n'
