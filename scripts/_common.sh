# Sourced by the other scripts: project root, venv python, Java for PySpark.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "No .venv found. Set it up first: see README 'Setup'."; exit 1; }
export JAVA_HOME="${JAVA_HOME:-$HOME/.local/jdk}"
export PATH="$JAVA_HOME/bin:$PATH"
