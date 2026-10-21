set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$task_root"
export UV_CACHE_DIR="$task_root/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$task_root/.venv-wsl"
if [ -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
    exec "$UV_PROJECT_ENVIRONMENT/bin/python" -m fracture_world "$@"
fi
exec uv run --locked --python 3.11 python -m fracture_world "$@"
