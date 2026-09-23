#!/bin/sh
# Run the portable MLX benchmark without sudo, Homebrew, or Git.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OWNED="$ROOT/.no-admin-bench"
TOOLS="$OWNED/tools"
PYTHONS="$OWNED/python"
CACHE="$OWNED/cache"
VENV="$OWNED/venv"

usage() {
    printf '%s\n' \
        "Usage: ./no_admin_bench.sh [--cleanup]" \
        "" \
        "With no option, uses an available Python 3.9+ or downloads a private fallback," \
        "then runs quick_bench.py --big. No administrator password is used." \
        "" \
        "--cleanup  Remove only .no-admin-bench (tools, Python, environment, and cache)." \
        "           Saved benchmark reports are preserved."
}

case "${1:-}" in
    "") ;;
    --cleanup)
        if [ -d "$OWNED" ]; then
            rm -rf -- "$OWNED"
            printf 'Removed %s\n' "$OWNED"
        else
            printf 'Nothing to remove: %s does not exist.\n' "$OWNED"
        fi
        exit 0
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    printf 'This benchmark requires an Apple Silicon Mac.\n' >&2
    exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
    printf 'curl is required to download the user-local Python toolchain.\n' >&2
    exit 1
fi

mkdir -p "$TOOLS" "$PYTHONS" "$CACHE" "$ROOT/benchmarks"
READY=0
if command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 9))' >/dev/null 2>&1; then
    printf 'Using the available %s to create a private environment...\n' "$(python3 --version 2>&1)"
    if python3 -m venv "$VENV" && "$VENV/bin/python" -m pip install -e "$ROOT"; then
        READY=1
    else
        printf 'The available Python could not prepare the environment; using the private fallback.\n'
        rm -rf -- "$VENV"
    fi
fi

if [ "$READY" -eq 0 ]; then
    UV="$TOOLS/uv"
    if [ ! -x "$UV" ]; then
        printf 'Downloading uv into %s (no sudo; no shell-profile changes)...\n' "$TOOLS"
        curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$TOOLS" sh
    fi
    export UV_CACHE_DIR="$CACHE"
    export UV_PYTHON_INSTALL_DIR="$PYTHONS"
    printf 'Preparing a private Python 3.12 fallback...\n'
    "$UV" venv --python 3.12 "$VENV"
    "$UV" pip install --python "$VENV/bin/python" -e "$ROOT"
fi

STAMP=$(date '+%Y%m%d-%H%M%S')
REPORT="$ROOT/benchmarks/no-admin-$STAMP.txt"

{
    printf 'mlx-kmeans no-admin benchmark\n'
    printf 'date: %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
    printf 'source directory: %s\n' "$ROOT"
    if command -v git >/dev/null 2>&1 && git -C "$ROOT" rev-parse HEAD >/dev/null 2>&1; then
        printf 'git commit: %s\n' "$(git -C "$ROOT" rev-parse HEAD)"
    fi
    printf 'source fingerprints:\n'
    shasum -a 256 "$ROOT/no_admin_bench.sh" "$ROOT/quick_bench.py" "$ROOT/mlx_kmeans/core.py" "$ROOT/pyproject.toml"
    printf '\n'
    "$VENV/bin/python" "$ROOT/quick_bench.py" --big
} 2>&1 | tee "$REPORT"

printf '\nSaved report: %s\n' "$REPORT"
printf 'To remove the private runtime and cache: ./no_admin_bench.sh --cleanup\n'
