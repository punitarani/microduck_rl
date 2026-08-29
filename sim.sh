#!/usr/bin/env bash
# Run the Microduck simulator on macOS with the ONNX policies from the runtime repo.
#
#   ./sim.sh                 walk / stand / sit / ground-pick / roulade / kicks
#   ./sim.sh roller          roller-skate policies (wheels under the feet)
#   ./sim.sh -- --debug      anything after -- goes straight to infer_policy.py
#
# Two macOS-specific things this handles, both of which stop the viewer dead
# otherwise:
#   1. mujoco.viewer.launch_passive refuses to run under plain `python` on macOS
#      (the GUI must own the main thread) -> we invoke `mjpython`.
#   2. `mjpython` dlopens the venv interpreter and needs a shared libpython next
#      to it, which a uv-created venv does not have -> we symlink the one that
#      ships with the uv-managed CPython into .venv/lib/ (idempotent; re-run
#      after any `uv venv`).
set -euo pipefail
cd "$(dirname "$0")"

POLICIES="${MICRODUCK_POLICIES:-$(cd .. && pwd)/microduck/policies}"
[ -d "$POLICIES" ] || { echo "No policy directory at $POLICIES" >&2
                        echo "Set MICRODUCK_POLICIES=/path/to/microduck/policies" >&2; exit 1; }

# --- libpython shim -------------------------------------------------------
LIBDIR=$(uv run python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
ABI=$(uv run python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
DYLIB="libpython${ABI}.dylib"
if [ ! -e ".venv/lib/$DYLIB" ] && [ -e "$LIBDIR/$DYLIB" ]; then
    ln -sfn "$LIBDIR/$DYLIB" ".venv/lib/$DYLIB"
    echo "linked .venv/lib/$DYLIB -> $LIBDIR/$DYLIB"
fi

# --- policy selection -----------------------------------------------------
# A leading `--` means "default mode, rest is passthrough"; only a bare word
# ahead of it selects a mode.
MODE=walk
if [ $# -gt 0 ] && [ "$1" != "--" ]; then MODE="$1"; shift; fi
[ "${1:-}" = "--" ] && shift || true

case "$MODE" in
  roller)
    ARGS=(--roller
          --walking    "$POLICIES/roller.onnx"
          --ground-pick "$POLICIES/roller_crouch.onnx")
    ;;
  walk|"")
    ARGS=(--walking    "$POLICIES/alpha_walking.onnx"
          --standing   "$POLICIES/alpha_stand.onnx"
          --sitstand   "$POLICIES/alpha_sitstand.onnx"
          --ground-pick "$POLICIES/alpha_ground_pick.onnx"
          --roulade    "$POLICIES/roulade.onnx"
          --kick-left  "$POLICIES/ball_kick_left.onnx"
          --kick-right "$POLICIES/ball_kick_right.onnx")
    ;;
  *) echo "usage: $0 [walk|roller] [-- extra infer_policy.py args]" >&2; exit 2 ;;
esac

# --new-cmd-obs = the 61-D unified command contract these ONNX files were
# trained against, and the only one robotd builds. Without it the policies get
# the legacy 51-D command layout and move in ways nobody can explain.
exec uv run mjpython scripts/infer_policy.py "${ARGS[@]}" --new-cmd-obs "$@"
