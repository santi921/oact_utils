"""Custom Parsl launchers used by workflow submission utilities."""

from __future__ import annotations

from pathlib import Path

from parsl.launchers.base import Launcher


class PbsdshLauncher(Launcher):
    """Launch Parsl managers across unique allocated PBS hosts via ``pbsdsh``.

    OpenPBS indexes ``pbsdsh -n`` by vnode entry in ``PBS_NODEFILE``, not by
    unique hostname. This launcher computes the first 0-based vnode index for
    each unique hostname and launches one Parsl manager on each such index,
    mirroring the "one manager per node" behavior we want from multi-node
    allocations while avoiding an MPI parent launcher.
    """

    def __init__(self, helper_dir: str | Path, debug: bool = True):
        super().__init__(debug=debug)
        self.helper_dir = str(Path(helper_dir).resolve())

    def __call__(self, command: str, tasks_per_node: int, nodes_per_block: int) -> str:
        debug_num = int(self.debug)

        return """set -e
export CORES=$(getconf _NPROCESSORS_ONLN)
[[ "{debug}" == "1" ]] && echo "Found cores : $CORES"
[[ "{debug}" == "1" ]] && echo "Found nodes : {nodes_per_block}"
TASKS_PER_NODE={tasks_per_node}
HELPER_DIR="{helper_dir}"
mkdir -p "$HELPER_DIR"

PARSL_PBSDH_CMD="$HELPER_DIR/cmd_$JOBNAME.sh"
cat << PBSDH_CMD_EOF > "$PARSL_PBSDH_CMD"
{command}
PBSDH_CMD_EOF
chmod a+x "$PARSL_PBSDH_CMD"

PARSL_PBSDH_WRAPPER="$HELPER_DIR/pbsdsh_$JOBNAME.sh"
cat << 'PBSDH_WRAPPER_EOF' > "$PARSL_PBSDH_WRAPPER"
set -e
CMD_PATH="$1"
TASKS_PER_NODE="$2"
PIDS=""
for COUNT in $(seq 1 1 "$TASKS_PER_NODE"); do
    "$CMD_PATH" &
    PIDS="$PIDS $!"
done

FAIL=0
for PID in $PIDS; do
    wait "$PID" || FAIL=1
done
exit "$FAIL"
PBSDH_WRAPPER_EOF
chmod a+x "$PARSL_PBSDH_WRAPPER"
trap 'rm -f "$PARSL_PBSDH_CMD" "$PARSL_PBSDH_WRAPPER"' EXIT

if [ -n "${{PBS_NODEFILE:-}}" ] && [ -f "$PBS_NODEFILE" ]; then
    NODE_INDICES=$(awk '!seen[$1]++ {{print NR-1}}' "$PBS_NODEFILE")
else
    NODE_INDICES="0"
fi

[[ "{debug}" == "1" ]] && echo "Using PBS vnode indices: $NODE_INDICES"
PIDS=""
for NODE_INDEX in $NODE_INDICES; do
    pbsdsh -n "$NODE_INDEX" -- /bin/bash "$PARSL_PBSDH_WRAPPER" "$PARSL_PBSDH_CMD" "$TASKS_PER_NODE" &
    PIDS="$PIDS $!"
done

FAIL=0
for PID in $PIDS; do
    wait "$PID" || FAIL=1
done

[[ "{debug}" == "1" ]] && echo "All workers done"
exit "$FAIL"
""".format(
            command=command,
            tasks_per_node=tasks_per_node,
            nodes_per_block=nodes_per_block,
            helper_dir=self.helper_dir,
            debug=debug_num,
        )
