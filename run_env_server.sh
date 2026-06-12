#!/bin/bash
# Start/stop the validator's MCTS env-server locally so SMOKE_MODE=game can run.
#
# Image: diagonalge/mcts-api:latest -- the official validator env-server (matches
# MCTS_API_DOCKER_IMAGE in gradients-ai/G.O.D core/constants.py). One server hosts
# all games; the game is selected by task_id in each /reset call.
#
# Usage:
#   bash run_env_server.sh start    # pulls + starts, prints the URL(s) to use
#   bash run_env_server.sh stop
#
# Knobs (env): NUM_SERVERS (default 1), BASE_PORT (default 8000), MCTS_IMAGE.
set -e

NUM_SERVERS="${NUM_SERVERS:-1}"
BASE_PORT="${BASE_PORT:-8000}"
IMAGE="${MCTS_IMAGE:-diagonalge/mcts-api:latest}"
ACTION="${1:-start}"

if [ "$ACTION" = "stop" ]; then
  for i in $(seq 0 $((NUM_SERVERS - 1))); do docker rm -f "mcts-env-$i" 2>/dev/null || true; done
  echo "stopped env server(s)"
  exit 0
fi

echo "pulling $IMAGE ..."
docker pull "$IMAGE" || echo "WARN: pull failed (using local copy if present)"

urls=()
for i in $(seq 0 $((NUM_SERVERS - 1))); do
  port=$((BASE_PORT + i))
  name="mcts-env-$i"
  docker rm -f "$name" >/dev/null 2>&1 || true
  # Published to a host port so the --network host trainer reaches it on localhost.
  docker run -d --name "$name" -p "${port}:8000" "$IMAGE" >/dev/null
  echo "started $name -> http://localhost:${port}"
  urls+=("http://localhost:${port}")
done

JOINED=$(IFS=,; echo "${urls[*]}")
echo ""
echo "Env server ready. Set these in run_rest_smoke_docker.sh, then run it:"
echo "  ENVIRONMENT_SERVER_URLS=\"${JOINED}\""
echo "  SMOKE_MODE=\"game\""
echo ""
echo "Quick reachability check:"
echo "  curl -s -X POST http://localhost:${BASE_PORT}/reset -H 'Content-Type: application/json' \\"
echo "    -d '{\"task_id\":200000001,\"seed\":1,\"opponent\":\"mcts\",\"mcts_max_simulations\":5,\"mcts_num_rollouts\":1}'"
