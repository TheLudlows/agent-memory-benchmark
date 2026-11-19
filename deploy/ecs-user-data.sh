#!/bin/bash
set -euo pipefail

AMB_IMAGE="${AMB_IMAGE:-ghcr.io/theludlows/agent-memory-benchmark:workbench}"
AMB_PULL="${AMB_PULL:-always}"

if ! command -v docker >/dev/null; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io
fi
systemctl enable --now docker

docker run --detach --init --pull "$AMB_PULL" \
  --name amb-workbench --restart unless-stopped \
  --add-host=host.docker.internal:host-gateway \
  --publish 127.0.0.1:8090:8090 \
  --volume amb-workbench-results:/app/outputs \
  --volume amb-workbench-config:/app/workbench-config \
  "$AMB_IMAGE"
