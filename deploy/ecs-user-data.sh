#!/bin/bash
set -euo pipefail

AMB_IMAGE="${AMB_IMAGE:-ghcr.io/theludlows/agent-memory-benchmark:workbench}"
AMB_PULL="${AMB_PULL:-always}"
AMB_WORKBENCH_BIND="${AMB_WORKBENCH_BIND:-127.0.0.1}"

if ! command -v docker >/dev/null; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io
fi
systemctl enable --now docker

docker run --detach --init --pull "$AMB_PULL" \
  --name amb-workbench --restart unless-stopped \
  --add-host=host.docker.internal:host-gateway \
  --publish "$AMB_WORKBENCH_BIND:8090:8090" \
  --env "AMB_WORKBENCH_ALLOWED_HOSTS=${AMB_WORKBENCH_ALLOWED_HOSTS:-}" \
  --volume amb-workbench-results:/app/outputs \
  --volume amb-workbench-config:/app/workbench-config \
  "$AMB_IMAGE"
