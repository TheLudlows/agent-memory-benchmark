#!/usr/bin/env bash
set -euo pipefail

revision=${1:-}
output_dir=${2:-}
if [[ ! $revision =~ ^[a-f0-9]{40}$ || -z $output_dir ]]; then
    echo "Usage: bash $0 <published-workbench-commit-sha> <output-directory>" >&2
    exit 2
fi

repo=${GITHUB_REPOSITORY:-TheLudlows/agent-memory-benchmark}
workbench="ghcr.io/${repo,,}:workbench-sha-$revision"
ubuntu_index=https://cloud-images.ubuntu.com/noble/current
base_image=noble-server-cloudimg-amd64.img
filename="amb-ecs-ubuntu-24.04-amd64-${revision:0:12}.qcow2"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

mkdir -p "$output_dir"
output_dir=$(realpath "$output_dir")
scratch=$(mktemp -d)
cd "$scratch"
vm_pid=
serial_log=
cleanup() {
    status=$?
    if [[ -n $vm_pid ]]; then
        kill "$vm_pid" 2>/dev/null || true
        wait "$vm_pid" 2>/dev/null || true
    fi
    if (( status != 0 )) && [[ -n $serial_log && -f $serial_log ]]; then
        tail -n 120 "$serial_log" >&2
    fi
    rm -rf "$scratch"
}
trap cleanup EXIT

create_seed() {
    local name=$1
    ssh-keygen -q -t ed25519 -N '' -f "$name.key"
    {
        printf '#cloud-config\nssh_pwauth: false\nssh_authorized_keys:\n'
        printf '  - %s\n' "$(cat "$name.key.pub")"
    } > "$name.user-data"
    printf 'instance-id: amb-%s\nlocal-hostname: amb-%s\n' "$name" "$name" > "$name.meta-data"
    cloud-localds "$name.seed" "$name.user-data" "$name.meta-data"
}

boot_vm() {
    local name=$1 disk=$2 port=$3 network_restriction=$4
    local cpu=max accel=tcg
    if [[ -r /dev/kvm && -w /dev/kvm ]]; then
        cpu=host
        accel=kvm
    fi
    serial_log="$scratch/$name.console.log"
    qemu-system-x86_64 -machine "q35,accel=$accel" -cpu "$cpu" -smp 4 -m 4096 \
        -display none -monitor none -serial "file:$serial_log" -no-reboot \
        -drive "file=$disk,if=virtio,format=qcow2,discard=unmap" \
        -cdrom "$name.seed" \
        -nic "user,model=virtio-net-pci,restrict=$network_restriction,hostfwd=tcp:127.0.0.1:$port-:22" &
    vm_pid=$!

    local -a connection=(-i "$name.key" -o BatchMode=yes -o ConnectTimeout=3
        -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)
    guest_ssh=(ssh "${connection[@]}" -p "$port" ubuntu@127.0.0.1)
    guest_scp=(scp "${connection[@]}" -P "$port")
    for attempt in {1..180}; do
        if "${guest_ssh[@]}" true 2>/dev/null; then
            "${guest_ssh[@]}" 'timeout 300 cloud-init status --wait'
            return
        fi
        kill -0 "$vm_pid"
        sleep 2
    done
    echo "SSH did not become available for $name" >&2
    return 1
}

curl -fsSLo SHA256SUMS.ubuntu --retry 3 "$ubuntu_index/SHA256SUMS"
curl -fsSLo "$base_image" --retry 3 "$ubuntu_index/$base_image"
awk -v image="$base_image" '$2 == image || $2 == "*"image { print }' SHA256SUMS.ubuntu | sha256sum --check
qemu-img create -f qcow2 -F qcow2 -b "$base_image" provision.qcow2 30G

docker pull --platform linux/amd64 "$workbench"
test "$(docker image inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' "$workbench")" = "$revision"
docker tag "$workbench" amb-workbench:local
docker save amb-workbench:local | gzip -1 > workbench.tar.gz

create_seed provision
boot_vm provision provision.qcow2 2222 off
"${guest_scp[@]}" workbench.tar.gz "$script_dir/ecs-user-data.sh" ubuntu@127.0.0.1:/var/tmp/
"${guest_ssh[@]}" sudo bash -se <<'PROVISION'
export DEBIAN_FRONTEND=noninteractive
apt-get -o Acquire::Retries=3 update
apt-get -o Acquire::Retries=3 install -y --no-install-recommends docker.io
systemctl enable --now docker
docker load -i /var/tmp/workbench.tar.gz
AMB_IMAGE=amb-workbench:local AMB_PULL=never bash /var/tmp/ecs-user-data.sh
rm -f /var/tmp/workbench.tar.gz /var/tmp/ecs-user-data.sh
apt-get clean
rm -rf /var/lib/apt/lists/*
rm -f /home/ubuntu/.ssh/authorized_keys /etc/ssh/ssh_host_*
cloud-init clean --logs --machine-id --seed
fstrim /
systemd-run --on-active=3s --unit=amb-image-poweroff /usr/bin/systemctl poweroff
PROVISION
# Waiting for QEMU guarantees the guest has flushed the filesystem before export.
wait "$vm_pid"
vm_pid=

qemu-img convert -c -O qcow2 provision.qcow2 "$filename"
qemu-img check "$filename"

qemu-img create -f qcow2 -F qcow2 -b "$filename" smoke.qcow2
create_seed smoke
boot_vm smoke smoke.qcow2 2223 on
"${guest_ssh[@]}" bash -se <<'SMOKE'
set -o pipefail
trap 'sudo docker ps -a >&2; sudo docker logs --tail 80 amb-workbench >&2' ERR
systemctl is-enabled docker
systemctl is-active docker
test "$(sudo docker inspect --format '{{.HostConfig.RestartPolicy.Name}}' amb-workbench)" = unless-stopped
python3 - <<'PY'
import json
import time
from urllib.request import urlopen

base = 'http://127.0.0.1:8090'
deadline = time.monotonic() + 180
while True:
    try:
        with urlopen(f'{base}/api/config', timeout=3) as response:
            config = json.load(response)
        break
    except OSError:
        if time.monotonic() >= deadline:
            raise
        time.sleep(2)
assert set(config['datasets']) == {'locomo', 'beam', 'longmemeval', 'lifebench-v2', 'precisionmembench'}
with urlopen(base, timeout=3) as response:
    assert b'evaluation-name' in response.read()
PY
sudo docker exec -i amb-workbench python - <<'PY'
from memory_bench.dataset import get_dataset
from memory_bench.utils import count_tokens

assert count_tokens('Agent Memory Benchmark') > 0
splits = {
    'locomo': {'locomo10': 1540},
    'beam': {'100k': 400, '500k': 700, '1m': 700, '10m': 200},
    'longmemeval': {'s': 500},
    'lifebench-v2': {'zh': 3380},
    'precisionmembench': {'single-turn': 77},
}
for name, sizes in splits.items():
    dataset = get_dataset(name)
    for split, expected in sizes.items():
        queries = dataset.load_queries(split)
        assert len(queries) == expected, (name, split, len(queries))
        documents = dataset.load_documents(split, user_ids={queries[0].user_id}, limit=1)
        assert documents and documents[0].content.strip(), (name, split)
        print(f'{name}/{split}: {len(queries)} questions')
PY
SMOKE
sha256sum "$filename" > SHA256SUMS
mv "$filename" SHA256SUMS "$output_dir/"
