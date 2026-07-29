#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="${1:-$(pwd)}"
service_user="${ATLAS_SERVICE_USER:-ubuntu}"
venv_dir="/opt/atlas-metadata/.venv"

if [[ ! -f "${project_dir}/requirements.txt" || ! -f "${project_dir}/app.py" ]]; then
  echo "Run bootstrap.sh from the metadata project directory or pass that directory as argument 1." >&2
  exit 2
fi

apt-get update
apt-get install -y --no-install-recommends python3-venv python3-pip openjdk-17-jre-headless git
python3 -m venv "${venv_dir}"
"${venv_dir}/bin/pip" install --upgrade pip wheel
"${venv_dir}/bin/pip" install -r "${project_dir}/requirements.txt"

if [[ ! -f /etc/atlas-metadata.env ]]; then
  install -m 600 /dev/null /etc/atlas-metadata.env
  tee /etc/atlas-metadata.env >/dev/null <<'EOF'
METADATA_PROVIDER=heuristic
METADATA_INDEX_BACKEND=hash
METADATA_EXTRACTOR=native
LLAMA_QUANTIZATION=4bit
LLAMA_DEVICE=auto
LLAMA_GPU_MEMORY_GIB=20
LLAMA_CPU_MEMORY_GIB=48
LLAMA_OFFLOAD_DIR=models/offload
EOF
fi
install -d -o "${service_user}" -g "${service_user}" "${project_dir}/artifacts" "${project_dir}/models/offload"
chown -R "${service_user}:${service_user}" /opt/atlas-metadata "${project_dir}"

tee /etc/systemd/system/atlas-metadata.service >/dev/null <<EOF
[Unit]
Description=Atlas Metadata Studio
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${service_user}
WorkingDirectory=${project_dir}
EnvironmentFile=-/etc/atlas-metadata.env
ExecStart=${venv_dir}/bin/streamlit run app.py --server.address=0.0.0.0 --server.port=8501
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now atlas-metadata.service
systemctl --no-pager status atlas-metadata.service || true
nvidia-smi || echo "No NVIDIA runtime detected; keep heuristic mode until a compatible GPU is attached."
