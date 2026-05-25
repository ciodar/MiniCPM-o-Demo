#!/bin/bash
# Lightning AI Studio launcher for MiniCPM-o 4.5 Full-Duplex Demo
#
# Usage:
#   bash lightning_start.sh                          # uses all GPUs
#   CUDA_VISIBLE_DEVICES=0,1 bash lightning_start.sh  # specific GPUs
#
# This starts:
#   1. Worker processes (one per GPU)
#   2. Gateway (HTTP API + frontend on port from config, default 8006)
#   3. LitServe proxy (frontend on port 8000, proxies to gateway)
#
# The LitServe proxy is optional; set LITSERVE_PORT="" to skip it.
#
# torch.compile is controlled via config.json: "service": { "compile": true }

set -e

# ── Cleanup helper ──────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[Cleanup] Stopping all services..."
    for pid_file in tmp/*.pid; do
        [ -f "$pid_file" ] && kill "$(cat "$pid_file")" 2>/dev/null || true
    done
}

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
mkdir -p tmp

PYTHON="${PYTHON:-python}"
GATEWAY_PORT=$(PYTHONPATH=. $PYTHON -c "
import sys; sys.path.insert(0,'$PROJECT_DIR')
from config import get_config
cfg = get_config()
print(cfg.service.gateway_port)
" 2>/dev/null || echo "8006")

WORKER_BASE_PORT=$(PYTHONPATH=. $PYTHON -c "
import sys; sys.path.insert(0,'$PROJECT_DIR')
from config import get_config
cfg = get_config()
print(cfg.service.worker_base_port)
" 2>/dev/null || echo "22400")

LITSERVE_PORT="${LITSERVE_PORT:-8000}"

# ── GPU detection ────────────────────────────────────────────────────────────
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$NUM_GPUS" -eq 0 ]; then
        echo "[ERROR] No GPU detected"
        exit 1
    fi
    GPU_LIST=$(seq 0 $((NUM_GPUS - 1)) | tr '\n' ',' | sed 's/,$//')
else
    GPU_LIST="$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=$(echo "$GPU_LIST" | tr ',' '\n' | wc -l)
fi

echo "=================================================="
echo "  MiniCPMO45 Lightning Launcher"
echo "=================================================="
echo "  GPUs:       $GPU_LIST ($NUM_GPUS)"
echo "  Gateway:    http://localhost:$GATEWAY_PORT"
echo "  LitServe:   http://localhost:$LITSERVE_PORT"
echo "  Workers:    localhost:$WORKER_BASE_PORT ~ localhost:$((WORKER_BASE_PORT + NUM_GPUS - 1))"
echo "=================================================="

# ── Build mobile frontend (optional) ────────────────────────────────────────
if [ -f "$PROJECT_DIR/frontend/mobile/package.json" ] && [ "${SKIP_MOBILE_BUILD:-0}" != "1" ]; then
    echo "[Mobile] Building frontend/mobile → static/mobile ..."
    if command -v npm >/dev/null 2>&1; then
        (cd "$PROJECT_DIR/frontend/mobile" && npm run build:static 2>/dev/null) && echo "[Mobile] Build OK" || echo "[Mobile] Build skipped (not critical)"
    elif command -v bun >/dev/null 2>&1; then
        (cd "$PROJECT_DIR/frontend/mobile" && bun run --bun build:static 2>/dev/null) && echo "[Mobile] Build OK" || echo "[Mobile] Build skipped (not critical)"
    else
        echo "[Mobile] npm/bun not found, skipping mobile build"
    fi
fi

# ── Start workers ────────────────────────────────────────────────────────────
WORKER_ADDRS=""
GPU_IDX=0
for GPU_ID in $(echo "$GPU_LIST" | tr ',' ' '); do
    WPORT=$((WORKER_BASE_PORT + GPU_IDX))
    echo "[Worker $GPU_IDX] Starting on GPU $GPU_ID, port $WPORT..."
    nohup env CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH=. $PYTHON worker.py \
        --port $WPORT \
        --gpu-id $GPU_ID \
        --worker-index $GPU_IDX \
        > "tmp/worker_${GPU_IDX}.log" 2>&1 &
    echo $! > "tmp/worker_${GPU_IDX}.pid"
    if [ -z "$WORKER_ADDRS" ]; then
        WORKER_ADDRS="localhost:$WPORT"
    else
        WORKER_ADDRS="$WORKER_ADDRS,localhost:$WPORT"
    fi
    GPU_IDX=$((GPU_IDX + 1))
done

# ── Wait for workers ────────────────────────────────────────────────────────
echo ""
echo "Waiting for workers to load model (~30-90s)..."
sleep 5
for i in $(seq 0 $((NUM_GPUS - 1))); do
    WPORT=$((WORKER_BASE_PORT + i))
    RETRY=0
    MAX_RETRIES=3000
    while [ $RETRY -lt $MAX_RETRIES ]; do
        if curl -s "http://localhost:$WPORT/health" 2>/dev/null | PYTHONPATH=. $PYTHON -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('model_loaded') else 1)" 2>/dev/null; then
            echo "[Worker $i] Ready ✓ (port $WPORT)"
            break
        fi
        RETRY=$((RETRY + 1))
        sleep 2
    done
    if [ $RETRY -eq $MAX_RETRIES ]; then
        echo "[ERROR] Worker $i FAILED to start — check tmp/worker_${i}.log"
        cleanup
        exit 1
    fi
done

# ── Start gateway ────────────────────────────────────────────────────────────
echo ""
echo "[Gateway] Starting on port $GATEWAY_PORT..."
nohup env PYTHONPATH=. $PYTHON gateway.py \
    --port $GATEWAY_PORT \
    --workers "$WORKER_ADDRS" \
    > "tmp/gateway.log" 2>&1 &
echo $! > "tmp/gateway.pid"

GATEWAY_READY=0
for i in $(seq 1 30); do
    if curl -s "http://localhost:$GATEWAY_PORT/health" 2>/dev/null | PYTHONPATH=. $PYTHON -c "import sys,json; d=json.load(sys.stdin); exit(0)" 2>/dev/null; then
        echo "[Gateway] Ready ✓ (port $GATEWAY_PORT)"
        GATEWAY_READY=1
        break
    fi
    sleep 2
done
if [ "$GATEWAY_READY" -eq 0 ]; then
    echo "[ERROR] Gateway failed to start on port $GATEWAY_PORT — check tmp/gateway.log"
    cleanup
    exit 1
fi

# ── Start LitServe proxy (optional) ─────────────────────────────────────────
if [ -n "$LITSERVE_PORT" ]; then
    echo ""
    echo "[LitServe] Starting proxy on port $LITSERVE_PORT → gateway :$GATEWAY_PORT..."
    nohup env PYTHONPATH=. $PYTHON litserve_server.py \
        --port $LITSERVE_PORT \
        --gateway-host localhost \
        --gateway-port $GATEWAY_PORT \
        > "tmp/litserve.log" 2>&1 &
    echo $! > "tmp/litserve.pid"

    LITSERVE_READY=0
    for i in $(seq 1 15); do
        if curl -s "http://localhost:$LITSERVE_PORT/health" 2>/dev/null | PYTHONPATH=. $PYTHON -c "import sys,json; d=json.load(sys.stdin); exit(0)" 2>/dev/null; then
            echo "[LitServe] Ready ✓ (port $LITSERVE_PORT)"
            LITSERVE_READY=1
            break
        fi
        sleep 1
    done
    if [ "$LITSERVE_READY" -eq 0 ]; then
        echo "[ERROR] LitServe proxy failed to start on port $LITSERVE_PORT — check tmp/litserve.log"
        cleanup
        exit 1
    fi
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  All services started!"
echo "  Gateway:   http://localhost:$GATEWAY_PORT"
echo "  LitServe:  http://localhost:$LITSERVE_PORT"
echo "  Admin:     http://localhost:$GATEWAY_PORT/admin"
echo "  Workers:   $WORKER_ADDRS"
echo ""
echo "  Logs:"
echo "    Workers:  tmp/worker_*.log"
echo "    Gateway:  tmp/gateway.log"
echo "    LitServe: tmp/litserve.log"
echo ""
echo "  To stop:"
echo "    kill \$(cat tmp/*.pid 2>/dev/null) 2>/dev/null"
echo "=================================================="
