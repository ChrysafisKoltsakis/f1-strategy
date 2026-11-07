#!/usr/bin/env bash
# Local Kafka broker for the live replay/dashboard (needs Docker).
#   scripts/kafka.sh start|stop|status
set -euo pipefail
NAME=f1-kafka
case "${1:-start}" in
  start)
    docker info >/dev/null 2>&1 || { echo "Docker isn't running. Start Docker Desktop (with WSL integration on) and retry."; exit 1; }
    if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
      echo "Kafka already running."
    elif docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
      docker start "$NAME" >/dev/null; echo "Kafka started."
    else
      docker run -d --name "$NAME" -p 9092:9092 \
        -e KAFKA_NODE_ID=1 -e KAFKA_PROCESS_ROLES=broker,controller \
        -e KAFKA_LISTENERS=PLAINTEXT://:9092,CONTROLLER://:9093 \
        -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
        -e KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER \
        -e KAFKA_CONTROLLER_QUORUM_VOTERS=1@localhost:9093 \
        -e KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT \
        -e CLUSTER_ID=f1strategy0001 \
        apache/kafka:latest >/dev/null
      echo "Kafka created and started."
    fi
    printf "Waiting for port 9092"
    for _ in $(seq 1 60); do
      (echo > /dev/tcp/localhost/9092) 2>/dev/null && { echo " ready."; exit 0; }
      printf "."; sleep 1
    done
    echo " timed out (check: docker logs $NAME)"; exit 1 ;;
  stop)   docker stop "$NAME" >/dev/null && echo "Kafka stopped." ;;
  status) docker ps -a --filter "name=$NAME" --format '{{.Names}}: {{.Status}}' ;;
  *) echo "usage: scripts/kafka.sh start|stop|status"; exit 1 ;;
esac
