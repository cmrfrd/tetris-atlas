#!/usr/bin/env bash

set -e  # Exit on any command failure

TENSORBOARD_DIR="/tmp/tensorboard"

# Remove existing directory if it exists
if [ -d "$TENSORBOARD_DIR" ]; then
    echo "Removing existing tensorboard directory: $TENSORBOARD_DIR"
    rm -rf "$TENSORBOARD_DIR"
fi

echo "Installing tensorboard protos to: $TENSORBOARD_DIR"
cd /tmp

TAG=2.20.0
git clone --filter=tree:0 --no-checkout --depth 1 \
  --branch "$TAG" https://github.com/tensorflow/tensorboard.git
cd tensorboard

git sparse-checkout init --cone
git sparse-checkout set tensorboard/compat/proto
git checkout "$TAG"

echo "Downloaded tensorboard protos to: $TENSORBOARD_DIR"
