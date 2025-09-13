#!/usr/bin/env bash

set -e  # Exit on any command failure

validate_path() {
    local path="$1"
    local name="$2"
    
    if [ -z "$path" ]; then
        echo "Error: $name is empty"
        exit 1
    fi
    
    # Must start with /tmp/
    if [[ "$path" != /tmp/* ]]; then
        echo "Error: $name must be under /tmp/ for safety: '$path'"
        exit 1
    fi
    
    # Must not be exactly /tmp or /
    if [ "$path" = "/tmp" ] || [ "$path" = "/" ]; then
        echo "Error: $name cannot be root directories: '$path'"
        exit 1
    fi
}

cd /tmp

TAG=2.20.0
ARCHIVE_URL="https://github.com/tensorflow/tensorboard/archive/refs/tags/${TAG}.tar.gz"
ARCHIVES_DIR="/tmp/tensorboard-archives"
ARCHIVE_PATH="${ARCHIVES_DIR}/tensorboard-${TAG}.tar.gz"
TENSORBOARD_DIR="/tmp/tensorboard"

# Validate all paths for safety
validate_path "$ARCHIVES_DIR" "ARCHIVES_DIR"
validate_path "$TENSORBOARD_DIR" "TENSORBOARD_DIR"

echo "Installing tensorboard protos ${TAG} to: $TENSORBOARD_DIR"

mkdir -p "$ARCHIVES_DIR"
mkdir -p "$TENSORBOARD_DIR"

if [ -f "$ARCHIVE_PATH" ]; then
    echo "Using cached archive: $ARCHIVE_PATH"
else
    echo "Downloading ${ARCHIVE_URL}"
    curl -fL "$ARCHIVE_URL" -o "$ARCHIVE_PATH"
    if [ ! -f "$ARCHIVE_PATH" ]; then
        echo "Error: Failed to download archive"
        exit 1
    fi
fi

# Clean destination directory to ensure consistent state
echo "Cleaning destination directory: $TENSORBOARD_DIR"
rm -rf "$TENSORBOARD_DIR"

old_umask=$(umask)
umask 022
echo "Extracting tarball to $TENSORBOARD_DIR..."
mkdir -p "$TENSORBOARD_DIR"
tar -xzf "$ARCHIVE_PATH" -C "$TENSORBOARD_DIR" --strip-components=1
umask "$old_umask"

# Verify the proto directory exists
PROTO_DIR="$TENSORBOARD_DIR/tensorboard/compat/proto"
if [ ! -d "$PROTO_DIR" ]; then
    echo "Error: Expected proto directory not found: $PROTO_DIR"
    exit 1
fi

echo "Downloaded tensorboard protos to: $TENSORBOARD_DIR"
