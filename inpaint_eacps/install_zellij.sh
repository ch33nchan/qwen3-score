#!/bin/bash
# Install zellij locally in .venv/bin

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Create .venv/bin if it doesn't exist
VENV_BIN="$PROJECT_ROOT/.venv/bin"
mkdir -p "$VENV_BIN"

# Detect architecture
ARCH=$(uname -m)
OS=$(uname -s | tr '[:upper:]' '[:lower:]')

if [ "$ARCH" = "x86_64" ]; then
    ARCH_NAME="x86_64"
elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    ARCH_NAME="aarch64"
else
    echo "Error: Unsupported architecture: $ARCH"
    exit 1
fi

if [ "$OS" = "linux" ]; then
    OS_NAME="unknown-linux-musl"
elif [ "$OS" = "darwin" ]; then
    OS_NAME="apple-darwin"
else
    echo "Error: Unsupported OS: $OS"
    exit 1
fi

ZELLIJ_BINARY="zellij-${ARCH_NAME}-${OS_NAME}"
ZELLIJ_URL="https://github.com/zellij-org/zellij/releases/latest/download/${ZELLIJ_BINARY}.tar.gz"
ZELLIJ_PATH="$VENV_BIN/zellij"

echo -e "${GREEN}Installing zellij to .venv/bin/${NC}"
echo -e "${YELLOW}Architecture: ${ARCH_NAME}-${OS_NAME}${NC}"
echo ""

# Download and extract
TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"

echo "Downloading zellij..."
curl -L "$ZELLIJ_URL" -o zellij.tar.gz

echo "Extracting..."
tar -xzf zellij.tar.gz

# Move to .venv/bin
if [ -f "$ZELLIJ_BINARY/zellij" ]; then
    mv "$ZELLIJ_BINARY/zellij" "$ZELLIJ_PATH"
    chmod +x "$ZELLIJ_PATH"
    echo -e "${GREEN}✓ Installed zellij to $ZELLIJ_PATH${NC}"
elif [ -f "zellij" ]; then
    mv "zellij" "$ZELLIJ_PATH"
    chmod +x "$ZELLIJ_PATH"
    echo -e "${GREEN}✓ Installed zellij to $ZELLIJ_PATH${NC}"
else
    echo "Error: Could not find zellij binary in archive"
    exit 1
fi

# Cleanup
cd "$PROJECT_ROOT"
rm -rf "$TMP_DIR"

# Verify installation
if [ -f "$ZELLIJ_PATH" ]; then
    VERSION=$("$ZELLIJ_PATH" --version 2>/dev/null || echo "installed")
    echo -e "${GREEN}✓ zellij installed successfully${NC}"
    echo -e "${YELLOW}Version: $VERSION${NC}"
    echo ""
    echo "To use zellij, run:"
    echo "  .venv/bin/zellij --version"
    echo ""
    echo "Or add to PATH:"
    echo "  export PATH=\"\$PATH:$VENV_BIN\""
else
    echo "Error: Installation failed"
    exit 1
fi
