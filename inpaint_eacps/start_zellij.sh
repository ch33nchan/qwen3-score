#!/bin/bash
# Start inpaint_eacps pipeline in zellij session

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
SESSION_NAME="inpaint_eacps"
FROM_FILE="project-label.json"
OUTPUT_DIR="inpaint_eacps"
DEVICE="cuda:0"
GEMINI_KEY="${GEMINI_API_KEY}"

# Parse arguments
TASK_IDS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --task_id)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                TASK_IDS+=("$1")
                shift
            done
            ;;
        --from_file)
            FROM_FILE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --gemini_api_key)
            GEMINI_KEY="$2"
            shift 2
            ;;
        --k_global)
            K_GLOBAL="$2"
            shift 2
            ;;
        --m_global)
            M_GLOBAL="$2"
            shift 2
            ;;
        --k_local)
            K_LOCAL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ ${#TASK_IDS[@]} -eq 0 ]; then
    echo "Error: --task_id is required"
    echo "Usage: $0 --task_id ID1 ID2 ... [--from_file FILE] [--output_dir DIR] [--device DEVICE] [--gemini_api_key KEY]"
    exit 1
fi

# Build command
CMD="cd '$PROJECT_ROOT' && source .venv/bin/activate && python3 inpaint_eacps/run.py"
CMD="$CMD --from_file '$FROM_FILE'"
CMD="$CMD --output_dir '$OUTPUT_DIR'"
CMD="$CMD --device '$DEVICE'"

if [ -n "$GEMINI_KEY" ]; then
    CMD="$CMD --gemini_api_key '$GEMINI_KEY'"
fi

if [ -n "$K_GLOBAL" ]; then
    CMD="$CMD --k_global $K_GLOBAL"
fi

if [ -n "$M_GLOBAL" ]; then
    CMD="$CMD --m_global $M_GLOBAL"
fi

if [ -n "$K_LOCAL" ]; then
    CMD="$CMD --k_local $K_LOCAL"
fi

CMD="$CMD --task_id ${TASK_IDS[@]}"

echo -e "${GREEN}Starting zellij session: ${SESSION_NAME}${NC}"
echo -e "${YELLOW}Task IDs: ${TASK_IDS[*]}${NC}"
echo -e "${YELLOW}Output directory: ${OUTPUT_DIR}${NC}"
echo -e "${YELLOW}Device: ${DEVICE}${NC}"
echo ""

# Check if session exists
if zellij list-sessions 2>/dev/null | grep -q "^${SESSION_NAME}$"; then
    echo -e "${YELLOW}Session ${SESSION_NAME} already exists.${NC}"
    echo "Options:"
    echo "  1. Attach to existing session"
    echo "  2. Kill and recreate session"
    read -p "Choice [1/2]: " choice
    if [ "$choice" = "2" ]; then
        zellij kill-session "${SESSION_NAME}"
        echo "Creating new session..."
        zellij attach --create "${SESSION_NAME}" -- "${SHELL}" -c "eval '$CMD'"
    else
        zellij attach "${SESSION_NAME}"
    fi
else
    echo "Creating new session ${SESSION_NAME}..."
    zellij attach --create "${SESSION_NAME}" -- "${SHELL}" -c "eval '$CMD'"
fi
