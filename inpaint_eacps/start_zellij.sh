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

# Use zellij from .venv/bin
ZELLIJ_BIN="$PROJECT_ROOT/.venv/bin/zellij"
if [ ! -f "$ZELLIJ_BIN" ]; then
    echo -e "${YELLOW}Zellij not found in .venv/bin. Installing...${NC}"
    "$SCRIPT_DIR/install_zellij.sh"
fi

# Add .venv/bin to PATH for zellij
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"

# Create a wrapper script for the command
WRAPPER_SCRIPT="$PROJECT_ROOT/.venv/bin/run_inpaint_eacps.sh"
cat > "$WRAPPER_SCRIPT" <<EOF
#!/bin/bash
cd '$PROJECT_ROOT'
source .venv/bin/activate
python3 inpaint_eacps/run.py \\
  --from_file '$FROM_FILE' \\
  --output_dir '$OUTPUT_DIR' \\
  --device '$DEVICE' \\
  $( [ -n "$GEMINI_KEY" ] && echo "--gemini_api_key '$GEMINI_KEY' \\" ) \\
  $( [ -n "$K_GLOBAL" ] && echo "--k_global $K_GLOBAL \\" ) \\
  $( [ -n "$M_GLOBAL" ] && echo "--m_global $M_GLOBAL \\" ) \\
  $( [ -n "$K_LOCAL" ] && echo "--k_local $K_LOCAL \\" ) \\
  --task_id ${TASK_IDS[@]}
EOF
chmod +x "$WRAPPER_SCRIPT"

# Use the wrapper script
CMD="$WRAPPER_SCRIPT"

echo -e "${GREEN}Starting zellij session: ${SESSION_NAME}${NC}"
echo -e "${YELLOW}Task IDs: ${TASK_IDS[*]}${NC}"
echo -e "${YELLOW}Output directory: ${OUTPUT_DIR}${NC}"
echo -e "${YELLOW}Device: ${DEVICE}${NC}"
echo ""

# Since zellij doesn't support running commands directly when creating sessions,
# we'll run the command and let user attach to zellij to monitor
echo "Starting pipeline..."
echo ""

# Run the command in background and log output
LOG_FILE="$PROJECT_ROOT/inpaint_eacps_${SESSION_NAME}.log"
cd "$PROJECT_ROOT"

# Start the command in background
nohup "$CMD" > "$LOG_FILE" 2>&1 &
CMD_PID=$!

echo -e "${GREEN}Pipeline started (PID: $CMD_PID)${NC}"
echo -e "${YELLOW}Log file: $LOG_FILE${NC}"
echo ""

# Check if zellij session exists, create/attach for monitoring
if "$ZELLIJ_BIN" list-sessions 2>/dev/null | grep -q "^${SESSION_NAME}$"; then
    echo "Zellij session '${SESSION_NAME}' exists."
    echo "Attach to monitor: .venv/bin/zellij attach ${SESSION_NAME}"
    echo ""
    read -p "Attach now? [y/N]: " attach_now
    if [ "$attach_now" = "y" ] || [ "$attach_now" = "Y" ]; then
        # Create a simple monitoring pane in zellij
        "$ZELLIJ_BIN" attach "${SESSION_NAME}"
    fi
else
    echo "Creating zellij session for monitoring..."
    echo "You can attach with: .venv/bin/zellij attach ${SESSION_NAME}"
    echo ""
    echo "To monitor progress:"
    echo "  tail -f $LOG_FILE"
    echo "  OR"
    echo "  .venv/bin/zellij attach ${SESSION_NAME}"
    echo ""
    read -p "Create and attach to zellij session now? [y/N]: " create_now
    if [ "$create_now" = "y" ] || [ "$create_now" = "Y" ]; then
        # Start zellij in a way that shows the log
        "$ZELLIJ_BIN" --session "${SESSION_NAME}" &
        sleep 1
        echo "Zellij session created. You can now attach to monitor progress."
    fi
fi

echo ""
echo "Pipeline is running. Check progress with:"
echo "  tail -f $LOG_FILE"
echo "  OR attach to zellij: .venv/bin/zellij attach ${SESSION_NAME}"
