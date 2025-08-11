
#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Get the directory of the current script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Launch the first backend server on port 9090 with 4 workers.
echo "Starting the first backend server on port 9090 with 4 workers..."
"$SCRIPT_DIR/launch_backend_service.sh" 9090 4 &

# Launch the second backend server on port 9091 with 2 workers.
echo "Starting the second backend server on port 9091 with 2 workers..."
"$SCRIPT_DIR/launch_backend_service.sh" 9091 2 &

echo "Both backend servers have been launched."
