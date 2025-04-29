# server.py - Simplified Entry Point
import logging
import sys

# Import necessary components from the new package structure
try:
    # Import the logging setup function
    from okx_mcp.config import setup_logging
    # Import the pre-configured mcp instance from the tools module
    from okx_mcp.tools import mcp
except ImportError as e:
    # Provide helpful error message if imports fail (e.g., running from wrong directory)
    print(f"Error importing modules from 'okx_mcp' package: {e}", file=sys.stderr)
    print("Please ensure you are running this script from the project root directory ('okx-mcp-2')", file=sys.stderr)
    print("and the 'okx_mcp' package structure (with __init__.py) is correct.", file=sys.stderr)
    sys.exit(1)
except SystemExit as e:
    # Catch SystemExit if OKXClient initialization failed in tools.py
    print(f"Critical error during initialization in okx_mcp.tools: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    # Catch any other unexpected import-time errors
    print(f"An unexpected error occurred during module imports: {e}", file=sys.stderr)
    sys.exit(1)


# Configure logging using the function from the config module
setup_logging()
# Get a logger specific to this entry point file, if needed for startup messages
logger = logging.getLogger(__name__)

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting FastMCP server using 'mcp' instance from okx_mcp.tools...")
    try:
        # Run the FastMCP server - the 'mcp' instance already has tools registered
        mcp.run()
    except KeyboardInterrupt:
         logger.info("Server stopped by user (KeyboardInterrupt).")
         sys.exit(0) # Clean exit on Ctrl+C
    except Exception as e:
        # Log any critical error during server run
        logger.critical(f"FastMCP server encountered a critical error during execution: {e}", exc_info=True)
        sys.exit(f"FastMCP server failed: {e}")