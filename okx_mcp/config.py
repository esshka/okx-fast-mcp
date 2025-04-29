# okx_mcp/config.py
import logging
import sys # Add sys import if needed for stderr logging later

def setup_logging():
    # --- PASTE THE LOGGING SETUP CODE FROM server.py HERE ---
    # Example structure (replace with actual code from server.py):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        # stream=sys.stderr # Consider if stderr stream is needed/used
    )
    # Set log levels for noisy libraries (e.g., requests, urllib3)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    # --- END OF PASTED CODE ---

    logger = logging.getLogger(__name__) # Use __name__ for the config module's logger
    logger.info("Logging configured via okx_mcp.config.setup_logging.")

# You can add other configuration-related code here later if needed.