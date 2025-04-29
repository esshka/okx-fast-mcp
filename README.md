# OKX MCP Server 🚀

A powerful Model Context Protocol (MCP) server that exposes OKX cryptocurrency exchange API functionality to LLM applications.

## Overview

This server implements the Model Context Protocol to provide LLMs with tools to interact with the OKX cryptocurrency exchange. It enables AI assistants to fetch market data, account information, and execute trades through a standardized interface.

The server handles API authentication, request signing, and provides structured responses suitable for LLM consumption.

## Features

- 📈 **Real-time Market Data**: Access ticker prices (`get_price`), candlestick data (`get_candlesticks`).
- 💰 **Account Information**: Check balances (`get_account_balance`), positions (`get_positions`), and trade history (`get_trade_history`).
- 🔄 **Trading Operations**:
    - Place SPOT limit (`place_spot_limit_order`) and market (`place_spot_market_order`) orders.
    - Place SWAP limit (`place_swap_limit_order`) and market (`place_swap_market_order`) orders.
    - Calculate SWAP position sizes (`calculate_position_size`).
- 🔒 **Secure Authentication**: Handles API key management and request signing using environment variables.
- 📊 **Smart Calculations**: Built-in utilities for SWAP contract size calculations and SPOT order size validation.
- 🧠 **LLM-Ready**: Structured responses designed for AI consumption via FastMCP.
- ✅ **Input Validation**: Includes basic validation for instrument IDs and order parameters (size, price).

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/okx-mcp-server.git
cd okx-mcp-server

# Install dependencies
pip install fastmcp requests
```

## Configuration

Set the following environment variables for API authentication:

```bash
export OKX_API_KEY="your_api_key"
export OKX_SECRET_KEY="your_secret_key"
export OKX_PASSPHRASE="your_passphrase"
```

> **Note**: Without these credentials, only public endpoints (`get_price`, `get_candlesticks`, `calculate_position_size`) will be accessible. Authenticated endpoints (`get_account_balance`, `get_positions`, `get_trade_history`, and all `place_*_order` tools) require valid credentials.

## Usage

Start the MCP server:

```bash
python server.py
```

The server exposes the following tools to MCP clients:

### Available Tools

The following tools are exposed via the MCP server:

| Tool                          | Description                                                                                                | Key Parameters                                                                      | Auth Required |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------- |
| `get_price`                   | Get latest market ticker price for a specific instrument.                                                  | `instrument` (e.g., "BTC-USDT", "ETH-USDT-SWAP")                                    | No            |
| `get_candlesticks`            | Get candlestick (k-line) data.                                                                             | `instrument`, `bar` (e.g., "1m", "1H", default: "1m"), `limit` (default: 100)       | No            |
| `get_account_balance`         | Get account balance information (total equity, details per currency).                                      | -                                                                                   | Yes           |
| `get_positions`               | Get current open positions.                                                                                | `instrument_type` (e.g., "SWAP", default: "SWAP"), `instrument_id` (optional)     | Yes           |
| `get_trade_history`           | Get recent trade (fill) history.                                                                           | `instrument_type`, `instrument_id`, `order_id` (all optional), `limit` (default: 100) | Yes           |
| `place_swap_limit_order`    | Place a **limit** order for a **SWAP** instrument.                                                           | `instrument` (e.g., "BTC-USDT-SWAP"), `side` ("buy"/"sell"), `size` (**USDT**), `price` | Yes           |
| `place_swap_market_order`   | Place a **market** order for a **SWAP** instrument.                                                          | `instrument` (e.g., "BTC-USDT-SWAP"), `side` ("buy"/"sell"), `size` (**USDT**)        | Yes           |
| `place_spot_limit_order`    | Place a **limit** order for a **SPOT** instrument.                                                           | `instrument` (e.g., "BTC-USDT"), `trade_mode` ("cash"/"cross"), `side`, `size` (**Base Currency**), `price` | Yes           |
| `place_spot_market_order`   | Place a **market** order for a **SPOT** instrument.                                                          | `instrument` (e.g., "BTC-USDT"), `trade_mode` ("cash"/"cross"), `side`, `size` (**Base Currency**) | Yes           |
| `calculate_position_size`     | Calculate contracts needed for a given USDT position size on a **SWAP** instrument.                        | `instrument` (e.g., "BTC-USDT-SWAP"), `usdt_size` (float)                           | No            |

**Important Note on Order Sizes:**

*   For **SWAP** orders (`place_swap_*`), the `size` parameter represents the desired position size in **USDT**. The server automatically calculates the corresponding number of contracts based on the current price and instrument details.
*   For **SPOT** orders (`place_spot_*`), the `size` parameter represents the amount of the **base currency** to buy or sell (e.g., `0.1` for 0.1 BTC in a BTC-USDT order). The server validates this size against the instrument's minimum size and lot size requirements.

## Example - Accessing through Python Client

```python
# Example using a hypothetical MCP client library
from mcp_client import MCPClient # Replace with your actual client implementation

# Connect to your hosted OKX MCP server
# Ensure the server is running: python server.py
client = MCPClient("http://localhost:8000") # Default FastMCP port is 8000

try:
    # Get BTC price
    btc_price = client.invoke("get_price", {"instrument": "BTC-USDT-SWAP"})
    print(f"BTC Price: {btc_price}")

    # --- Authenticated Example (Requires API Keys Set) ---
    # Ensure OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE are set as environment variables
    # when starting server.py
    if client.is_tool_available("get_account_balance"): # Check if auth tools might be ready
        # Get account balance
        balance = client.invoke("get_account_balance")
        print(f"Account Balance: {balance}")

        # Place a SWAP limit order (Example - use with caution!)
        # Buys 100 USDT worth of BTC-USDT-SWAP if price reaches 60000
        swap_order_details = {
            "instrument": "BTC-USDT-SWAP",
            "side": "buy",
            "size": "100", # Size in USDT for SWAPS
            "price": "60000"
        }
        swap_order_result = client.invoke("place_swap_limit_order", swap_order_details)
        print(f"Swap Limit Order Result: {swap_order_result}")

        # Place a SPOT market order (Example - use with caution!)
        # Sells 0.001 ETH on the ETH-USDT spot market at the current market price
        spot_order_details = {
            "instrument": "ETH-USDT",
            "trade_mode": "cash",
            "side": "sell",
            "size": "0.001" # Size in Base Currency (ETH) for SPOT
        }
        spot_order_result = client.invoke("place_spot_market_order", spot_order_details)
        print(f"Spot Market Order Result: {spot_order_result}")

except Exception as e:
    print(f"An error occurred: {e}")

```

## Integration with FastMCP

This server is built with [FastMCP](https://gofastmcp.com), the Pythonic way to build MCP servers. FastMCP provides a clean interface for exposing functionality to LLMs:

```python
from fastmcp import FastMCP

mcp = FastMCP("OKX API 🚀")

@mcp.tool()
def get_price(instrument: str) -> Dict[str, Any]:
    """Get the latest market ticker price for a specific OKX instrument."""
    # Implementation...
    
# More tools...

if __name__ == "__main__":
    mcp.run()
```

## Model Context Protocol

The Model Context Protocol (MCP) is a standardized way to provide context and tools to LLMs. It's often described as "the USB-C port for AI," providing a uniform way to connect language models to external resources.

## Integration with Claude Products

You can integrate this OKX MCP server with various Claude products to give your AI assistant direct access to cryptocurrency market data and trading capabilities.

### Configuration with mcp.json

Claude products like Cursor, Claude CLI, and Claude Desktop use an `mcp.json` configuration file to connect to MCP servers. Here's how to properly configure the OKX MCP server:

1. Create or edit the `mcp.json` file in the appropriate location:
   - Cursor: `~/.cursor/mcp.json`
   - Claude CLI: `~/.claude/mcp.json`
   - Claude Desktop: `~/.claude-desktop/mcp.json` (location may vary by platform)

2. Add the OKX configuration to your `mcp.json`:

```json
{
  "mcpServers": {
    "okx": {
      "command": "/path/to/your/python",
      "args": ["/path/to/your/okx-mcp-server/server.py"],
      "env": {
        "OKX_API_KEY": "your_api_key",
        "OKX_SECRET_KEY": "your_secret_key",
        "OKX_PASSPHRASE": "your_passphrase"
      }
    }
    // Other MCP servers can be added here
  }
}
```

3. Restart your Claude product to load the new configuration.

### Using OKX Tools with Claude

Once configured, you can ask Claude to perform tasks using the specific OKX tools:

*   "Use `get_price` for the instrument SOL-USDT-SWAP."
*   "Call `get_candlesticks` for ETH-USDT with a bar of '1H' and limit 50."
*   "Invoke `get_account_balance`."
*   "Use `get_positions` for instrument_type 'SWAP'."
*   "Can you call `place_swap_limit_order`? Instrument BTC-USDT-SWAP, side buy, size 50 USDT, price 62500." (Use caution with real trading!)
*   "Use `place_spot_market_order` to sell 0.1 ETH on ETH-USDT using cash trade mode." (Use caution!)
*   "What contract size do I need for 250 USDT on ADA-USDT-SWAP? Use `calculate_position_size`."

Claude will use the specified OKX tools to fulfill your requests. Remember that placing orders involves real financial risk.

> **Security Note**: The `mcp.json` file will contain sensitive API credentials (`OKX_API_KEY`, `OKX_SECRET_KEY`, `OKX_PASSPHRASE`). **Treat this file as highly confidential.** Ensure it has restrictive file permissions (e.g., `chmod 600 ~/.cursor/mcp.json`) and is never committed to version control or shared publicly. Anyone with access to these keys can control your OKX account.

## Error Handling

The server attempts to catch and report errors from the OKX API. You might encounter:

*   `OKXError`: Specific errors returned by the OKX API (e.g., insufficient balance, invalid parameters, authentication failure). The error message often includes the OKX error code and details from the API.
*   `ConnectionError`: Network issues connecting to the OKX API.
*   `ValueError`: Invalid input parameters provided to a tool (e.g., non-numeric size, incorrectly formatted instrument ID).
*   `PermissionError`: Attempting to use an authenticated endpoint without providing valid API credentials when starting the server.
*   `RuntimeError`: Unexpected server-side errors.

Check the server logs (`server.py` output) and the error messages returned by the tools for more details.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 