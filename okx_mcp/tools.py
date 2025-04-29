# okx_mcp/tools.py
import logging
import csv
import io
import json
import yaml # Added for YAML formatting
from typing import Optional, Dict, Any, List, Union
from fastmcp import FastMCP

# Relative imports for client, services, and error
from .client import OKXClient, OKXError, API_V5_PREFIX
from . import services # Import the services module

logger = logging.getLogger(__name__)

# --- Initialize MCP and Client ---
# Create a single client instance for all tools to use
# This assumes credentials are set in the environment where server.py runs
try:
    okx_client = OKXClient()
    logger.info("OKXClient initialized successfully in tools module.")
except (ValueError, Exception) as e:
    logger.critical(f"Failed to initialize OKXClient in tools.py: {e}", exc_info=True)
    # Raise SystemExit to prevent the server from starting with a non-functional client
    raise SystemExit(f"Critical error during OKXClient initialization in tools module: {e}") from e

# Instantiate FastMCP - use the name from the original server.py
mcp = FastMCP("OKX API 🚀") # Or the actual name used previously

# --- Helper for CSV Formatting (if needed, adapt from original) ---
def _format_to_csv(data: List[Dict[str, Any]], fieldnames: List[str]) -> str:
    """Formats a list of dictionaries into a CSV string."""
    if not data:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(data)
    return output.getvalue()

# --- Data Formatting Abstraction ---
def format_data(data: Union[List[Dict[str, Any]], Dict[str, Any]], format_type: str, headers: Optional[List[str]] = None) -> Any:
    """
    Formats the given data into the specified format.

    Args:
        data: The data to format (list of dictionaries or a single dictionary).
        format_type: The desired format ('json', 'csv', 'md', 'yaml').
        headers: Optional list of headers (required for CSV/MD if data is list of dicts).

    Returns:
        The formatted data (string for text formats, original data for 'json').
        Returns an error string if formatting fails or format is unsupported.
    """
    format_type = format_type.lower()
    logger.debug(f"Formatting data to type: {format_type}")

    if not data:
        logger.warning("Formatting requested for empty data.")
        if format_type == 'json':
            return data # Return empty list/dict as is
        else:
            return "No data to format."

    is_list = isinstance(data, list)

    try:
        if format_type == 'json':
            # For JSON, return the raw Python object (list/dict)
            # MCP handles JSON serialization automatically if needed
            return data
        elif format_type == 'csv':
            if not is_list:
                return "error: CSV format requires a list of dictionaries."
            if not headers:
                if data:
                    headers = list(data[0].keys())
                else:
                    return "error: Cannot determine headers for empty CSV data."
            return _format_to_csv(data, headers) # Use existing helper
        elif format_type == 'yaml':
            return yaml.dump(data, allow_unicode=True, sort_keys=False)
        elif format_type == 'md': # Markdown Table
            if not is_list:
                return "error: Markdown table format requires a list of dictionaries."
            if not data:
                return "No data for Markdown table."

            if not headers:
                headers = list(data[0].keys())

            # Create header row
            header_line = "| " + " | ".join(headers) + " |"
            separator_line = "| " + " | ".join(['---'] * len(headers)) + " |"

            # Create data rows
            data_lines = []
            for row_dict in data:
                row_values = [str(row_dict.get(h, '')) for h in headers]
                data_lines.append("| " + " | ".join(row_values) + " |")

            return "\n".join([header_line, separator_line] + data_lines)
        else:
            logger.error(f"Unsupported format type requested: {format_type}")
            return f"error: Unsupported format type '{format_type}'. Supported formats: json, csv, md, yaml."

    except Exception as e:
        logger.exception(f"Error formatting data to {format_type}: {e}")
        return f"error: Failed to format data as {format_type}: {e}"


# --- MCP Tools ---

@mcp.tool()
def get_price(instrument: str) -> Dict[str, Any]:
    """Get the latest market ticker price for a specific instrument (e.g., BTC-USDT-SWAP)."""
    logger.info(f"Tool: Fetching price for instrument: {instrument}")
    endpoint = f"{API_V5_PREFIX}/market/ticker"
    try:
        response_data = okx_client.make_request("GET", endpoint, params={"instId": instrument})
        if response_data and response_data.get("data"):
            ticker_data = response_data["data"][0]
            # Select relevant fields
            result = {
                "instrument": ticker_data.get("instId"),
                "last_price": ticker_data.get("last"),
                "bid_price": ticker_data.get("bidPx"),
                "ask_price": ticker_data.get("askPx"),
                "24h_high": ticker_data.get("high24h"),
                "24h_low": ticker_data.get("low24h"),
                "24h_volume_contracts": ticker_data.get("vol24h"), # Volume in contracts/base currency
                "24h_volume_usdt": ticker_data.get("volCcy24h"), # Volume in quote currency (USDT for linear)
                "timestamp_ms": ticker_data.get("ts"),
            }
            logger.info(f"Tool: Price for {instrument}: {result['last_price']}")
            return result
        else:
            logger.warning(f"Tool: No price data returned for {instrument}")
            return {"error": f"No data found for instrument {instrument}"}
    except (ConnectionError, OKXError, ValueError) as e:
        logger.error(f"Tool: Error fetching price for {instrument}: {e}", exc_info=True)
        return {"error": f"Failed to fetch price for {instrument}: {e}"}
    except Exception as e:
        logger.exception(f"Tool: Unexpected error in get_price for {instrument}: {e}")
        return {"error": f"An unexpected error occurred: {e}"}


@mcp.tool()
def get_candlesticks(instrument: str, bar: str = "1m", limit: int = 100, format: str = "json") -> Any:
    """
    Get candlestick (k-line) data for an instrument.
    Returns data in the specified format (json, csv, md, yaml). Default is json.
    """
    logger.info(f"Tool: Fetching candlesticks for {instrument} (Bar: {bar}, Limit: {limit}, Format: {format})")
    endpoint = f"{API_V5_PREFIX}/market/candles"
    params = {"instId": instrument, "bar": bar, "limit": str(limit)}
    try:
        response_data = okx_client.make_request("GET", endpoint, params=params)
        if response_data and response_data.get("data"):
            # OKX returns data as a list of lists: [ts, o, h, l, c, vol, volCcy, ...]
            header = ["timestamp_ms", "open", "high", "low", "close", "volume_contracts", "volume_usdt"]
            # Convert list of lists to list of dicts
            candles_data = [dict(zip(header, k[:len(header)])) for k in response_data["data"]]
            logger.info(f"Tool: Fetched {len(candles_data)} candlesticks for {instrument}")

            # Format the data using the new abstraction
            formatted_output = format_data(candles_data, format, headers=header)
            return formatted_output
        else:
            logger.warning(f"Tool: No candlestick data returned for {instrument}")
            # Return appropriate empty/error based on format
            return format_data([], format) # Format empty list
    except (ConnectionError, OKXError, ValueError) as e:
        logger.error(f"Tool: Error fetching candlesticks for {instrument}: {e}", exc_info=True)
        return {"error": f"Failed to fetch candlesticks for {instrument}: {e}"} # Return error dict for JSON default
    except Exception as e:
        logger.exception(f"Tool: Unexpected error in get_candlesticks for {instrument}: {e}")
        return {"error": f"An unexpected error occurred: {e}"} # Return error dict for JSON default


@mcp.tool()
def get_account_balance() -> Dict[str, Any]:
    """Get account balance information (total equity, available balance per currency)."""
    logger.info("Tool: Fetching account balance")
    endpoint = f"{API_V5_PREFIX}/account/balance"
    try:
        # This requires authentication
        response_data = okx_client.make_request("GET", endpoint, auth=True)
        if response_data and response_data.get("data"):
            # Process the balance data - structure might vary slightly
            # Example: Extract total equity and details per currency
            account_info = response_data["data"][0] # Usually a list containing one account summary
            total_equity = account_info.get("totalEq")
            details = account_info.get("details", [])
            balances = {item.get("ccy"): {"equity": item.get("eq"), "available": item.get("availEq")} for item in details}

            result = {
                "total_equity_usd": total_equity,
                "currency_balances": balances
            }
            logger.info(f"Tool: Account balance fetched. Total Equity: {total_equity}")
            return result
        else:
            logger.warning("Tool: No account balance data returned.")
            return {"error": "No account balance data found."}
    except (ConnectionError, OKXError, ValueError) as e:
        logger.error(f"Tool: Error fetching account balance: {e}", exc_info=True)
        return {"error": f"Failed to fetch account balance: {e}"}
    except Exception as e:
        logger.exception(f"Tool: Unexpected error in get_account_balance: {e}")
        return {"error": f"An unexpected error occurred: {e}"}


@mcp.tool()
def get_positions(instrument_type: str = "SWAP", instrument_id: Optional[str] = None) -> List[Dict[str, Any]]:
     """Get current open positions (SWAP, FUTURES, MARGIN, OPTION)."""
     logger.info(f"Tool: Fetching positions - Type: {instrument_type}, ID: {instrument_id}")
     endpoint = f"{API_V5_PREFIX}/account/positions"
     params: Dict[str, Any] = {"instType": instrument_type}
     if instrument_id:
         params["instId"] = instrument_id
     try:
         response_data = okx_client.make_request("GET", endpoint, params=params, auth=True)
         if response_data and response_data.get("code") == "0":
             positions = response_data.get("data", [])
             logger.info(f"Tool: Fetched {len(positions)} positions.")
             # Optionally filter/simplify the output if needed
             return positions
         else:
             logger.warning(f"Tool: No position data returned or error in response: {response_data}")
             # Return empty list or error structure based on preference
             return [] # Returning empty list if no positions or error code != 0
     except (ConnectionError, OKXError, ValueError) as e:
         logger.error(f"Tool: Error fetching positions: {e}", exc_info=True)
         # Return empty list on error to avoid breaking flows expecting a list
         return [{"error": f"Failed to fetch positions: {e}"}]
     except Exception as e:
         logger.exception(f"Tool: Unexpected error in get_positions: {e}")
         return [{"error": f"An unexpected error occurred: {e}"}]


@mcp.tool()
def get_trade_history(
    instrument_type: str = "SWAP",
    instrument_id: Optional[str] = None,
    order_type: Optional[str] = None, # market, limit, etc.
    state: Optional[str] = None,      # filled
    limit: int = 100
) -> List[Dict[str, Any]]:
     """Get recent trade (fill) history."""
     logger.info(f"Tool: Fetching trade history - Type: {instrument_type}, ID: {instrument_id}, Limit: {limit}")
     endpoint = f"{API_V5_PREFIX}/trade/fills-history" # Use fills-history for last 3 months
     params: Dict[str, Any] = {"instType": instrument_type, "limit": str(limit)}
     if instrument_id:
         params["instId"] = instrument_id
     if order_type:
         params["ordType"] = order_type
     if state:
          params["state"] = state # Note: This endpoint might not support 'state', check API docs. Fills are usually always 'filled'.

     try:
         response_data = okx_client.make_request("GET", endpoint, params=params, auth=True)
         if response_data and response_data.get("code") == "0":
             trades = response_data.get("data", [])
             logger.info(f"Tool: Fetched {len(trades)} trade history entries.")
             return trades
         else:
             logger.warning(f"Tool: No trade history data returned or error in response: {response_data}")
             return []
     except (ConnectionError, OKXError, ValueError) as e:
         logger.error(f"Tool: Error fetching trade history: {e}", exc_info=True)
         return [{"error": f"Failed to fetch trade history: {e}"}]
     except Exception as e:
         logger.exception(f"Tool: Unexpected error in get_trade_history: {e}")
         return [{"error": f"An unexpected error occurred: {e}"}]


@mcp.tool()
def place_swap_limit_order(
    instrument: str,
    side: str, # buy, sell
    size_usdt: str, # Order size in USDT
    price: str,
    position_side: Optional[str] = None, # long, short (for hedge mode)
    client_order_id: Optional[str] = None,
    tag: Optional[str] = None,
    reduce_only: Optional[bool] = None
) -> Dict[str, Any]:
    """Place a limit order for SWAP trading, specifying size in USDT."""
    logger.info(f"Tool: Placing SWAP limit order - Inst: {instrument}, Side: {side}, Size(USDT): {size_usdt}, Px: {price}")
    if not instrument.endswith('-SWAP'):
        logger.error(f"Tool: Invalid instrument for SWAP order: {instrument}")
        return {"error": "Invalid instrument for SWAP order. Must end with '-SWAP'."}
    try:
        # 1. Convert USDT size to contract size using the service
        contract_size_str = services._convert_usdt_to_contracts(okx_client, instrument, size_usdt)
        logger.info(f"Tool: Converted {size_usdt} USDT to {contract_size_str} contracts for {instrument}")

        # 2. Place the order using the internal service function
        # Trade mode for SWAP is typically 'cross' or 'isolated' - assuming 'cross' if not specified
        # The service function handles the API call
        return services._place_order_internal(
            client=okx_client,
            instrument=instrument,
            trade_mode="cross", # Or allow as parameter if needed
            side=side,
            order_type="limit",
            size=contract_size_str, # Use the calculated contract size
            order_price=price,
            position_side=position_side,
            client_order_id=client_order_id,
            tag=tag,
            reduce_only=reduce_only
        )
    except (ValueError, ConnectionError, OKXError) as e:
        logger.error(f"Tool: Error placing SWAP limit order for {instrument}: {e}", exc_info=True)
        return {"error": f"Failed to place SWAP limit order: {e}"}
    except Exception as e:
        logger.exception(f"Tool: Unexpected error in place_swap_limit_order for {instrument}: {e}")
        return {"error": f"An unexpected error occurred: {e}"}


@mcp.tool()
def place_spot_limit_order(
    instrument: str,
    side: str, # buy, sell
    size: str, # Order size in base currency (e.g., BTC amount for BTC-USDT)
    price: str,
    trade_mode: str = "cash", # Usually 'cash' for SPOT
    client_order_id: Optional[str] = None,
    tag: Optional[str] = None
    # reduce_only typically not applicable to SPOT
) -> Dict[str, Any]:
    """Place a limit order for SPOT trading."""
    logger.info(f"Tool: Placing SPOT limit order - Inst: {instrument}, Side: {side}, Size: {size}, Px: {price}")
    if '-SWAP' in instrument or '-FUTURES' in instrument:
         logger.error(f"Tool: Invalid instrument for SPOT order: {instrument}")
         return {"error": "Invalid instrument for SPOT order. Do not use SWAP or FUTURES instruments."}
    try:
        # 1. Validate and correct size using the service
        # SPOT uses base currency size, so validate against instrument rules
        corrected_size = services.validate_and_correct_order_size(okx_client, instrument, size, 'SPOT')
        logger.info(f"Tool: Validated/corrected SPOT size for {instrument}: {corrected_size} (Original: {size})")

        # 2. Place the order using the internal service function
        return services._place_order_internal(
            client=okx_client,
            instrument=instrument,
            trade_mode=trade_mode, # Should be 'cash' for spot
            side=side,
            order_type="limit",
            size=corrected_size, # Use validated/corrected size
            order_price=price,
            # position_side not applicable for SPOT
            client_order_id=client_order_id,
            tag=tag
            # reduce_only not applicable
        )
    except (ValueError, ConnectionError, OKXError) as e:
         logger.error(f"Tool: Error placing SPOT limit order for {instrument}: {e}", exc_info=True)
         return {"error": f"Failed to place SPOT limit order: {e}"}
    except Exception as e:
        logger.exception(f"Tool: Unexpected error in place_spot_limit_order for {instrument}: {e}")
        return {"error": f"An unexpected error occurred: {e}"}


@mcp.tool()
def calculate_position_size(instrument: str, usdt_size: float) -> Dict[str, Any]:
    """Calculate the number of contracts needed for a given USDT position size (for SWAP)."""
    logger.info(f"Tool: Calculating position size for {instrument}, USDT: {usdt_size}")
    if not instrument.endswith('-SWAP'):
         logger.error(f"Tool: Position size calculation currently only supported for SWAP instruments: {instrument}")
         return {"error": "Position size calculation currently only supported for SWAP instruments."}
    if usdt_size <= 0:
         return {"error": "USDT size must be positive."}

    try:
        # 1. Calculate contract size using service
        contract_size = services.calculate_contract_size(okx_client, instrument, usdt_size)
        if contract_size is None:
             # Error already logged in service
             return {"error": f"Could not calculate contract size for {instrument}."}

        # 2. Get details for context (min size, lot size) using service
        details = services.get_instrument_details(okx_client, instrument)
        if details is None:
             return {"error": f"Could not retrieve instrument details for {instrument}."}

        price, ctVal, lotSz, minSz = details

        # 3. Format result
        result = {
            "instrument": instrument,
            "usdt_size": usdt_size,
            "calculated_contracts": contract_size,
            "current_price": price,
            "contract_value": ctVal,
            "lot_size": lotSz,
            "min_order_size_contracts": minSz,
            "message": f"To open a position worth approximately {usdt_size} USDT for {instrument} at current price {price}, you need about {contract_size:.8f} contracts. Note: Order size must be a multiple of {lotSz} and at least {minSz}."
        }
        logger.info(f"Tool: Position size calculation for {instrument} complete.")
        return result
    except (ConnectionError, OKXError, ValueError) as e:
        logger.error(f"Tool: Error calculating position size for {instrument}: {e}", exc_info=True)
        return {"error": f"Failed to calculate position size: {e}"}
    except Exception as e:
        logger.exception(f"Tool: Unexpected error in calculate_position_size for {instrument}: {e}")
        return {"error": f"An unexpected error occurred: {e}"}


# --- Grid Algo Order Tools ---

@mcp.tool()
def place_spot_grid_algo_order(
    instrument_id: str,
    max_price: str,
    min_price: str,
    grid_num: str,
    grid_run_type: str = "1", # 1: Arithmetic, 2: Geometric
    investment_amount: Optional[str] = None, # Total investment in quote ccy (e.g., USDT)
    base_order_size: Optional[str] = None, # Size per grid order in base ccy (e.g., BTC) - use one of investment or base size
    tp_trigger_px: Optional[str] = None,
    sl_trigger_px: Optional[str] = None,
    tag: Optional[str] = None,
    client_order_id: Optional[str] = None
) -> Dict[str, Any]:
    """Place a spot grid algo order."""
    logger.info(f"Tool: Placing SPOT Grid Algo order for {instrument_id}")
    if investment_amount is None and base_order_size is None:
         return {"error": "Either investment_amount (quote ccy) or base_order_size (base ccy) must be provided."}
    if investment_amount is not None and base_order_size is not None:
         return {"error": "Provide either investment_amount OR base_order_size, not both."}

    # Build the data payload for the service function
    data: Dict[str, Any] = {
        "instId": instrument_id,
        "algoOrdType": "grid",
        "maxPx": max_price,
        "minPx": min_price,
        "gridNum": grid_num,
        "runType": grid_run_type,
        "instType": "SPOT", # Specify instrument type
    }
    if investment_amount:
        data["quoteSz"] = investment_amount # Use quoteSz for investment amount
    if base_order_size:
         data["baseSz"] = base_order_size # Use baseSz for size per grid
    if tp_trigger_px:
        data["tpTriggerPx"] = tp_trigger_px
    if sl_trigger_px:
        data["slTriggerPx"] = sl_trigger_px
    if tag:
        data["tag"] = tag
    if client_order_id:
         data["algoClOrdId"] = client_order_id # Use algoClOrdId for client ID

    try:
        # Call the service function
        return services._place_grid_algo_order_internal(client=okx_client, data=data)
    except (ValueError, ConnectionError, OKXError) as e:
        logger.error(f"Tool: Error placing SPOT Grid Algo order for {instrument_id}: {e}", exc_info=True)
        return {"error": f"Failed to place SPOT Grid Algo order: {e}"}
    except Exception as e:
        logger.exception(f"Tool: Unexpected error in place_spot_grid_algo_order for {instrument_id}: {e}")
        return {"error": f"An unexpected error occurred: {e}"}

@mcp.tool()
def get_grid_algo_order_list(
    algo_order_type: str, # grid, iceberg, twap, etc.
    instrument_type: str = "SPOT", # SPOT, SWAP, FUTURES, OPTION
    instrument_id: Optional[str] = None,
    state: Optional[str] = None, # effective, paused, stopping
    algo_id: Optional[str] = None,
    limit: int = 100,
    format: str = "json" # Added format parameter
) -> Any: # Returns JSON by default, or formatted string
     """Get a list of pending or historical grid algo orders. Supports json, csv, md, yaml formats."""
     logger.info(f"Tool: Fetching Grid Algo list - Type: {algo_order_type}, InstType: {instrument_type}, Format: {format}")
     # Determine endpoint based on state (pending vs history) - assuming pending for now
     endpoint = f"{API_V5_PREFIX}/tradingBot/grid/orders-algo-pending"
     # TODO: Add logic to switch to /tradingBot/grid/orders-algo-history based on state or a dedicated parameter

     params: Dict[str, Any] = {
         "algoOrdType": algo_order_type,
         "instType": instrument_type,
         "limit": str(limit)
     }
     if instrument_id: params["instId"] = instrument_id
     if state: params["state"] = state # Check API docs if 'state' is valid for pending endpoint
     if algo_id: params["algoId"] = algo_id

     try:
         response_data = okx_client.make_request("GET", endpoint, params=params, auth=True)
         if response_data and response_data.get("code") == "0":
             orders = response_data.get("data", [])
             logger.info(f"Tool: Fetched {len(orders)} grid algo orders.")
             # Format the data using the abstraction
             return format_data(orders, format)
         else:
             logger.warning(f"Tool: No grid algo list data or error: {response_data}")
             # Return appropriate empty/error based on format
             msg = f"No grid algo orders found or API error occurred: {response_data.get('msg', 'Unknown')}"
             if format == "json":
                 return {"error": msg}
             else:
                 return f"error: {msg}"
     except (ConnectionError, OKXError, ValueError) as e:
         logger.error(f"Tool: Error fetching grid algo list: {e}", exc_info=True)
         if format == "json":
             return {"error": f"Failed to fetch grid algo list: {e}"}
         else:
             return f"error: Failed to fetch grid algo list: {e}"
     except Exception as e:
         logger.exception(f"Tool: Unexpected error in get_grid_algo_order_list: {e}")
         if format == "json":
             return {"error": f"An unexpected error occurred: {e}"}
         else:
             return f"error: An unexpected error occurred: {e}"


@mcp.tool()
def get_funding_rate(instrument_id: str, format: str = "json") -> Any: # Returns JSON by default
    """Get the current funding rate for a SWAP instrument. Supports json, csv, md, yaml formats."""
    logger.info(f"Tool: Fetching funding rate for {instrument_id}, Format: {format}")
    if not instrument_id.endswith("-SWAP"):
        msg = "Funding rate is only applicable to SWAP instruments."
        return {"error": msg} if format == "json" else f"error: {msg}"

    endpoint = f"{API_V5_PREFIX}/public/funding-rate"
    params = {"instId": instrument_id}
    try:
        response_data = okx_client.make_request("GET", endpoint, params=params)
        if response_data and response_data.get("code") == "0" and response_data.get("data"):
            rates = response_data["data"]
            logger.info(f"Tool: Fetched funding rate for {instrument_id}.")
            return format_data(rates, format)
        else:
            logger.warning(f"Tool: No funding rate data or error: {response_data}")
            msg = f"No funding rate data found or API error: {response_data.get('msg', 'Unknown')}"
            if format == "json":
                return {"error": msg}
            else:
                return f"error: {msg}"
    except (ConnectionError, OKXError, ValueError) as e:
        logger.error(f"Tool: Error fetching funding rate for {instrument_id}: {e}", exc_info=True)
        if format == "json":
            return {"error": f"Failed to fetch funding rate: {e}"}
        else:
            return f"error: Failed to fetch funding rate: {e}"
    except Exception as e:
        logger.exception(f"Tool: Unexpected error in get_funding_rate for {instrument_id}: {e}")
        if format == "json":
            return {"error": f"An unexpected error occurred: {e}"}
        else:
            return f"error: An unexpected error occurred: {e}"

# --- Add other Grid Algo tools similarly (place_contract_grid, amend, stop, get_details) ---
# Note: place_contract_grid, amend, stop, get_details were not explicitly listed in the original server.py content provided,
# but the instructions mention adding them "similarly". I will add placeholders for these based on the structure.

@mcp.tool()
def place_contract_grid_algo_order(
    instrument_id: str,
    max_price: str,
    min_price: str,
    grid_num: str,
    grid_run_type: str = "1", # 1: Arithmetic, 2: Geometric
    investment_amount: Optional[str] = None, # Total investment in quote ccy (e.g., USDT)
    contract_order_size: Optional[str] = None, # Size per grid order in contracts - use one of investment or contract size
    tp_trigger_px: Optional[str] = None,
    sl_trigger_px: Optional[str] = None,
    tag: Optional[str] = None,
    client_order_id: Optional[str] = None
) -> Dict[str, Any]:
    """Place a contract grid algo order (for SWAP/FUTURES)."""
    logger.info(f"Tool: Placing Contract Grid Algo order for {instrument_id}")
    if investment_amount is None and contract_order_size is None:
         return {"error": "Either investment_amount (quote ccy) or contract_order_size (contracts) must be provided."}
    if investment_amount is not None and contract_order_size is not None:
         return {"error": "Provide either investment_amount OR contract_order_size, not both."}
    if not (instrument_id.endswith('-SWAP') or instrument_id.endswith('-FUTURES')):
         return {"error": "Contract grid orders are only for SWAP or FUTURES instruments."}

    # Build the data payload for the service function
    data: Dict[str, Any] = {
        "instId": instrument_id,
        "algoOrdType": "grid",
        "maxPx": max_price,
        "minPx": min_price,
        "gridNum": grid_num,
        "runType": grid_run_type,
        "instType": "SWAP" if instrument_id.endswith('-SWAP') else "FUTURES", # Determine instrument type
    }
    if investment_amount:
        data["quoteSz"] = investment_amount # Use quoteSz for investment amount
    if contract_order_size:
         data["baseSz"] = contract_order_size # Use baseSz for size per grid (contracts)
    if tp_trigger_px:
        data["tpTriggerPx"] = tp_trigger_px
    if sl_trigger_px:
        data["slTriggerPx"] = sl_trigger_px
    if tag:
        data["tag"] = tag
    if client_order_id:
         data["algoClOrdId"] = client_order_id # Use algoClOrdId for client ID

    try:
        # Call the service function
        return services._place_grid_algo_order_internal(client=okx_client, data=data)
    except (ValueError, ConnectionError, OKXError) as e:
        logger.error(f"Tool: Error placing Contract Grid Algo order for {instrument_id}: {e}", exc_info=True)
        return {"error": f"Failed to place Contract Grid Algo order: {e}"}
    except Exception as e:
        logger.exception(f"Tool: Unexpected error in place_contract_grid_algo_order for {instrument_id}: {e}")
        return {"error": f"An unexpected error occurred: {e}"}

@mcp.tool()
def amend_grid_algo_order(
    algo_id: str,
    instrument_id: str,
    max_price: Optional[str] = None,
    min_price: Optional[str] = None,
    tp_trigger_px: Optional[str] = None,
    sl_trigger_px: Optional[str] = None,
    new_client_order_id: Optional[str] = None
) -> Dict[str, Any]:
    """Amend a pending grid algo order."""
    logger.info(f"Tool: Amending Grid Algo order {algo_id} for {instrument_id}")
    if not any([max_price, min_price, tp_trigger_px, sl_trigger_px, new_client_order_id]):
        return {"error": "At least one parameter (max_price, min_price, tp_trigger_px, sl_trigger_px, new_client_order_id) must be provided to amend."}

    data: Dict[str, Any] = {
        "algoId": algo_id,
        "instId": instrument_id,
    }
    if max_price: data["maxPx"] = max_price
    if min_price: data["minPx"] = min_price
    if tp_trigger_px: data["tpTriggerPx"] = tp_trigger_px
    if sl_trigger_px: data["slTriggerPx"] = sl_trigger_px
    if new_client_order_id: data["newAlgoClOrdId"] = new_client_order_id

    try:
        endpoint = f"{API_V5_PREFIX}/tradingBot/grid/amend-order-algo"
        response_data = okx_client.make_request("POST", endpoint, data=data, auth=True)
        if response_data and response_data.get("code") == "0":
            logger.info(f"Tool: Successfully amended grid algo order {algo_id}.")
            return response_data.get("data", [{}])[0] # Return the first item in data list
        else:
            logger.warning(f"Tool: Failed to amend grid algo order {algo_id}: {response_data}")
            return {"error": f"Failed to amend grid algo order {algo_id}: {response_data.get('msg', 'Unknown')}"}
    except (ValueError, ConnectionError, OKXError) as e:
        logger.error(f"Tool: Error amending Grid Algo order {algo_id}: {e}", exc_info=True)
        return {"error": f"Failed to amend Grid Algo order: {e}"}
    except Exception as e:
        logger.exception(f"Tool: Unexpected error in amend_grid_algo_order for {algo_id}: {e}")
        return {"error": f"An unexpected error occurred: {e}"}

@mcp.tool()
def stop_grid_algo_order(
    algo_id: str,
    instrument_id: str,
    stop_type: str = "1" # 1: Cancel all pending orders and close position, 2: Cancel all pending orders and keep position
) -> Dict[str, Any]:
    """Stop a pending grid algo order."""
    logger.info(f"Tool: Stopping Grid Algo order {algo_id} for {instrument_id} with stop type {stop_type}")

    data: Dict[str, Any] = {
        "algoId": algo_id,
        "instId": instrument_id,
        "stopType": stop_type
    }

    try:
        endpoint = f"{API_V5_PREFIX}/tradingBot/grid/stop-order-algo"
        response_data = okx_client.make_request("POST", endpoint, data=data, auth=True)
        if response_data and response_data.get("code") == "0":
            logger.info(f"Tool: Successfully stopped grid algo order {algo_id}.")
            return response_data.get("data", [{}])[0] # Return the first item in data list
        else:
            logger.warning(f"Tool: Failed to stop grid algo order {algo_id}: {response_data}")
            return {"error": f"Failed to stop grid algo order {algo_id}: {response_data.get('msg', 'Unknown')}"}
    except (ValueError, ConnectionError, OKXError) as e:
        logger.error(f"Tool: Error stopping Grid Algo order {algo_id}: {e}", exc_info=True)
        return {"error": f"Failed to stop Grid Algo order: {e}"}
    except Exception as e:
        logger.exception(f"Tool: Unexpected error in stop_grid_algo_order for {algo_id}: {e}")
        return {"error": f"An unexpected error occurred: {e}"}

@mcp.tool()
def get_grid_algo_order_details(
    algo_id: str,
    algo_order_type: str # grid, iceberg, twap, etc.
) -> Dict[str, Any]:
    """Get details of a specific grid algo order."""
    logger.info(f"Tool: Fetching details for Grid Algo order {algo_id} (Type: {algo_order_type})")

    endpoint = f"{API_V5_PREFIX}/tradingBot/grid/orders-algo-details"
    params: Dict[str, Any] = {
        "algoId": algo_id,
        "algoOrdType": algo_order_type
    }

    try:
        response_data = okx_client.make_request("GET", endpoint, params=params, auth=True)
        if response_data and response_data.get("code") == "0" and response_data.get("data"):
            details = response_data["data"][0] # Details are usually in the first item of the data list
            logger.info(f"Tool: Fetched details for grid algo order {algo_id}.")
            return details
        else:
            logger.warning(f"Tool: No grid algo details data or error: {response_data}")
            return {"error": f"No grid algo order details found for {algo_id} or API error: {response_data.get('msg', 'Unknown')}"}
    except (ConnectionError, OKXError, ValueError) as e:
        logger.error(f"Tool: Error fetching grid algo details for {algo_id}: {e}", exc_info=True)
        return {"error": f"Failed to fetch grid algo details: {e}"}
    except Exception as e:
        logger.exception(f"Tool: Unexpected error in get_grid_algo_order_details for {algo_id}: {e}")
        return {"error": f"An unexpected error occurred: {e}"}

@mcp.tool()
def get_funding_rate_history(instrument_id: str, before: Optional[str] = None, after: Optional[str] = None, limit: Optional[int] = 100, format: str = "json") -> Any:
    """Get historical funding rates for a SWAP instrument. Supports json, csv, md, yaml formats."""
    logger.info(f"Tool: Fetching funding rate history for {instrument_id}, Format: {format}")
    if not instrument_id.endswith("-SWAP"):
        msg = "Funding rate history is only applicable to SWAP instruments."
        return {"error": msg} if format == "json" else f"error: {msg}"

    endpoint = f"{API_V5_PREFIX}/public/funding-rate-history"
    params: Dict[str, Any] = {"instId": instrument_id, "limit": str(limit)}
    if before: params["before"] = before
    if after: params["after"] = after

    try:
        response_data = okx_client.make_request("GET", endpoint, params=params)
        if response_data and response_data.get("code") == "0" and response_data.get("data"):
            rates = response_data["data"]
            logger.info(f"Tool: Fetched {len(rates)} funding rate history entries for {instrument_id}.")
            return format_data(rates, format)
        else:
            logger.warning(f"Tool: No funding rate history data or error: {response_data}")
            msg = f"No funding rate history data found or API error: {response_data.get('msg', 'Unknown')}"
            if format == "json":
                return {"error": msg}
            else:
                return f"error: {msg}"
    except (ConnectionError, OKXError, ValueError) as e:
        logger.error(f"Tool: Error fetching funding rate history for {instrument_id}: {e}", exc_info=True)
        if format == "json":
            return {"error": f"Failed to fetch funding rate history: {e}"}
        else:
            return f"error: Failed to fetch funding rate history: {e}"
    except Exception as e:
        logger.exception(f"Tool: Unexpected error in get_funding_rate_history for {instrument_id}: {e}")
        if format == "json":
            return {"error": f"An unexpected error occurred: {e}"}
        else:
            return f"error: An unexpected error occurred: {e}"

@mcp.tool()
def get_open_interest(instrument_type: str, underlying: Optional[str] = None, instrument_family: Optional[str] = None, instrument_id: Optional[str] = None, format: str = "json") -> Any:
    """Get open interest data for instruments. Supports json, csv, md, yaml formats."""
    logger.info(f"Tool: Fetching open interest for type: {instrument_type}, Format: {format}")
    endpoint = f"{API_V5_PREFIX}/public/open-interest"
    params: Dict[str, Any] = {"instType": instrument_type}
    if underlying: params["uly"] = underlying
    if instrument_family: params["instFamily"] = instrument_family
    if instrument_id: params["instId"] = instrument_id

    try:
        response_data = okx_client.make_request("GET", endpoint, params=params)
        if response_data and response_data.get("code") == "0" and response_data.get("data"):
            interest_data = response_data["data"]
            logger.info(f"Tool: Fetched {len(interest_data)} open interest entries.")
            return format_data(interest_data, format)
        else:
            logger.warning(f"Tool: No open interest data or error: {response_data}")
            msg = f"No open interest data found or API error: {response_data.get('msg', 'Unknown')}"
            if format == "json":
                return {"error": msg}
            else:
                return f"error: {msg}"
    except (ConnectionError, OKXError, ValueError) as e:
        logger.error(f"Tool: Error fetching open interest: {e}", exc_info=True)
        if format == "json":
            return {"error": f"Failed to fetch open interest: {e}"}
        else:
            return f"error: Failed to fetch open interest: {e}"
    except Exception as e:
        logger.exception(f"Tool: Unexpected error in get_open_interest: {e}")
        if format == "json":
            return {"error": f"An unexpected error occurred: {e}"}
        else:
            return f"error: An unexpected error occurred: {e}"

# --- End of MCP Tools ---