# okx_mcp/tools.py
import logging
import json
from typing import Optional, Dict, Any, List, Union
from fastmcp import FastMCP

# Relative imports for client, services, and error
from .client import OKXClient, OKXError, API_V5_PREFIX
from . import services # Import the services module
from .utils import format_data # Import format_data from utils

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

# --- MCP Tools ---

@mcp.tool()
def get_price(instrument: str, format: str = "json") -> Any:
    """Get the latest market ticker price for a specific instrument (e.g., BTC-USDT-SWAP). Supports json and md formats."""
    # Validate format
    if format.lower() not in ["json", "md"]:
        logger.warning(f"Tool: Unsupported format requested: {format}. Defaulting to json.")
        format = "json"
        
    logger.info(f"Tool: Fetching price for instrument: {instrument}. Format: {format}")
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
            return format_data(result, format)
        else:
            logger.warning(f"Tool: No price data returned for {instrument}")
            error_msg = f"No data found for instrument {instrument}"
            if format == "json":
                return {"error": error_msg}
            else:
                return f"error: {error_msg}"
    except (ConnectionError, OKXError, ValueError) as e:
        logger.error(f"Tool: Error fetching price for {instrument}: {e}", exc_info=True)
        if format == "json":
            return {"error": f"Failed to fetch price for {instrument}: {e}"}
        else:
            return f"error: Failed to fetch price for {instrument}: {e}"
    except Exception as e:
        logger.exception(f"Tool: Unexpected error in get_price for {instrument}: {e}")
        if format == "json":
            return {"error": f"An unexpected error occurred: {e}"}
        else:
            return f"error: An unexpected error occurred: {e}"


@mcp.tool()
def get_candlesticks(instrument: str, bar: str = "1m", limit: int = 100, format: str = "json") -> Any:
    """
    Get candlestick (k-line) data for an OKX instrument and return it as formatted text.

    Args:
        instrument: The instrument ID (e.g., BTC-USDT, BTC-USDT-SWAP).
        bar: Candlestick interval (e.g., 1m, 5m, 1H, 1D). Default is "1m".
        limit: Number of candlesticks to retrieve (max typically 100). Default is 100.
        format: Output format (json, csv, md, yaml). Default is "json".

    Returns:
        A formatted representation of candlestick data containing timestamp, open, high, low, close,
        volume in contracts, and volume in USDT.
        Returns empty data structure if no candlesticks found matching criteria.

    Raises:
        ConnectionError: If the API request fails.
        OKXError: If the OKX API returns an error.
        ValueError: If the parameters are invalid or response format is unexpected.

    Example:
        >>> get_candlesticks("BTC-USDT", "15m", 50, "csv")
        "timestamp_ms,open,high,low,close,volume_contracts,volume_usdt
         1598918400000,11469.8,11470.9,11469.8,11470.9,2,0.17431"
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
def get_account_balance(format: str = "json") -> Any:
    """Get account balance information (total equity, available balance per currency). Supports json and md formats."""
    # Validate format
    if format.lower() not in ["json", "md"]:
        logger.warning(f"Tool: Unsupported format requested: {format}. Defaulting to json.")
        format = "json"
        
    logger.info(f"Tool: Fetching account balance. Format: {format}")
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
            
            # Create a structured result dictionary
            result = {
                "total_equity_usd": total_equity,
                "currency_balances": {}
            }
            
            # Format the balances for better readability
            for item in details:
                ccy = item.get("ccy", "Unknown")
                result["currency_balances"][ccy] = {
                    "equity": item.get("eq"),
                    "available": item.get("availEq"),
                    "frozen": item.get("frozenBal", "0")
                }
            
            logger.info(f"Tool: Account balance fetched. Total Equity: {total_equity}")
            return format_data(result, format)
        else:
            logger.warning("Tool: No account balance data returned.")
            # Format the error message based on the requested format
            error_msg = "No account balance data found."
            if format == "json":
                return {"error": error_msg}
            else:
                return f"error: {error_msg}"
    except (ConnectionError, OKXError, ValueError) as e:
        logger.error(f"Tool: Error fetching account balance: {e}", exc_info=True)
        if format == "json":
            return {"error": f"Failed to fetch account balance: {e}"}
        else:
            return f"error: Failed to fetch account balance: {e}"
    except Exception as e:
        logger.exception(f"Tool: Unexpected error in get_account_balance: {e}")
        if format == "json":
            return {"error": f"An unexpected error occurred: {e}"}
        else:
            return f"error: An unexpected error occurred: {e}"


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
    instrument_type: Optional[str] = None,
    instrument_id: Optional[str] = None,
    order_id: Optional[str] = None,
    limit: int = 100,
    format: str = "json"
) -> Any:
     """
     Get recent trade (fill) history.
     Returns data in the specified format (json, csv, md, yaml). Default is json.
     """
     logger.info(f"Tool: Fetching trade history - Type: {instrument_type}, ID: {instrument_id}, Limit: {limit}, Format: {format}")
     endpoint = f"{API_V5_PREFIX}/trade/fills"
     params: Dict[str, Any] = {"limit": str(limit)}
     if instrument_type:
         params["instType"] = instrument_type
     if instrument_id:
         params["instId"] = instrument_id
     if order_id:
         params["ordId"] = order_id

     try:
         response_data = okx_client.make_request("GET", endpoint, params=params, auth=True)
         if response_data and response_data.get("code") == "0":
             trades = response_data.get("data", [])
             logger.info(f"Tool: Fetched {len(trades)} trade history entries.")
             
             # Format the data using the abstraction
             return format_data(trades, format)
         else:
             logger.warning(f"Tool: No trade history data returned or error in response: {response_data}")
             # Return appropriate format-based error message
             error_msg = "No trade history found or API error occurred"
             if format == "json":
                 return {"error": error_msg}
             else:
                 return f"error: {error_msg}"
     except (ConnectionError, OKXError, ValueError) as e:
         logger.error(f"Tool: Error fetching trade history: {e}", exc_info=True)
         if format == "json":
             return [{"error": f"Failed to fetch trade history: {e}"}]
         else:
             return f"error: Failed to fetch trade history: {e}"
     except Exception as e:
         logger.exception(f"Tool: Unexpected error in get_trade_history: {e}")
         if format == "json":
             return [{"error": f"An unexpected error occurred: {e}"}]
         else:
             return f"error: An unexpected error occurred: {e}"


@mcp.tool()
def place_swap_limit_order(
    instrument: str,
    side: str, # buy, sell
    size_usdt: str, # Order size in USDT
    price: str,
    position_side: str, # long, short
    stop_loss_price: Optional[str] = None,
    take_profit_price: Optional[str] = None
) -> Dict[str, Any]:
    """
    Place a limit order for SWAP trading on OKX, specifying size in USDT.
    
    Args:
        instrument: The instrument ID (e.g., BTC-USDT-SWAP). Must end with '-SWAP'.
        side: Order side, must be either "buy" or "sell".
        size_usdt: Order size in USDT (e.g., "100" for 100 USDT). Will be converted to contract quantity.
        price: Limit price for the order (e.g., "30000" for $30,000).
        position_side: Position side, must be either "long" or "short".
        stop_loss_price: Optional stop loss price.
        take_profit_price: Optional take profit price.
        
    Returns:
        A dictionary containing order details including orderId, clientOrderId, and other relevant information.
        Returns an error dictionary if the order placement fails.
        
    Raises:
        OKXError: If the OKX API returns an error (includes error code and message).
        ValueError: If parameters are invalid (e.g., instrument not ending with '-SWAP').
        ConnectionError: If the API request fails.
        
    Example:
        >>> place_swap_limit_order(
        ...     instrument="BTC-USDT-SWAP",
        ...     side="buy",
        ...     size_usdt="100",
        ...     price="30000",
        ...     position_side="long",
        ...     stop_loss_price="29000",
        ...     take_profit_price="32000"
        ... )
        {'ordId': '123456789', 'clOrdId': 'okx_123456789', 'tag': '', 'sCode': '0', 'sMsg': ''}
    """
    # Strip whitespace from instrument to prevent errors
    instrument = instrument.strip()
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
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price
        )
    except OKXError as e:
        logger.error(f"Tool: OKX API Error placing SWAP limit order for {instrument}: {e}", exc_info=True)
        # Return full raw error details for OKX API errors
        return {
            "error": f"Failed to place SWAP limit order: {e}",
            "error_code": getattr(e, "code", None),
            "error_message": getattr(e, "message", str(e)),
            "error_data": getattr(e, "data", None),
            "error_raw": str(e),
            "error_type": "OKXError"
        }
    except ValueError as e:
        logger.error(f"Tool: Value Error placing SWAP limit order for {instrument}: {e}", exc_info=True)
        return {"error": f"Failed to place SWAP limit order: {e}"}
    except ConnectionError as e:
        logger.error(f"Tool: Connection Error placing SWAP limit order for {instrument}: {e}", exc_info=True)
        return {"error": f"Failed to connect to OKX API: {e}"}
    except Exception as e:
        logger.exception(f"Tool: Unexpected error in place_swap_limit_order for {instrument}: {e}")
        return {"error": f"An unexpected error occurred: {e}"}


@mcp.tool()
def place_spot_limit_order(
    instrument: str,
    side: str, # buy, sell
    size: str, # Order size in base currency (e.g., BTC amount for BTC-USDT)
    price: str,
    stop_loss_price: Optional[str] = None,
    take_profit_price: Optional[str] = None 
) -> Dict[str, Any]:
    """
    Place a limit order for SPOT trading on OKX.
    
    Args:
        instrument: The instrument ID (e.g., BTC-USDT). Must not contain '-SWAP'.
        side: Order side, must be either "buy" or "sell".
        size: Order size in base currency (e.g., BTC amount for BTC-USDT).
        price: Limit price for the order.
        stop_loss_price: Optional stop loss price.
        take_profit_price: Optional take profit price.
        trade_mode: Trading mode, either "cash" or "cross". Default is "cash".
        
    Returns:
        A dictionary containing order details including orderId, clientOrderId, and other relevant information.
        Returns an error dictionary if the order placement fails.
        
    Raises:
        OKXError: If the OKX API returns an error.
        ValueError: If parameters are invalid.
        ConnectionError: If the API request fails.
        
    Example:
        >>> place_spot_limit_order(
        ...     instrument="BTC-USDT",
        ...     side="buy",
        ...     size="0.01",
        ...     price="30000",
        ...     stop_loss_price="29000",
        ...     take_profit_price="32000"
        ... )
        {'ordId': '123456789', 'clOrdId': 'okx_123456789', 'tag': '', 'sCode': '0', 'sMsg': ''}
    """
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
            trade_mode="cash",
            side=side,
            order_type="limit",
            size=corrected_size,
            order_price=price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price
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
    grid_run_type: str = "1",
    investment_amount: Optional[str] = None,
    base_order_size: Optional[str] = None, 
    tp_trigger_px: Optional[str] = None,
    sl_trigger_px: Optional[str] = None
) -> Dict[str, Any]:
    """
    Place a spot grid algo order on OKX.
    
    A grid trading strategy divides the price range into several price levels and 
    automatically buys at lower levels and sells at higher levels.
    
    Args:
        instrument_id: Instrument ID (e.g., "BTC-USDT").
        max_price: Upper price of the grid range.
        min_price: Lower price of the grid range.
        grid_num: Number of grids (integer as string).
        grid_run_type: Grid type: "1" (Arithmetic), "2" (Geometric). Default is "1".
        investment_amount: Investment amount in quote currency (e.g., USDT).
            Required if base_order_size is not set.
        base_order_size: Investment amount in base currency (e.g., BTC).
            Required if investment_amount is not set.
        tp_trigger_px: Take-profit trigger price.
        sl_trigger_px: Stop-loss trigger price.
    
    Returns:
        A dictionary containing order details including algoId and client-supplied algoId.
        Example: {"algoId": "448965992920907776", "algoClOrdId": "", "sCode": "0", "sMsg": ""}
        
        If an error occurs, returns a dictionary with error information:
        Example: {"error": "Failed to place SPOT Grid Algo order: Invalid parameters"}
    
    Raises:
        ValueError: If neither investment_amount nor base_order_size is provided,
                   or if both are provided, or if parameters are otherwise invalid.
        ConnectionError: If the API request fails.
        OKXError: If the OKX API returns an error.
    
    Example:
        >>> place_spot_grid_algo_order(
        ...     instrument_id="BTC-USDT",
        ...     max_price="35000",
        ...     min_price="25000",
        ...     grid_num="10",
        ...     investment_amount="1000",
        ...     tp_trigger_px="40000"
        ... )
        {"algoId": "448965992920907776", "algoClOrdId": "", "sCode": "0", "sMsg": ""}
    """
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
    algo_order_type: str, # grid, contract_grid, etc.
    instrument_type: str = "SPOT", # SPOT, SWAP, FUTURES, OPTION
    instrument_id: Optional[str] = None,
    algo_id: Optional[str] = None,
    state: Optional[str] = None, # effective, paused, stopping
    after: Optional[str] = None, # pagination cursor
    before: Optional[str] = None, # pagination cursor
    limit: int = 100,
    format: str = "json" # json, csv, md, yaml
) -> Any:
    """
    Get a list of pending grid algo orders and return it as formatted text (requires authentication).

    Args:
        algo_order_type: Algo order type: "grid" (spot) or "contract_grid" (contract).
        instrument_type: Filter by Instrument type: "SPOT", "MARGIN", "FUTURES", "SWAP". Default is "SPOT".
        instrument_id: Filter by Instrument ID (e.g., "BTC-USDT"). Optional.
        algo_id: Filter by Algo ID. Optional.
        state: Filter by state: "starting", "running", "stopping", "pending_signal", "no_close_position". Optional.
        after: Pagination: Return records earlier than the requested algoId. Optional.
        before: Pagination: Return records newer than the requested algoId. Optional.
        limit: Number of results per request (max 100, default 100). Optional.
        format: Output format (json, csv, md, yaml). Default is "json".

    Returns:
        A formatted representation of pending grid algo orders containing details like:
        algoId, clientAlgoId, instrument, state, maxPrice, minPrice, gridNum, etc.
        Returns appropriate empty/error structure if no orders found or an error occurs.

    Raises:
        ValueError: If parameters are invalid.
        PermissionError: If authentication fails.
        ConnectionError: If API request fails.
        OKXError: If OKX API returns an error.

    Example:
        >>> get_grid_algo_order_list("grid", instrument_type="SPOT", format="csv")
        "AlgoID,ClientAlgoID,Instrument,InstType,AlgoType,State,MaxPrice,MinPrice,GridNum,RunType,..."
    """
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
    if after: params["after"] = after
    if before: params["before"] = before

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


@mcp.tool()
def place_contract_grid_algo_order(
    instrument_id: str,
    max_price: str,
    min_price: str,
    grid_num: str,
    grid_run_type: str = "1", 
    investment_amount: Optional[str] = None, 
    contract_order_size: Optional[str] = None, 
    tp_trigger_px: Optional[str] = None,
    sl_trigger_px: Optional[str] = None
) -> Dict[str, Any]:
    """
    Place a contract grid algo order on OKX (for SWAP/FUTURES).
    
    A contract grid trading strategy divides the price range into several price levels and
    automatically executes trades at predefined price points within the SWAP/FUTURES markets.
    
    Args:
        instrument_id: Instrument ID (e.g., "BTC-USDT-SWAP"). Must end with '-SWAP' or '-FUTURES'.
        max_price: Upper price of the grid range.
        min_price: Lower price of the grid range.
        grid_num: Number of grids (integer as string).
        grid_run_type: Grid type: "1" (Arithmetic), "2" (Geometric). Default is "1".
        investment_amount: Investment amount in quote currency (e.g., USDT).
            Required if contract_order_size is not set.
        contract_order_size: Size per grid order in contracts.
            Required if investment_amount is not set.
        tp_trigger_px: Take-profit trigger price.
        sl_trigger_px: Stop-loss trigger price.
    
    Returns:
        A dictionary containing order details including algoId and client-supplied algoId.
        Example: {"algoId": "448965992920907776", "algoClOrdId": "", "sCode": "0", "sMsg": ""}
        
        If an error occurs, returns a dictionary with error information:
        Example: {"error": "Failed to place Contract Grid Algo order: Invalid instrument"}
    
    Raises:
        ValueError: If neither investment_amount nor contract_order_size is provided,
                   or if both are provided, or if instrument_id doesn't end with '-SWAP' or '-FUTURES'.
        ConnectionError: If the API request fails.
        OKXError: If the OKX API returns an error.
    
    Example:
        >>> place_contract_grid_algo_order(
        ...     instrument_id="BTC-USDT-SWAP",
        ...     max_price="35000",
        ...     min_price="25000",
        ...     grid_num="10",
        ...     investment_amount="1000",
        ...     tp_trigger_px="40000"
        ... )
        {"algoId": "448965992920907777", "algoClOrdId": "", "sCode": "0", "sMsg": ""}
    """
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
        "instType": "SWAP" if instrument_id.endswith('-SWAP') else "FUTURES", 
    }
    if investment_amount:
        data["quoteSz"] = investment_amount 
    if contract_order_size:
         data["baseSz"] = contract_order_size # Use baseSz for size per grid (contracts)
    if tp_trigger_px:
        data["tpTriggerPx"] = tp_trigger_px
    if sl_trigger_px:
        data["slTriggerPx"] = sl_trigger_px

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
    tp_ratio: Optional[str] = None,
    sl_ratio: Optional[str] = None
) -> Dict[str, Any]:
    """
    Amend an existing grid algo order on OKX (requires authentication).
    
    Allows modifying parameters of a running grid algo order such as take-profit price,
    stop-loss price, or adding a new client order ID.
    
    Args:
        algo_id: The Algo ID of the order to amend.
        instrument_id: Instrument ID associated with the order (e.g., "BTC-USDT", "BTC-USDT-SWAP").
        max_price: New upper price of the grid range. Optional.
        min_price: New lower price of the grid range. Optional.
        tp_trigger_px: New take-profit trigger price. Set to "" to cancel existing take-profit. Optional.
        sl_trigger_px: New stop-loss trigger price. Set to "" to cancel existing stop-loss. Optional.
        tp_ratio: New take-profit ratio for contract grid orders (e.g., "0.1" for 10%). Set to "" to cancel. Optional.
        sl_ratio: New stop-loss ratio for contract grid orders (e.g., "0.1" for 10%). Set to "" to cancel. Optional.
    
    Returns:
        A dictionary containing the amended order details.
        Example: {"algoId": "448965992920907776", "sCode": "0", "sMsg": ""}
        
        If an error occurs, returns a dictionary with error information:
        Example: {"error": "Failed to amend grid algo order: Invalid parameters"}
    
    Raises:
        ValueError: If no amendable parameters are provided or parameters are invalid.
        PermissionError: If authentication fails.
        ConnectionError: If the API request fails.
        OKXError: If the OKX API returns an error.
    
    Example:
        >>> amend_grid_algo_order(
        ...     algo_id="448965992920907776",
        ...     instrument_id="BTC-USDT-SWAP",
        ...     sl_trigger_px="29000",
        ...     tp_trigger_px="35000"
        ... )
        {"algoId": "448965992920907776", "sCode": "0", "sMsg": ""}
    """
    logger.info(f"Tool: Amending Grid Algo order {algo_id} for {instrument_id}")
    if not any([max_price, min_price, tp_trigger_px, sl_trigger_px, tp_ratio, sl_ratio, new_client_order_id]):
        return {"error": "At least one parameter (max_price, min_price, tp_trigger_px, sl_trigger_px, tp_ratio, sl_ratio, new_client_order_id) must be provided to amend."}

    data: Dict[str, Any] = {
        "algoId": algo_id,
        "instId": instrument_id,
    }
    if max_price: data["maxPx"] = max_price
    if min_price: data["minPx"] = min_price
    if tp_trigger_px: data["tpTriggerPx"] = tp_trigger_px
    if sl_trigger_px: data["slTriggerPx"] = sl_trigger_px
    if tp_ratio: data["tpRatio"] = tp_ratio
    if sl_ratio: data["slRatio"] = sl_ratio

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
    algo_order_type: str # grid, contract_grid
) -> Dict[str, Any]:
    """
    Get detailed information for a specific grid algo order (requires authentication).
    
    Retrieves comprehensive details about a grid algo order including its configuration,
    current performance metrics, and status information.
    
    Args:
        algo_id: The Algo ID of the order to retrieve details for.
        algo_order_type: Algo order type: "grid" (spot) or "contract_grid" (contract).
    
    Returns:
        A dictionary containing detailed information about the specified grid algo order including:
        - Basic order information (algoId, instId, cTime, uTime, state)
        - Configuration parameters (maxPx, minPx, gridNum, runType, etc.)
        - Performance metrics (totalPnl, gridProfit, floatProfit, arbitrageNum, etc.)
        - Instrument-specific details (for Spot or Contract)
        - Current status indicators
        
        If an error occurs, returns a dictionary with error information:
        Example: {"error": "Failed to retrieve grid algo details: Order not found"}
    
    Raises:
        ValueError: If parameters are invalid or the order is not found.
        PermissionError: If authentication fails.
        ConnectionError: If API request fails.
        OKXError: If OKX API returns an error.
    
    Example:
        >>> get_grid_algo_order_details("448965992920907776", "grid")
        {
            "algoId": "448965992920907776",
            "instId": "BTC-USDT",
            "state": "running",
            "maxPx": "35000",
            "minPx": "25000",
            "gridNum": "10",
            "totalPnl": "12.5",
            "gridProfit": "10.2",
            "floatProfit": "2.3",
            ...
        }
    """
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