# okx_mcp/services.py
import math
import time
import uuid
import json # Keep if needed by extracted functions
import logging
from typing import Optional, Dict, Any, List, Tuple # Add Tuple if needed
from decimal import Decimal, ROUND_DOWN # Keep if needed by extracted functions

# Use relative imports for client and potentially constants/errors
from .client import OKXClient, OKXError, API_V5_PREFIX

logger = logging.getLogger(__name__)

# --- Service Functions ---

# PASTE and ADAPT get_instruments function here
# - Add 'client: OKXClient' parameter
# - Use client.make_request and API_V5_PREFIX from client module
def get_instruments(client: OKXClient, instrument_type: str, instrument_id: Optional[str] = None) -> Dict[str, Any]:
    """Fetches instrument details from the OKX API via the client."""
    logger.info(f"Service: Fetching instruments - Type: {instrument_type}, ID: {instrument_id}")
    endpoint = f"{API_V5_PREFIX}/public/instruments"
    params = {"instType": instrument_type}
    if instrument_id:
        params["instId"] = instrument_id
    try:
        return client.make_request("GET", endpoint, params=params)
    except (ConnectionError, OKXError) as e:
        logger.error(f"Service: Error fetching instruments: {e}", exc_info=True)
        # Re-raise or return error structure depending on desired handling
        raise # Or return {"error": str(e)}


# PASTE and ADAPT get_instrument_details function here
# - Add 'client: OKXClient' parameter
# - This function should now primarily call client.get_instrument_details_cached
def get_instrument_details(client: OKXClient, instrument_id: str, instrument_type: str = 'SWAP') -> Optional[List[float]]:
    """Service layer function to get instrument details, utilizing the client's cache."""
    logger.debug(f"Service: Requesting cached details for {instrument_id} ({instrument_type})")
    # Directly call the client's cached method
    details = client.get_instrument_details_cached(instrument_id, instrument_type)
    if details is None:
         logger.warning(f"Service: Failed to get instrument details for {instrument_id} ({instrument_type}) from client.")
    return details


# PASTE and ADAPT calculate_contract_size function here
# - Add 'client: OKXClient' parameter
# - Call the service function get_instrument_details(client, ...)
def calculate_contract_size(client: OKXClient, instrument_id: str, usdt_size: float) -> Optional[float]:
    """Calculates the number of contracts based on USDT size."""
    logger.debug(f"Service: Calculating contract size for {instrument_id}, USDT: {usdt_size}")
    # Use the service layer function which uses the client cache
    details = get_instrument_details(client, instrument_id, 'SWAP') # Assuming SWAP for this calc
    if details is None:
        logger.error(f"Service: Cannot calculate contract size, failed to get details for {instrument_id}")
        return None

    contract_price, contract_value, _, _ = details # price, ctVal, lotSz, minSz

    if contract_price <= 0 or contract_value <= 0:
         logger.error(f"Service: Invalid details for {instrument_id}: price={contract_price}, ctVal={contract_value}")
         return None

    try:
        # Calculation: (USDT size / price) / contract value per contract
        contracts = (usdt_size / contract_price) / contract_value
        logger.info(f"Service: Calculated contracts for {instrument_id}: {contracts} (USDT: {usdt_size}, Price: {contract_price}, CtVal: {contract_value})")
        return contracts
    except ZeroDivisionError:
         logger.error(f"Service: Zero division error during contract size calculation for {instrument_id}")
         return None


# PASTE and ADAPT validate_and_correct_order_size function here
# - Add 'client: OKXClient' parameter
# - Call the service function get_instrument_details(client, ...)
# - Ensure Decimal is imported if used
def validate_and_correct_order_size(client: OKXClient, instrument_id: str, size: str, instrument_type: str = 'SWAP') -> str:
    """Validates and corrects order size based on instrument's min size and lot size."""
    logger.debug(f"Service: Validating size '{size}' for {instrument_id} ({instrument_type})")
    details = get_instrument_details(client, instrument_id, instrument_type)
    if details is None:
        logger.error(f"Service: Cannot validate size, failed to get details for {instrument_id}")
        raise ValueError(f"Could not retrieve instrument details for {instrument_id} to validate size.")

    _, _, lot_size, min_size = details # price, ctVal, lotSz, minSz

    try:
        size_decimal = Decimal(str(size))
        min_size_decimal = Decimal(str(min_size))
        lot_size_decimal = Decimal(str(lot_size))

        if size_decimal < min_size_decimal:
            logger.warning(f"Service: Order size {size_decimal} for {instrument_id} is below minimum {min_size_decimal}. Adjusting to minimum.")
            return str(min_size_decimal)

        # Calculate remainder when divided by lot size
        remainder = size_decimal % lot_size_decimal

        if remainder != Decimal(0):
            # Adjust down to the nearest multiple of lot_size
            # precision = lot_size_decimal.as_tuple().exponent * -1 # Get precision
            # corrected_size = (size_decimal // lot_size_decimal) * lot_size_decimal
            # Use quantize for potentially better handling of floating point issues
            corrected_size = (size_decimal - remainder).quantize(lot_size_decimal)

            logger.warning(f"Service: Order size {size_decimal} for {instrument_id} is not a multiple of lot size {lot_size_decimal}. Adjusting down to {corrected_size}.")

            # Final check: ensure corrected size is still >= min_size
            if corrected_size < min_size_decimal:
                 logger.warning(f"Service: Corrected size {corrected_size} is now below minimum {min_size_decimal}. Using minimum size instead.")
                 return str(min_size_decimal)

            return str(corrected_size)
        else:
            # Size is already valid
            logger.debug(f"Service: Size {size_decimal} for {instrument_id} is valid.")
            return str(size_decimal)

    except (ValueError, TypeError) as e:
        logger.error(f"Service: Error validating size '{size}' for {instrument_id}: {e}", exc_info=True)
        raise ValueError(f"Invalid size format '{size}' or instrument details.") from e


# PASTE and ADAPT _convert_usdt_to_contracts function here
# - Add 'client: OKXClient' parameter
# - Call service functions calculate_contract_size(client, ...) and get_instrument_details(client, ...)
# - Ensure Decimal is imported if used
def _convert_usdt_to_contracts(client: OKXClient, instrument: str, usdt_size_str: str) -> str:
    """Converts USDT amount to contract size string, respecting instrument precision."""
    logger.debug(f"Service: Converting USDT '{usdt_size_str}' to contracts for {instrument}")
    try:
        usdt_size = float(usdt_size_str)
        if usdt_size <= 0:
            raise ValueError("USDT size must be positive.")

        # Calculate ideal contract size
        contract_size_float = calculate_contract_size(client, instrument, usdt_size)
        if contract_size_float is None:
            raise ValueError(f"Could not calculate contract size for {instrument}.")

        # Get instrument details for precision (lot size)
        details = get_instrument_details(client, instrument, 'SWAP') # Assuming SWAP
        if details is None:
            raise ValueError(f"Could not get instrument details for {instrument} to determine precision.")

        _, _, lot_size, min_size = details
        lot_size_decimal = Decimal(str(lot_size))
        min_size_decimal = Decimal(str(min_size))
        contract_size_decimal = Decimal(str(contract_size_float))

        # Determine the number of decimal places from lot_size
        precision_places = abs(lot_size_decimal.as_tuple().exponent) if lot_size_decimal.is_finite() else 0

        # Floor the contract size to the required precision
        # Example: If lotSz is 0.01, precision is 2. If contract_size is 12.345, floor to 12.34
        quantizer = Decimal('1e-' + str(precision_places))
        corrected_size = contract_size_decimal.quantize(quantizer, rounding=ROUND_DOWN)

        logger.debug(f"Service: Initial calculated contracts: {contract_size_decimal}, LotSz: {lot_size_decimal}, Precision: {precision_places}, Floored size: {corrected_size}")

        # Ensure the result is not below the minimum size
        if corrected_size < min_size_decimal:
             logger.warning(f"Service: Calculated contract size {corrected_size} for {instrument} is below minimum {min_size_decimal}. Cannot convert.")
             # Depending on requirements, either raise error or return min_size? Raising is safer.
             raise ValueError(f"Calculated contract size {corrected_size} is below minimum order size {min_size_decimal}.")

        # Final validation against lot size (should already be correct due to flooring, but double-check)
        final_size_str = validate_and_correct_order_size(client, instrument, str(corrected_size), 'SWAP')

        logger.info(f"Service: Converted {usdt_size_str} USDT to {final_size_str} contracts for {instrument}")
        return final_size_str

    except (ValueError, TypeError) as e:
        logger.error(f"Service: Error converting USDT to contracts for {instrument}: {e}", exc_info=True)
        raise ValueError(f"Failed to convert USDT size '{usdt_size_str}' to contracts for {instrument}: {e}") from e


# PASTE and ADAPT _place_order_internal function here
# - Add 'client: OKXClient' parameter
# - Use client.make_request
# - Remove calls to get_instrument_details or get_instruments if they are no longer needed here
#   (validation/conversion should happen before calling this)
def _place_order_internal(
    client: OKXClient,
    instrument: str,
    trade_mode: str, # cross, isolated, cash
    side: str,       # buy, sell
    order_type: str, # market, limit, post_only, fok, ioc
    size: str,       # Order size (already validated and corrected)
    order_price: Optional[str] = None, # Required for limit orders
    position_side: Optional[str] = None, # long, short, net (for hedge mode)
    client_order_id: Optional[str] = None,
    tag: Optional[str] = None,
    reduce_only: Optional[bool] = None,
    # is_swap parameter removed, determined by instrument name or context
) -> Dict[str, Any]:
    """Internal helper to place a standard order using the client."""
    logger.info(f"Service: Placing order - Inst: {instrument}, Mode: {trade_mode}, Side: {side}, Type: {order_type}, Size: {size}, Px: {order_price}")

    endpoint = f"{API_V5_PREFIX}/trade/order"
    payload: Dict[str, Any] = {
        "instId": instrument,
        "tdMode": trade_mode,
        "side": side,
        "ordType": order_type,
        "sz": size,
    }
    if order_price and order_type in ["limit", "post_only", "fok", "ioc"]:
        payload["px"] = order_price
    if position_side:
        payload["posSide"] = position_side
    if client_order_id:
        payload["clOrdId"] = client_order_id
    if tag:
        payload["tag"] = tag
    if reduce_only is not None:
        payload["reduceOnly"] = reduce_only

    try:
        # Make the authenticated request using the client
        response_data = client.make_request("POST", endpoint, data=payload, auth=True)
        logger.info(f"Service: Order placement response for {instrument}: {response_data}")

        # Basic check for success indication in response
        if response_data.get("code") == "0" and response_data.get("data") and response_data["data"][0].get("sCode") == "0":
            logger.info(f"Service: Order placed successfully for {instrument}. Order ID: {response_data['data'][0].get('ordId')}")
        else:
            # Log the full error details if available
            error_info = response_data.get("data", [{}])[0]
            sCode = error_info.get("sCode", "N/A")
            sMsg = error_info.get("sMsg", "Unknown error")
            logger.error(f"Service: Order placement failed for {instrument}. sCode: {sCode}, sMsg: {sMsg}. Full response: {response_data}")
            # Consider raising an OKXError here based on sCode/sMsg if needed by callers
            # raise OKXError(code=sCode, message=sMsg)

        return response_data # Return the full response

    except (ConnectionError, OKXError, ValueError, TypeError) as e:
        logger.error(f"Service: Error placing order for {instrument}: {e}", exc_info=True)
        raise # Re-raise the caught exception


# PASTE and ADAPT _place_grid_algo_order_internal function here
# - Add 'client: OKXClient' parameter
# - Use client.make_request
# - The 'data' parameter should contain the specific grid parameters
def _place_grid_algo_order_internal(
    client: OKXClient,
    # Remove individual grid params like instrument_id, max_price etc.
    # They should be included in the 'data' dict by the caller (tool)
    data: Dict[str, Any] # This dict contains all necessary grid parameters
) -> Dict[str, Any]:
    """Internal helper to place or amend grid algo orders using the client."""
    # Infer action from endpoint? Or pass explicitly? Let's assume endpoint implies action.
    # This function might need splitting if amend/stop have different base endpoints.
    # Assuming the caller provides the correct endpoint in 'data' or we determine it.
    # For now, let's assume it's for placing new orders.

    instrument_id = data.get("instId", "N/A")
    algo_order_type = data.get("algoOrdType", "grid") # Default to grid if not specified
    logger.info(f"Service: Placing {algo_order_type} algo order for {instrument_id}")
    logger.debug(f"Service: Algo order data: {data}")

    # Endpoint for placing grid orders
    endpoint = f"{API_V5_PREFIX}/tradingBot/grid/order-algo"

    try:
        # Make the authenticated request using the client
        response_data = client.make_request("POST", endpoint, data=data, auth=True)
        logger.info(f"Service: Algo order placement response for {instrument_id}: {response_data}")

        # Basic check for success
        if response_data.get("code") == "0" and response_data.get("data") and response_data["data"][0].get("sCode") == "0":
             algo_id = response_data['data'][0].get('algoId')
             logger.info(f"Service: Algo order placed successfully for {instrument_id}. Algo ID: {algo_id}")
        else:
            error_info = response_data.get("data", [{}])[0]
            sCode = error_info.get("sCode", "N/A")
            sMsg = error_info.get("sMsg", "Unknown error")
            logger.error(f"Service: Algo order placement failed for {instrument_id}. sCode: {sCode}, sMsg: {sMsg}. Full response: {response_data}")
            # raise OKXError(code=sCode, message=sMsg)

        return response_data

    except (ConnectionError, OKXError, ValueError, TypeError) as e:
        logger.error(f"Service: Error placing algo order for {instrument_id}: {e}", exc_info=True)
        raise

# --- End of Service Functions ---