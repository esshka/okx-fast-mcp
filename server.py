# server.py
import os
import time
import hmac
import base64
import json
import requests
import hashlib
import logging
import math
import uuid
from typing import Optional, Dict, Any, List
from fastmcp import FastMCP

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,  # Set default level to INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
# Set requests logger level to WARNING to avoid overly verbose logs from the library
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# --- FastMCP Setup ---
mcp = FastMCP("OKX API 🚀")

# --- Constants ---
OKX_BASE_URL = "https://www.okx.com"
API_V5_PREFIX = "/api/v5"

class OKXError(Exception):
    """Custom exception for OKX API errors."""
    def __init__(self, message: str, code: Optional[str] = None, status_code: Optional[int] = None):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.message = message

    def __str__(self):
        # Provide more context in the error message
        if self.code:
            return f"OKX API Error (code={self.code}): {self.message}"
        elif self.status_code:
            return f"OKX HTTP Error (status={self.status_code}): {self.message}"
        else:
            return f"OKX Error: {self.message}"

class OKXClient:
    """
    Client for interacting with the OKX API (v5).
    Handles authentication and request signing for private endpoints.
    """
    def __init__(self):
        """Initializes the OKXClient, loading credentials from environment variables."""
        logger.info("Initializing OKX MCP server...")

        # Load credentials securely
        self.api_key = os.environ.get("OKX_API_KEY")
        self.secret_key = os.environ.get("OKX_SECRET_KEY")
        self.passphrase = os.environ.get("OKX_PASSPHRASE")

        self.has_private_access = bool(self.api_key and self.secret_key and self.passphrase)

        if self.has_private_access:
            logger.info("Private API access configured.")
            logger.debug(f"API Key loaded: {self.api_key}")
            # Secret key might be base64 encoded, try decoding
            if self.secret_key:
                try:
                    decoded_secret = base64.b64decode(self.secret_key)
                    # Check if decoding resulted in a plausible key length (e.g., 64 for SHA256)
                    # This check is heuristic and might need adjustment based on actual key formats
                    if len(decoded_secret) > 16: # Arbitrary length check
                         logger.debug("Attempting to use base64 decoded secret key.")
                         self.secret_key = decoded_secret.decode('utf-8')
                    else:
                         logger.debug("Decoded secret key seems too short, using original.")
                except (base64.binascii.Error, UnicodeDecodeError):
                    logger.debug("Secret key is not base64 encoded or decode failed, using as is.")
                logger.debug(f"Secret Key loaded: {self.secret_key[:3]}...{self.secret_key[-3:]}") # Log masked key
            if self.passphrase:
                 logger.debug(f"Passphrase loaded: {self.passphrase[:3]}...{self.passphrase[-3:]}") # Log masked passphrase
        else:
            logger.warning("Private API access not configured. Only public endpoints are available.")
            logger.warning("Set OKX_API_KEY, OKX_SECRET_KEY, and OKX_PASSPHRASE environment variables for private access.")

        self.base_url = OKX_BASE_URL

    def _get_timestamp(self) -> str:
        """Generates the required ISO 8601 timestamp format for OKX API."""
        return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())

    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """Generates the API request signature."""
        if not self.secret_key:
            raise ValueError("Secret key is required for signature generation but not configured.")

        message = timestamp + method.upper() + request_path + body
        logger.debug(f"Generating signature for message: '{message}'")

        try:
            mac = hmac.new(
                bytes(self.secret_key, encoding='utf8'),
                bytes(message, encoding='utf8'),
                digestmod=hashlib.sha256
            )
            signature = base64.b64encode(mac.digest()).decode('utf8')
            logger.debug(f"Generated signature: {signature}")
            return signature
        except Exception as e:
            logger.error(f"Error generating signature: {e}", exc_info=True)
            raise RuntimeError("Failed to generate API signature.") from e

    def _add_auth_headers(self, headers: Dict[str, str], method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, body: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Adds authentication headers to the request."""
        if not self.has_private_access:
            raise PermissionError("API credentials not configured for private endpoint access.")

        timestamp = self._get_timestamp()

        # Construct the request path including query parameters for GET requests
        full_request_path = endpoint
        if method.upper() == "GET" and params:
            # OKX requires query parameters to be sorted alphabetically for signature
            query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
            if query_string:
                full_request_path += "?" + query_string

        # Prepare body string for signature (must be exactly as sent in the request)
        signature_body = ""
        if method.upper() != "GET" and body:
            # Use compact JSON serialization without extra whitespace
            signature_body = json.dumps(body, separators=(',', ':'))

        signature = self._generate_signature(timestamp, method, full_request_path, signature_body)

        auth_headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "x-simulated-trading": "0"  # Use "1" for demo/paper trading
        }
        headers.update(auth_headers)

        # Log headers safely (mask sensitive parts)
        log_headers = headers.copy()
        log_headers["OK-ACCESS-SIGN"] = "***"
        log_headers["OK-ACCESS-PASSPHRASE"] = "***"
        logger.debug(f"Auth headers added: {json.dumps(log_headers)}")

        return headers

    def make_request(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None, auth: bool = False) -> Dict[str, Any]:
        """
        Makes an HTTP request to the specified OKX API endpoint.

        Args:
            method: HTTP method (GET, POST, etc.).
            endpoint: API endpoint path (e.g., /api/v5/market/ticker).
            params: URL query parameters.
            data: Request body data (for POST, PUT, etc.).
            auth: Whether the endpoint requires authentication.

        Returns:
            The JSON response data from the API.

        Raises:
            ConnectionError: If the request fails due to network issues.
            PermissionError: If auth=True but credentials are not configured.
            OKXError: If the API returns an error (non-zero code).
            ValueError: If the response is not valid JSON or required keys are missing.
            RuntimeError: For unexpected errors during the request.
        """
        if not endpoint.startswith(API_V5_PREFIX):
             logger.warning(f"Endpoint '{endpoint}' does not start with '{API_V5_PREFIX}'. Prepending it.")
             endpoint = API_V5_PREFIX + endpoint

        url = self.base_url + endpoint
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8", # Specify charset
        }

        request_body = None
        if data:
            # Use compact JSON serialization for the request body
            request_body = json.dumps(data, separators=(',', ':'))

        if auth:
            try:
                headers = self._add_auth_headers(headers, method, endpoint, params, data) # Pass original data dict
            except PermissionError as pe:
                logger.error(f"Authentication required but failed: {pe}")
                raise

        logger.info(f"Making API request: {method.upper()} {endpoint}")
        logger.debug(f"URL: {url}")
        if params:
            logger.debug(f"Params: {json.dumps(params)}")
        if request_body:
            # Log request body carefully, potentially truncating large bodies
            log_body = request_body[:500] + ('...' if len(request_body) > 500 else '')
            logger.debug(f"Body: {log_body}")
        # Log headers safely (mask sensitive parts again if not already done)
        log_headers = headers.copy()
        if "OK-ACCESS-SIGN" in log_headers: log_headers["OK-ACCESS-SIGN"] = "***"
        if "OK-ACCESS-PASSPHRASE" in log_headers: log_headers["OK-ACCESS-PASSPHRASE"] = "***"
        logger.debug(f"Headers: {json.dumps(log_headers)}")

        # Store request details for error reporting
        request_details = {
            "method": method.upper(),
            "url": url,
            "params": params,
            "headers": log_headers,
            "body": log_body if request_body else None
        }

        try:
            response = requests.request(
                method=method.upper(),
                url=url,
                params=params,
                data=request_body, # Send the serialized JSON string
                headers=headers,
                timeout=15 # Slightly increased timeout
            )

            logger.debug(f"Response Status Code: {response.status_code}")
            logger.debug(f"Response Headers: {dict(response.headers)}")

            response_text = response.text
            # Log response body carefully, potentially truncating large bodies
            log_response_body = response_text[:500] + ('...' if len(response_text) > 500 else '')
            logger.debug(f"Response Body: {log_response_body}")

            # Attempt to parse JSON response
            try:
                response_data = response.json()
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse JSON response: {response_text}", exc_info=True)
                logger.error(f"Request details: {json.dumps(request_details)}")
                raise ValueError(f"Invalid JSON response received from API: {str(json_err)}") from json_err

            # Check HTTP status code first
            if response.status_code != 200:
                error_msg = response_data.get("msg", "Unknown HTTP error")
                error_code = response_data.get("code", None)
                logger.error(f"OKX API HTTP error: Status={response.status_code}, Code={error_code}, Message={error_msg}")
                logger.error(f"Request details: {json.dumps(request_details)}")
                raise OKXError(message=error_msg, code=error_code, status_code=response.status_code)

            # Check OKX specific error code in the response body
            api_code = response_data.get("code")
            if api_code != "0":
                error_msg = response_data.get("msg", "Unknown OKX API error")
                logger.error(f"OKX API returned an error: Code={api_code}, Message={error_msg}")
                logger.error(f"Request details: {json.dumps(request_details)}")
                
                # Translate common error codes to more user-friendly messages
                if api_code == "50111":
                    error_msg = f"Invalid API key or permissions: {error_msg}"
                elif api_code == "50001":
                    error_msg = f"Insufficient balance: {error_msg}"
                elif api_code in ["51002", "51008"]:
                    error_msg = f"Order size issue: {error_msg}"
                elif api_code == "51010": 
                    error_msg = f"Invalid order price: {error_msg}"
                elif api_code == "51015":
                    error_msg = f"Rate limit exceeded: {error_msg}"
                elif api_code == "50012":
                    error_msg = f"Leverage setting issue: {error_msg}"
                elif api_code == "50006":
                    error_msg = f"Account in liquidation or position issue: {error_msg}"
                elif api_code == "1":
                    error_msg = f"Operation failed, verify instrument ID and parameters: {error_msg}"
                    # Give more detailed suggestions for error code 1
                    if "instId" in request_body or (params and "instId" in params):
                        error_msg += "\nPossible solutions:\n"
                        error_msg += "1. Verify the instrument ID exists and is correctly formatted (e.g., 'SOL-USDT-SWAP')\n"
                        error_msg += "2. Ensure the order size meets minimum requirements and is a multiple of the lot size\n"
                        error_msg += "3. For limit orders, ensure the price is a multiple of the instrument's tick size\n"
                        error_msg += "4. Check if your account has sufficient balance and proper leverage settings\n"
                        error_msg += "5. Verify you have permission to trade this instrument"
                
                # Include request details in the error message
                error_with_details = f"{error_msg}\nRequest: {method.upper()} {url}"
                if params:
                    error_with_details += f"\nParams: {json.dumps(params)}"
                if request_body:
                    error_with_details += f"\nBody: {log_body}"
                
                raise OKXError(message=error_with_details, code=api_code, status_code=response.status_code)

            logger.debug("API request successful.")
            return response_data

        except requests.exceptions.Timeout:
            logger.error(f"Request timed out: {method.upper()} {url}", exc_info=True)
            logger.error(f"Request details: {json.dumps(request_details)}")
            raise ConnectionError(f"Request timed out connecting to OKX API endpoint: {endpoint}\nRequest details: {json.dumps(request_details)}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request failed: {method.upper()} {url} - {req_err}", exc_info=True)
            logger.error(f"Request details: {json.dumps(request_details)}")
            raise ConnectionError(f"Failed to connect to OKX API: {str(req_err)}\nRequest details: {json.dumps(request_details)}") from req_err
        except (OKXError, ValueError, PermissionError) as api_err:
            # Re-raise specific API or value errors
            raise api_err
        except Exception as e:
            logger.exception(f"An unexpected error occurred during the API request to {endpoint}: {e}")
            logger.error(f"Request details: {json.dumps(request_details)}")
            raise RuntimeError(f"An unexpected error occurred while making the API request.\nRequest details: {json.dumps(request_details)}") from e

    def get_instruments(self, instrument_type: str, instrument_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch instrument information from OKX API.
        
        Args:
            instrument_type: The instrument type (e.g., 'SWAP', 'SPOT', 'FUTURES')
            instrument_id: Specific instrument ID (optional)
            
        Returns:
            Dictionary containing instrument information.
            
        Raises:
            ConnectionError: If the API request fails.
            OKXError: If the API returns an error.
            ValueError: If the response format is unexpected.
        """
        logger.info(f"Fetching instrument information for type: {instrument_type}" + 
                    (f", ID: {instrument_id}" if instrument_id else ""))
        
        endpoint = f"{API_V5_PREFIX}/public/instruments"
        params = {"instType": instrument_type}
        
        if instrument_id:
            params["instId"] = instrument_id
            
        try:
            return self.make_request(
                method="GET",
                endpoint=endpoint,
                params=params
            )
        except (ConnectionError, OKXError, ValueError) as e:
            logger.error(f"Error fetching instrument information: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.exception(f"An unexpected error occurred while fetching instrument information: {e}")
            raise RuntimeError(f"An unexpected error occurred while fetching instrument information.") from e
    
    # Simple in-memory cache for instrument details
    _instrument_details_cache = {}
    _CACHE_TTL_SECONDS = 60  # Cache TTL of 60 seconds
    
    def get_instrument_details(self, instrument_id: str, instrument_type: str = 'SWAP') -> Optional[List[float]]:
        """
        Get contract price, value, lot size, and min size for an instrument (SWAP by default).
        Uses simple caching to avoid excessive API calls.

        Args:
            instrument_id: Instrument ID (e.g., "BTC-USDT-SWAP")
            instrument_type: Instrument type (e.g., "SWAP", "SPOT")

        Returns:
            A list containing [contract_price, contract_value, lot_size, min_size] or None if an error occurs.
            
        Raises:
            ConnectionError: If the API request fails.
            OKXError: If the API returns an error.
            ValueError: If the response format is unexpected.
        """
        now = time.time()
        cache_key = f"{instrument_id}_{instrument_type}"
        cached_entry = self._instrument_details_cache.get(cache_key)

        # Check cache validity
        if cached_entry and (now - cached_entry['timestamp']) < self._CACHE_TTL_SECONDS:
            logger.debug(f"Using cached details for {instrument_id}")
            return cached_entry['details']

        logger.info(f"Fetching fresh instrument details for {instrument_id}")
        try:
            # Fetch current price
            ticker_response = self.make_request(
                method="GET",
                endpoint=f"{API_V5_PREFIX}/market/ticker",
                params={"instId": instrument_id}
            )
            
            if not ticker_response.get("data") or len(ticker_response["data"]) == 0:
                logger.warning(f"No ticker data found for instrument: {instrument_id}")
                return None
                
            contract_price = float(ticker_response["data"][0]["last"])

            # Fetch instrument size details
            instrument_response = self.get_instruments(instrument_type=instrument_type, instrument_id=instrument_id)
            
            if not instrument_response.get("data") or len(instrument_response["data"]) == 0:
                logger.warning(f"No instrument data found for: {instrument_id}")
                return None

            instrument_info = instrument_response["data"][0]
            contract_value = float(instrument_info.get("ctVal", 1.0))  # Default to 1.0 for SPOT or if not present
            lot_size = float(instrument_info.get("lotSz", 1.0))       # Default to 1.0
            min_size = float(instrument_info.get("minSz", 1.0))       # Default to 1.0

            details = [contract_price, contract_value, lot_size, min_size]

            # Update cache
            self._instrument_details_cache[cache_key] = {'timestamp': now, 'details': details}
            logger.debug(f"Cached instrument details for {instrument_id}: {details}")
            return details

        except (KeyError, IndexError, ValueError, TypeError) as e:
            logger.error(f"Error processing instrument details for {instrument_id}: {e}", exc_info=True)
            return None
        except (ConnectionError, OKXError) as e:
            logger.error(f"API error fetching instrument details for {instrument_id}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.exception(f"An unexpected error occurred while fetching instrument details for {instrument_id}: {e}")
            raise RuntimeError(f"An unexpected error occurred while fetching instrument details.") from e

    def calculate_contract_size(self, instrument_id: str, usdt_size: float) -> Optional[float]:
        """
        Calculate number of contracts for a given USDT position size (for SWAPs).
        Ensures the result is a multiple of the lot size and >= min size.

        Args:
            instrument_id: Instrument ID (e.g., "BTC-USDT-SWAP")
            usdt_size: Position size in USDT

        Returns:
            The contract size (float) or None if calculation fails.
            
        Raises:
            ConnectionError: If the API request fails.
            OKXError: If the API returns an error.
            ValueError: If parameters are invalid or calculations fail.
        """
        try:
            if usdt_size <= 0:
                raise ValueError("USDT position size must be greater than zero")
                
            details = self.get_instrument_details(instrument_id, instrument_type='SWAP')
            if not details:
                logger.warning(f"Unable to get instrument details for {instrument_id}")
                return None

            contract_price, contract_value, lot_size, min_size = details

            if contract_price <= 0 or contract_value <= 0 or lot_size <= 0:
                logger.warning(f"Invalid instrument details for calculation: {details}")
                return None

            # Calculate raw number of contracts
            raw_contracts = usdt_size / (contract_price * contract_value)

            # Calculate contracts, ensuring it's a multiple of lot_size
            epsilon = 1e-9  # Small buffer to avoid floating point issues
            contracts = math.floor((raw_contracts / lot_size) + epsilon) * lot_size

            # Ensure it's at least the minimum size
            contracts = max(contracts, min_size)

            # Ensure contracts is still a multiple of lot_size after applying min_size
            contracts = math.ceil((contracts / lot_size) - epsilon) * lot_size
            contracts = max(contracts, min_size)

            # Ensure the final size is representable without excessive decimal places for the API
            precision = max(0, -int(math.log10(lot_size))) if lot_size > 0 else 0
            contracts = round(contracts, precision)

            logger.info(f"Calculated {contracts} contracts for {usdt_size} USDT on {instrument_id}")
            return contracts

        except ValueError as ve:
            logger.error(f"Invalid parameters for contract size calculation: {ve}", exc_info=True)
            raise
        except (ConnectionError, OKXError) as e:
            logger.error(f"API error calculating contract size for {instrument_id}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.exception(f"An unexpected error occurred calculating contract size for {instrument_id}: {e}")
            raise RuntimeError(f"An unexpected error calculating contract size.") from e

    def validate_and_correct_order_size(self, instrument_id: str, size: str, instrument_type: str = 'SWAP') -> str:
        """
        Validates an order size against instrument requirements and corrects it if needed.
        Ensures the size is a multiple of the lot size and >= min size.

        Args:
            instrument_id: Instrument ID (e.g., "BTC-USDT-SWAP")
            size: The order size as a string
            instrument_type: Instrument type (default: 'SWAP')

        Returns:
            The validated and corrected size as a string
            
        Raises:
            ValueError: If the size is invalid or cannot be corrected
            ConnectionError: If the API request fails
            OKXError: If the API returns an error
        """
        try:
            size_float = float(size)
            if size_float <= 0:
                raise ValueError(f"Order size must be greater than zero, got: {size}")
                
            details = self.get_instrument_details(instrument_id, instrument_type)
            if not details:
                raise ValueError(f"Unable to get instrument details for {instrument_id}")
                
            _, _, lot_size, min_size = details
            
            # Ensure size meets minimum requirement
            if size_float < min_size:
                logger.warning(f"Order size {size} is below minimum size {min_size}, adjusting to minimum")
                size_float = min_size
            
            # Ensure size is a multiple of lot_size
            epsilon = 1e-9  # Small buffer to avoid floating point issues
            remainder = size_float % lot_size
            if remainder > epsilon and (lot_size - remainder) > epsilon:
                # Round to the nearest multiple of lot_size
                adjusted_size = round(size_float / lot_size) * lot_size
                
                # If rounding down would go below min_size, round up
                if adjusted_size < min_size:
                    adjusted_size = math.ceil(size_float / lot_size) * lot_size
                
                logger.warning(f"Order size {size} is not a multiple of lot size {lot_size}, adjusted to {adjusted_size}")
                size_float = adjusted_size
            
            # Ensure precision matches lot_size
            precision = max(0, -int(math.log10(lot_size))) if lot_size > 0 else 0
            size_float = round(size_float, precision)
            
            # Convert back to string with appropriate precision
            result = f"{size_float:.{precision}f}".rstrip('0').rstrip('.') if '.' in f"{size_float:.{precision}f}" else f"{size_float:.{precision}f}"
            logger.info(f"Validated and corrected order size: {size} → {result} for {instrument_id}")
            return result
            
        except ValueError as ve:
            if "is not a multiple of lot size" in str(ve) or "is below the minimum allowed size" in str(ve):
                raise  # Re-raise specific error
            logger.error(f"Error validating order size: {ve}")
            raise ValueError(f"Invalid order size format: {size}. Must be a valid number.") from ve
        except (ConnectionError, OKXError) as e:
            logger.error(f"API error validating order size: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error validating order size: {e}")
            raise ValueError(f"Unexpected error validating order size: {str(e)}") from e

# --- Initialize Client ---
try:
    okx_client = OKXClient()
except Exception as e:
    logger.critical(f"Failed to initialize OKXClient: {e}", exc_info=True)
    # Depending on the desired behavior, you might exit or prevent server start
    raise SystemExit("Critical error during OKXClient initialization.") from e


# --- MCP Tools ---

@mcp.tool()
def get_price(instrument: str) -> Dict[str, Any]:
    """
    Get the latest market ticker price for a specific OKX instrument.

    Args:
        instrument: The instrument ID (e.g., BTC-USDT, ETH-USDT-SWAP).

    Returns:
        A dictionary containing ticker information (price, bid, ask, volume, etc.).

    Raises:
        ConnectionError: If the API request fails.
        OKXError: If the API returns an error.
        ValueError: If the response format is unexpected or data is missing.
    """
    logger.info(f"Fetching price for instrument: {instrument}")
    endpoint = f"{API_V5_PREFIX}/market/ticker"
    try:
        response_data = okx_client.make_request(
            method="GET",
            endpoint=endpoint,
            params={"instId": instrument}
        )

        if not response_data.get("data") or len(response_data["data"]) == 0:
            logger.warning(f"No ticker data found for instrument: {instrument}")
            raise ValueError(f"No data returned from OKX API for instrument {instrument}")

        ticker = response_data["data"][0]
        # Convert timestamp from ms string to ISO format
        ts_ms = int(ticker.get("ts", 0))
        timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(ts_ms / 1000))

        result = {
            "instrument": ticker.get("instId"),
            "lastPrice": ticker.get("last"),
            "bid": ticker.get("bidPx"),
            "ask": ticker.get("askPx"),
            "high24h": ticker.get("high24h"),
            "low24h": ticker.get("low24h"),
            "volume24h": ticker.get("vol24h"), # Base currency volume
            "volumeCurrency24h": ticker.get("volCcy24h"), # Quote currency volume
            "timestamp": timestamp_iso
        }
        logger.info(f"Successfully fetched price for {instrument}: Last={result['lastPrice']}")
        return result

    except (ConnectionError, OKXError, ValueError) as e:
        logger.error(f"Error fetching price for {instrument}: {e}", exc_info=True)
        raise # Re-raise the caught exception
    except Exception as e:
        logger.exception(f"An unexpected error occurred while fetching price for {instrument}: {e}")
        raise RuntimeError(f"An unexpected error occurred while fetching price for {instrument}.") from e

@mcp.tool()
def get_candlesticks(instrument: str, bar: str = "1m", limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get candlestick (k-line) data for an OKX instrument.

    Args:
        instrument: The instrument ID (e.g., BTC-USDT).
        bar: Candlestick interval (e.g., 1m, 5m, 1H, 1D). Default is "1m".
        limit: Number of candlesticks to retrieve (max typically 100-300 depending on API). Default is 100.

    Returns:
        A list of dictionaries, each representing a candlestick. Returns empty list if no data.

    Raises:
        ConnectionError: If the API request fails.
        OKXError: If the API returns an error.
        ValueError: If the response format is unexpected.
    """
    logger.info(f"Fetching candlesticks for instrument: {instrument}, bar: {bar}, limit: {limit}")
    endpoint = f"{API_V5_PREFIX}/market/candles"
    params = {
        "instId": instrument,
        "bar": bar,
        "limit": limit
    }
    try:
        response_data = okx_client.make_request(
            method="GET",
            endpoint=endpoint,
            params=params
        )

        candles_data = response_data.get("data", [])
        if not candles_data:
            logger.warning(f"No candlestick data found for {instrument} with bar={bar}, limit={limit}.")
            return [] # Return empty list is valid if no data exists

        result = []
        for candle in candles_data:
            # Expected format: [ts, o, h, l, c, vol, volCcy, ...]
            if len(candle) >= 7:
                ts_ms = int(candle[0])
                timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(ts_ms / 1000))
                result.append({
                    "timestamp": timestamp_iso,
                    "open": candle[1],
                    "high": candle[2],
                    "low": candle[3],
                    "close": candle[4],
                    "volume": candle[5],          # Base currency volume
                    "volumeCurrency": candle[6]   # Quote currency volume
                })
            else:
                 logger.warning(f"Skipping malformed candle data: {candle}")

        logger.info(f"Successfully fetched {len(result)} candlesticks for {instrument}.")
        return result

    except (ConnectionError, OKXError, ValueError) as e:
        logger.error(f"Error fetching candlesticks for {instrument}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.exception(f"An unexpected error occurred while fetching candlesticks for {instrument}: {e}")
        raise RuntimeError(f"An unexpected error occurred while fetching candlesticks for {instrument}.") from e

@mcp.tool()
def get_account_balance() -> Dict[str, Any]:
    """
    Get account balance information (requires authentication).

    Returns:
        A dictionary containing total equity and detailed balances per currency.

    Raises:
        ConnectionError: If the API request fails.
        PermissionError: If authentication fails or credentials are missing.
        OKXError: If the API returns an error.
        ValueError: If the response format is unexpected or data is missing.
    """
    logger.info("Fetching account balance...")
    endpoint = f"{API_V5_PREFIX}/account/balance"
    try:
        response_data = okx_client.make_request(
            method="GET",
            endpoint=endpoint,
            auth=True
        )

        if not response_data.get("data") or len(response_data["data"]) == 0:
            logger.error("No data returned from OKX API for account balance.")
            raise ValueError("No data returned from OKX API for account balance")

        balance_data = response_data["data"][0]
        details = balance_data.get("details", []) # Handle case where 'details' might be missing

        result = {
            "totalEquity": balance_data.get("totalEq"),
            "unrealizedPnL": balance_data.get("adjEq"), # Adjusted/Total Equity in USD
            "details": [
                {
                    "currency": detail.get("ccy"),
                    "equity": detail.get("eq"),
                    "available": detail.get("availEq"),
                    "frozen": detail.get("frozenBal"),
                    "orderFrozen": detail.get("ordFrozen"),
                    "unrealizedPnL": detail.get("upl"),
                    "liability": detail.get("liab"),
                    "interest": detail.get("interest")
                }
                for detail in details
            ]
        }
        logger.info(f"Successfully fetched account balance. Total Equity: {result['totalEquity']}")
        return result

    except PermissionError as pe:
         logger.error(f"Authentication failed while fetching account balance: {pe}")
         raise PermissionError("Authentication failed. Check OKX API credentials.") from pe
    except (ConnectionError, OKXError, ValueError) as e:
        logger.error(f"Error fetching account balance: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.exception("An unexpected error occurred while fetching account balance.")
        raise RuntimeError("An unexpected error occurred while fetching account balance.") from e

@mcp.tool()
def get_positions(instrument_type: str = "SWAP", instrument_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get current open positions (requires authentication).

    Args:
        instrument_type: Instrument type (e.g., MARGIN, SWAP, FUTURES, OPTION). Default is SWAP.
        instrument_id: Specific instrument ID to filter by (e.g., BTC-USDT-SWAP). Optional.

    Returns:
        A list of dictionaries, each representing an open position. Returns empty list if no positions.

    Raises:
        ConnectionError: If the API request fails.
        PermissionError: If authentication fails or credentials are missing.
        OKXError: If the API returns an error.
        ValueError: If the response format is unexpected.
    """
    log_msg = f"Fetching positions for type: {instrument_type}"
    if instrument_id:
        log_msg += f", instrument: {instrument_id}"
    logger.info(log_msg)

    endpoint = f"{API_V5_PREFIX}/account/positions"
    params = {"instType": instrument_type}
    if instrument_id:
        params["instId"] = instrument_id

    try:
        response_data = okx_client.make_request(
            method="GET",
            endpoint=endpoint,
            params=params,
            auth=True
        )

        positions_data = response_data.get("data", [])
        if not positions_data:
            logger.info(f"No open positions found for the specified criteria.")
            return []

        result = []
        for pos in positions_data:
            c_time_ms = int(pos.get("cTime", 0))
            u_time_ms = int(pos.get("uTime", 0))
            created_time_iso = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(c_time_ms / 1000)) if c_time_ms else None
            updated_time_iso = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(u_time_ms / 1000)) if u_time_ms else None

            result.append({
                "instrument": pos.get("instId"),
                "type": pos.get("instType"),
                "marginMode": pos.get("mgnMode"), # cross, isolated
                "position": pos.get("pos"), # Position size
                "positionSide": pos.get("posSide"), # long, short, net
                "averagePrice": pos.get("avgPx"),
                "unrealizedPnL": pos.get("upl"),
                "leverage": pos.get("lever"),
                "liquidationPrice": pos.get("liqPx"),
                "margin": pos.get("margin"), # Position margin
                "notionalUsd": pos.get("notionalUsd"), # Notional value in USD
                "createdTime": created_time_iso,
                "updatedTime": updated_time_iso,
            })

        logger.info(f"Successfully fetched {len(result)} positions.")
        return result

    except PermissionError as pe:
         logger.error(f"Authentication failed while fetching positions: {pe}")
         raise PermissionError("Authentication failed. Check OKX API credentials.") from pe
    except (ConnectionError, OKXError, ValueError) as e:
        logger.error(f"Error fetching positions: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.exception("An unexpected error occurred while fetching positions.")
        raise RuntimeError("An unexpected error occurred while fetching positions.") from e

@mcp.tool()
def get_trade_history(instrument_type: Optional[str] = None, instrument_id: Optional[str] = None, order_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get recent trade (fill) history (requires authentication).

    Args:
        instrument_type: Filter by instrument type (e.g., SPOT, SWAP). Optional.
        instrument_id: Filter by instrument ID (e.g., BTC-USDT). Optional.
        order_id: Filter by order ID. Optional.
        limit: Number of trades to retrieve (max 100). Default is 100.

    Returns:
        A list of dictionaries, each representing a trade fill. Returns empty list if no history found.

    Raises:
        ConnectionError: If the API request fails.
        PermissionError: If authentication fails or credentials are missing.
        OKXError: If the API returns an error.
        ValueError: If the response format is unexpected.
    """
    log_msg = f"Fetching trade history, limit: {limit}"
    params = {"limit": limit}
    if instrument_type:
        params["instType"] = instrument_type
        log_msg += f", type: {instrument_type}"
    if instrument_id:
        params["instId"] = instrument_id
        log_msg += f", instrument: {instrument_id}"
    if order_id:
        params["ordId"] = order_id
        log_msg += f", orderId: {order_id}"
    logger.info(log_msg)

    endpoint = f"{API_V5_PREFIX}/trade/fills"

    try:
        response_data = okx_client.make_request(
            method="GET",
            endpoint=endpoint,
            params=params,
            auth=True
        )

        trades_data = response_data.get("data", [])
        if not trades_data:
            logger.info("No trade history found for the specified criteria.")
            return []

        result = []
        for trade in trades_data:
            ts_ms = int(trade.get("ts", 0))
            timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(ts_ms / 1000)) if ts_ms else None

            result.append({
                "instrument": trade.get("instId"),
                "tradeId": trade.get("tradeId"),
                "orderId": trade.get("ordId"),
                "price": trade.get("fillPx"), # Fill price
                "size": trade.get("fillSz"),   # Fill size
                "side": trade.get("side"),     # buy, sell
                "positionSide": trade.get("posSide"), # long, short, net (for derivatives)
                "executionType": trade.get("execType"), # T (Taker), M (Maker)
                "feeCurrency": trade.get("feeCcy"),
                "fee": trade.get("fee"),
                "timestamp": timestamp_iso
            })

        logger.info(f"Successfully fetched {len(result)} trade history entries.")
        return result

    except PermissionError as pe:
         logger.error(f"Authentication failed while fetching trade history: {pe}")
         raise PermissionError("Authentication failed. Check OKX API credentials.") from pe
    except (ConnectionError, OKXError, ValueError) as e:
        logger.error(f"Error fetching trade history: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.exception("An unexpected error occurred while fetching trade history.")
        raise RuntimeError("An unexpected error occurred while fetching trade history.") from e

def _place_order_internal(
    instrument: str,
    trade_mode: str,
    side: str,
    order_type: str,
    size: str,
    order_price: Optional[str] = None,
    client_order_id: Optional[str] = None,
    tag: Optional[str] = None,
    is_swap: bool = False
) -> Dict[str, Any]:
    """
    Internal helper function to place orders. Not exposed as a tool.
    Handles common order placement logic for all order types.
    """
    logger.info(f"Placing order for instrument: {instrument}, side: {side}, type: {order_type}")
    
    # Validate instrument and parameters for SWAP instruments
    if is_swap:
        try:
            # Get instrument details to validate parameters
            instrument_type = "SWAP"
            details = okx_client.get_instrument_details(instrument, instrument_type)
            
            if not details:
                error_msg = f"Unable to find instrument details for {instrument}. Please verify the instrument ID exists and is active."
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            _, _, lot_size, min_size = details
            
            # Validate size is a multiple of lot_size and >= min_size
            try:
                size_float = float(size)
                if size_float < min_size:
                    error_msg = f"Order size {size} is below the minimum allowed size {min_size} for {instrument}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Check if size is a multiple of lot_size (with small epsilon for floating point precision)
                epsilon = 1e-9
                remainder = size_float % lot_size
                if remainder > epsilon and (lot_size - remainder) > epsilon:
                    error_msg = f"Order size {size} is not a multiple of lot size {lot_size} for {instrument}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Validate price if provided and order type requires it
                if order_price is not None and order_type in ['limit', 'post_only', 'fok', 'ioc']:
                    # Get tick_size from instrument info
                    instrument_response = okx_client.get_instruments(instrument_type=instrument_type, instrument_id=instrument)
                    if not instrument_response.get("data") or len(instrument_response["data"]) == 0:
                        logger.warning(f"No instrument data found for: {instrument}")
                        raise ValueError(f"Could not find instrument data for {instrument}")
                    
                    instrument_info = instrument_response["data"][0]
                    tick_size = float(instrument_info.get("tickSz", "0.01"))  # Default to 0.01 if not found
                    
                    try:
                        price_float = float(order_price)
                        # Check if price is a multiple of tick_size
                        price_remainder = price_float % tick_size
                        if price_remainder > epsilon and (tick_size - price_remainder) > epsilon:
                            error_msg = f"Order price {order_price} is not a multiple of tick size {tick_size} for {instrument}"
                            logger.error(error_msg)
                            raise ValueError(error_msg)
                    except ValueError as e:
                        if "not a multiple of tick size" in str(e):
                            raise  # Re-raise our specific error
                        error_msg = f"Invalid order price format: {order_price}. Must be a valid number."
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                
            except ValueError as e:
                if any(x in str(e) for x in ["is not a multiple of lot size", "is below the minimum allowed size", "not a multiple of tick size"]):
                    raise  # Re-raise our specific error
                error_msg = f"Invalid order size format: {size}. Must be a valid number."
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            logger.debug(f"Order parameters validated for {instrument}")
        except (ConnectionError, OKXError, ValueError) as e:
            if isinstance(e, ValueError) and ("multiple of lot size" in str(e) or "minimum allowed size" in str(e)):
                raise  # Re-raise our specific error
            error_msg = f"Error validating instrument {instrument} or parameters: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    # Generate client_order_id if not provided
    if client_order_id is None:
        # Extract the instrument base (e.g., BTC from BTC-USDT-SWAP)
        instrument_base = instrument.split('-')[0]
        # Generate a client order ID with instrument prefix, timestamp, and uuid
        ts = int(time.time())
        random_uuid = uuid.uuid4().hex[:10]
        client_order_id = f"{instrument_base}{ts}{random_uuid}"
        client_order_id = client_order_id[:32]  # Ensure under 32 characters
        logger.debug(f"Generated client order ID: {client_order_id}")
    
    # Prepare API request data
    data = {
        "instId": instrument,
        "tdMode": trade_mode,
        "side": side,
        "ordType": order_type,
        "sz": size
    }
    
    # Add SWAP-specific parameters
    if is_swap:
        position_side = "long" if side.lower() == "buy" else "short"
        data["posSide"] = position_side
        logger.debug(f"Added position side for SWAP instrument: {position_side}")
    
    # Add optional parameters if provided
    if order_price is not None:
        data["px"] = order_price
    if client_order_id is not None:
        data["clOrdId"] = client_order_id
    if tag is not None:
        data["tag"] = tag
    
    endpoint = f"{API_V5_PREFIX}/trade/order"
    
    try:
        response_data = okx_client.make_request(
            method="POST",
            endpoint=endpoint,
            data=data,
            auth=True
        )
        
        if not response_data.get("data") or len(response_data["data"]) == 0:
            logger.error("No data returned from OKX API for placing order.")
            raise ValueError("No data returned from OKX API for placing order")
        
        order_data = response_data["data"][0]
        # Check for specific error codes in the response that indicate order issues
        status_code = order_data.get("sCode")
        status_msg = order_data.get("sMsg", "Unknown error")
        
        if status_code and status_code != "0":
            error_msg = f"Order placement failed: {status_msg} (Code: {status_code})"
            logger.error(error_msg)
            # Map common error codes to more descriptive messages
            if status_code == "50001":
                error_msg = f"Insufficient balance to place order for {instrument} of size {size}"
            elif status_code == "51008":
                error_msg = f"Order size {size} is below the minimum allowed size for {instrument}"
            elif status_code == "51010": 
                error_msg = f"Order price for {instrument} is outside the allowed price range"
            elif status_code == "51004":
                error_msg = f"Order size {size} for {instrument} is too large for your account"
            elif status_code == "51015" or status_code == "51016":
                error_msg = f"Order placed too frequently for {instrument}, rate limit exceeded"
            elif status_code in ["50011", "50012"]:
                error_msg = f"Leverage setting issue for {instrument}, check account leverage settings"
            elif status_code == "50006":
                error_msg = f"Account in liquidation risk for {instrument}, cannot place new orders"
            
            # Add request details to the error message
            error_with_details = f"{error_msg}\nRequest details:\n"
            error_with_details += f"  Instrument: {instrument}\n"
            error_with_details += f"  Order Type: {order_type}\n"
            error_with_details += f"  Side: {side}\n"
            error_with_details += f"  Size: {size}\n"
            if order_price:
                error_with_details += f"  Price: {order_price}\n"
            error_with_details += f"  Trade Mode: {trade_mode}\n"
            if is_swap:
                error_with_details += f"  Position Side: {position_side}\n"
            
            # Log the full request data
            logger.error(f"Order request data: {json.dumps(data)}")
            
            raise OKXError(message=error_with_details, code=status_code, status_code=200)
        
        result = {
            "orderId": order_data.get("ordId"),
            "clientOrderId": order_data.get("clOrdId"),
            "tag": order_data.get("tag"),
            "timestamp": order_data.get("ts"),
            "statusCode": status_code,
            "statusMessage": status_msg
        }
        
        logger.info(f"Successfully placed order. Order ID: {result['orderId']}, Client Order ID: {result['clientOrderId']}")
        return result
    
    except PermissionError as pe:
        logger.error(f"Authentication failed while placing order: {pe}")
        raise PermissionError("Authentication failed. Check OKX API credentials.") from pe
    except ConnectionError as ce:
        error_msg = f"Network error while placing {order_type} {side} order for {instrument}: {str(ce)}"
        logger.error(error_msg)
        raise ConnectionError(error_msg) from ce
    except OKXError as oe:
        # Enhance OKX specific errors with more context
        context_msg = f"Failed to place {order_type} {side} order of size {size} for {instrument}"
        enhanced_msg = f"{context_msg}: {str(oe)}"
        logger.error(enhanced_msg)
        # Don't modify the original error message as it may already contain the request details
        raise oe
    except ValueError as ve:
        error_msg = f"Invalid parameter when placing {order_type} {side} order for {instrument}: {str(ve)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from ve
    except Exception as e:
        error_msg = f"Unexpected error placing {order_type} {side} order for {instrument} of size {size}: {str(e)}"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from e

def _convert_usdt_to_contracts(instrument: str, usdt_size_str: str) -> str:
    """
    Helper function to convert USDT size to contract size for SWAP instruments.
    
    Args:
        instrument: SWAP instrument ID (e.g., "BTC-USDT-SWAP")
        usdt_size_str: Size in USDT as a string
        
    Returns:
        Contract size as a string with appropriate precision
        
    Raises:
        ValueError: If conversion fails or parameters are invalid
    """
    logger.info(f"Converting USDT size {usdt_size_str} to contract size for {instrument}")
    try:
        usdt_size = float(usdt_size_str)
        if usdt_size <= 0:
            raise ValueError(f"USDT size must be greater than zero, got: {usdt_size_str}")
        
        # Use calculate_contract_size to get the appropriate contract size
        contract_size = okx_client.calculate_contract_size(instrument, usdt_size)
        if contract_size is None:
            raise ValueError(f"Failed to calculate contract size for {instrument} with {usdt_size} USDT")
        
        # Convert to string with appropriate precision
        details = okx_client.get_instrument_details(instrument, 'SWAP')
        if not details:
            raise ValueError(f"Unable to get instrument details for {instrument}")
        
        _, _, lot_size, _ = details
        # Determine precision based on lot_size
        precision = max(0, -int(math.log10(lot_size))) if lot_size > 0 else 0
        contract_size_str = f"{contract_size:.{precision}f}".rstrip('0').rstrip('.') if '.' in f"{contract_size:.{precision}f}" else f"{contract_size:.{precision}f}"
        
        logger.info(f"Converted {usdt_size} USDT to {contract_size_str} contracts for {instrument}")
        return contract_size_str
    except ValueError as ve:
        if "Failed to calculate" in str(ve) or "USDT size must be" in str(ve) or "Unable to get instrument" in str(ve):
            raise  # Re-raise our specific error
        raise ValueError(f"Invalid USDT size format: {usdt_size_str}. Must be a valid number.") from ve

@mcp.tool()
def place_swap_limit_order(
    instrument: str,
    side: str,
    size: str,
    price: str,
    client_order_id: Optional[str] = None,
    tag: Optional[str] = None
) -> Dict[str, Any]:
    """
    Place a limit order for SWAP trading on OKX (requires authentication).
    Use this tool specifically for placing limit orders on perpetual swap contracts.
    
    Args:
        instrument: SWAP instrument ID (e.g., "BTC-USDT-SWAP", "ETH-USDT-SWAP")
        side: Order side ("buy", "sell")
        size: Size in USDT to trade (will be converted to contract quantity)
        price: Limit price for the order
        client_order_id: Optional client-specified order ID
        tag: Optional order tag for tracking
        
    Returns:
        Dictionary containing order details including orderId and clientOrderId
        
    Example:
        >>> place_swap_limit_order(
        ...     instrument="BTC-USDT-SWAP",
        ...     side="buy",
        ...     size="100",  # 100 USDT
        ...     price="65000"
        ... )
        {'orderId': '1234567890', 'clientOrderId': 'okx_1234567890', ...}
        
    Raises:
        ValueError: If instrument is not a SWAP product or parameters are invalid
        PermissionError: If authentication fails
        ConnectionError: If API request fails
        OKXError: If OKX API returns an error
    """
    if not instrument.endswith('-SWAP'):
        raise ValueError(f"Invalid SWAP instrument: {instrument}. Must end with '-SWAP'")
    
    try:
        # Convert USDT size to contract size
        contract_size_str = _convert_usdt_to_contracts(instrument, size)
        
        # Validate price is a multiple of tickSz
        try:
            # Get instrument tick size
            instrument_response = okx_client.get_instruments(instrument_type='SWAP', instrument_id=instrument)
            if not instrument_response.get("data") or len(instrument_response["data"]) == 0:
                raise ValueError(f"Could not find instrument data for {instrument}")
            
            instrument_info = instrument_response["data"][0]
            tick_size = float(instrument_info.get("tickSz", "0.01"))  # Default to 0.01 if not found
            
            # Validate and adjust price
            price_float = float(price)
            epsilon = 1e-9  # Small buffer to avoid floating point issues
            price_remainder = price_float % tick_size
            
            if price_remainder > epsilon and (tick_size - price_remainder) > epsilon:
                # Round to the nearest valid price (multiple of tick_size)
                adjusted_price = round(price_float / tick_size) * tick_size
                # Format with appropriate precision
                precision = max(0, -int(math.log10(tick_size))) if tick_size > 0 else 0
                adjusted_price = round(adjusted_price, precision)
                price = f"{adjusted_price:.{precision}f}".rstrip('0').rstrip('.') if '.' in f"{adjusted_price:.{precision}f}" else f"{adjusted_price:.{precision}f}"
                logger.warning(f"Order price adjusted from {price_float} to {price} to meet tick size requirements")
        except (ValueError, KeyError, TypeError) as e:
            if "Could not find instrument data" in str(e):
                raise  # Re-raise our specific error
            logger.warning(f"Error validating price, proceeding with original price: {e}")
            # Continue with original price and let the API validate it
        
        return _place_order_internal(
            instrument=instrument,
            trade_mode="cross",
            side=side,
            order_type="limit",
            size=contract_size_str,  # Use the converted contract size
            order_price=price,
            client_order_id=client_order_id,
            tag=tag,
            is_swap=True
        )
    except (ValueError, ConnectionError, OKXError) as e:
        error_msg = f"Error preparing SWAP limit order for {instrument}: {str(e)}"
        logger.error(error_msg)
        # Enhance error message based on common issues
        if "lot size" in str(e).lower():
            error_msg += f" (Order size must be a multiple of the instrument's lot size)"
        elif "minimum" in str(e).lower():
            error_msg += f" (Order size must meet the minimum size requirement)"
        elif "tick size" in str(e).lower():
            error_msg += f" (Order price must be a multiple of the instrument's tick size)"
        raise type(e)(error_msg)

@mcp.tool()
def place_swap_market_order(
    instrument: str,
    side: str,
    size: str,
    client_order_id: Optional[str] = None,
    tag: Optional[str] = None
) -> Dict[str, Any]:
    """
    Place a market order for SWAP trading on OKX (requires authentication).
    Use this tool specifically for placing market orders on perpetual swap contracts.
    Market orders are executed immediately at the best available price.
    
    Args:
        instrument: SWAP instrument ID (e.g., "BTC-USDT-SWAP", "ETH-USDT-SWAP")
        side: Order side ("buy", "sell")
        size: Size in USDT to trade (will be converted to contract quantity)
        client_order_id: Optional client-specified order ID
        tag: Optional order tag for tracking
        
    Returns:
        Dictionary containing order details including orderId and clientOrderId
        
    Example:
        >>> place_swap_market_order(
        ...     instrument="BTC-USDT-SWAP",
        ...     side="buy",
        ...     size="100"  # 100 USDT
        ... )
        {'orderId': '1234567890', 'clientOrderId': 'okx_1234567890', ...}
        
    Raises:
        ValueError: If instrument is not a SWAP product or parameters are invalid
        PermissionError: If authentication fails
        ConnectionError: If API request fails
        OKXError: If OKX API returns an error
    """
    if not instrument.endswith('-SWAP'):
        raise ValueError(f"Invalid SWAP instrument: {instrument}. Must end with '-SWAP'")
    
    try:
        # Convert USDT size to contract size
        contract_size_str = _convert_usdt_to_contracts(instrument, size)
            
        return _place_order_internal(
            instrument=instrument,
            trade_mode="cross",
            side=side,
            order_type="market",
            size=contract_size_str,  # Use the converted contract size
            client_order_id=client_order_id,
            tag=tag,
            is_swap=True
        )
    except (ValueError, ConnectionError, OKXError) as e:
        error_msg = f"Error preparing SWAP market order for {instrument}: {str(e)}"
        logger.error(error_msg)
        # Enhance error message based on common issues
        if "lot size" in str(e).lower():
            error_msg += f" (Order size must be a multiple of the instrument's lot size)"
        elif "minimum" in str(e).lower():
            error_msg += f" (Order size must meet the minimum size requirement)"
        raise type(e)(error_msg)

@mcp.tool()
def place_spot_limit_order(
    instrument: str,
    trade_mode: str,
    side: str,
    size: str,
    price: str,
    client_order_id: Optional[str] = None,
    tag: Optional[str] = None
) -> Dict[str, Any]:
    """
    Place a limit order for SPOT trading on OKX (requires authentication).
    Use this tool specifically for placing limit orders on spot trading pairs.
    
    Args:
        instrument: SPOT instrument ID (e.g., "BTC-USDT", "ETH-USDT")
        trade_mode: Trade mode ("cash", "cross")
        side: Order side ("buy", "sell")
        size: Amount of base currency to trade
        price: Limit price for the order
        client_order_id: Optional client-specified order ID
        tag: Optional order tag for tracking
        
    Returns:
        Dictionary containing order details including orderId and clientOrderId
        
    Example:
        >>> place_spot_limit_order(
        ...     instrument="BTC-USDT",
        ...     trade_mode="cash",
        ...     side="buy",
        ...     size="0.1",
        ...     price="65000"
        ... )
        {'orderId': '1234567890', 'clientOrderId': 'okx_1234567890', ...}
        
    Raises:
        ValueError: If instrument contains '-SWAP' or parameters are invalid
        PermissionError: If authentication fails
        ConnectionError: If API request fails
        OKXError: If OKX API returns an error
    """
    if '-SWAP' in instrument:
        raise ValueError(f"Invalid SPOT instrument: {instrument}. Must not contain '-SWAP'")
    
    try:
        # Validate and correct order size based on instrument requirements
        logger.info(f"Validating order size for {instrument}: {size}")
        corrected_size = okx_client.validate_and_correct_order_size(instrument, size, 'SPOT')
        if corrected_size != size:
            logger.warning(f"Order size was adjusted from {size} to {corrected_size} to meet instrument requirements")
            size = corrected_size
        
        # Validate price is a multiple of tickSz
        try:
            # Get instrument tick size
            instrument_response = okx_client.get_instruments(instrument_type='SPOT', instrument_id=instrument)
            if not instrument_response.get("data") or len(instrument_response["data"]) == 0:
                raise ValueError(f"Could not find instrument data for {instrument}")
            
            instrument_info = instrument_response["data"][0]
            tick_size = float(instrument_info.get("tickSz", "0.01"))  # Default to 0.01 if not found
            
            # Validate and adjust price
            price_float = float(price)
            epsilon = 1e-9  # Small buffer to avoid floating point issues
            price_remainder = price_float % tick_size
            
            if price_remainder > epsilon and (tick_size - price_remainder) > epsilon:
                # Round to the nearest valid price (multiple of tick_size)
                adjusted_price = round(price_float / tick_size) * tick_size
                # Format with appropriate precision
                precision = max(0, -int(math.log10(tick_size))) if tick_size > 0 else 0
                adjusted_price = round(adjusted_price, precision)
                price = f"{adjusted_price:.{precision}f}".rstrip('0').rstrip('.') if '.' in f"{adjusted_price:.{precision}f}" else f"{adjusted_price:.{precision}f}"
                logger.warning(f"Order price adjusted from {price_float} to {price} to meet tick size requirements")
        except (ValueError, KeyError, TypeError) as e:
            if "Could not find instrument data" in str(e):
                raise  # Re-raise our specific error
            logger.warning(f"Error validating price, proceeding with original price: {e}")
            # Continue with original price and let the API validate it
        
        return _place_order_internal(
            instrument=instrument,
            trade_mode=trade_mode,
            side=side,
            order_type="limit",
            size=size,
            order_price=price,
            client_order_id=client_order_id,
            tag=tag,
            is_swap=False
        )
    except (ValueError, ConnectionError, OKXError) as e:
        error_msg = f"Error preparing SPOT limit order for {instrument}: {str(e)}"
        logger.error(error_msg)
        # Enhance error message based on common issues
        if "lot size" in str(e).lower():
            error_msg += f" (Order size must be a multiple of the instrument's lot size)"
        elif "minimum" in str(e).lower():
            error_msg += f" (Order size must meet the minimum size requirement)"
        elif "tick size" in str(e).lower():
            error_msg += f" (Order price must be a multiple of the instrument's tick size)"
        raise type(e)(error_msg)

@mcp.tool()
def place_spot_market_order(
    instrument: str,
    trade_mode: str,
    side: str,
    size: str,
    client_order_id: Optional[str] = None,
    tag: Optional[str] = None
) -> Dict[str, Any]:
    """
    Place a market order for SPOT trading on OKX (requires authentication).
    Use this tool specifically for placing market orders on spot trading pairs.
    Market orders are executed immediately at the best available price.
    
    Args:
        instrument: SPOT instrument ID (e.g., "BTC-USDT", "ETH-USDT")
        trade_mode: Trade mode ("cash", "cross")
        side: Order side ("buy", "sell")
        size: Amount of base currency to trade
        client_order_id: Optional client-specified order ID
        tag: Optional order tag for tracking
        
    Returns:
        Dictionary containing order details including orderId and clientOrderId
        
    Example:
        >>> place_spot_market_order(
        ...     instrument="BTC-USDT",
        ...     trade_mode="cash",
        ...     side="buy",
        ...     size="0.1"
        ... )
        {'orderId': '1234567890', 'clientOrderId': 'okx_1234567890', ...}
        
    Raises:
        ValueError: If instrument contains '-SWAP' or parameters are invalid
        PermissionError: If authentication fails
        ConnectionError: If API request fails
        OKXError: If OKX API returns an error
    """
    if '-SWAP' in instrument:
        raise ValueError(f"Invalid SPOT instrument: {instrument}. Must not contain '-SWAP'")
    
    try:
        # Validate and correct order size based on instrument requirements
        logger.info(f"Validating order size for {instrument}: {size}")
        corrected_size = okx_client.validate_and_correct_order_size(instrument, size, 'SPOT')
        if corrected_size != size:
            logger.warning(f"Order size was adjusted from {size} to {corrected_size} to meet instrument requirements")
            size = corrected_size
        
        return _place_order_internal(
            instrument=instrument,
            trade_mode=trade_mode,
            side=side,
            order_type="market",
            size=size,
            client_order_id=client_order_id,
            tag=tag,
            is_swap=False
        )
    except (ValueError, ConnectionError, OKXError) as e:
        error_msg = f"Error preparing SPOT market order for {instrument}: {str(e)}"
        logger.error(error_msg)
        # Enhance error message based on common issues
        if "lot size" in str(e).lower():
            error_msg += f" (Order size must be a multiple of the instrument's lot size)"
        elif "minimum" in str(e).lower():
            error_msg += f" (Order size must meet the minimum size requirement)"
        raise type(e)(error_msg)

@mcp.tool()
def calculate_position_size(instrument: str, usdt_size: float) -> Dict[str, Any]:
    """
    Calculate the number of contracts needed for a given USDT position size on a SWAP instrument.

    Args:
        instrument: The instrument ID (e.g., BTC-USDT-SWAP).
        usdt_size: Position size in USDT.

    Returns:
        A dictionary containing contract size and related instrument details.

    Raises:
        ConnectionError: If the API request fails.
        OKXError: If the API returns an error.
        ValueError: If the parameters are invalid or instrument details cannot be fetched.
    """
    logger.info(f"Calculating position size for {usdt_size} USDT on instrument: {instrument}")
    
    try:
        # Ensure instrument is a SWAP product
        if not instrument.endswith('-SWAP'):
            error_msg = f"Instrument must be a SWAP product (e.g., BTC-USDT-SWAP), got: {instrument}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if usdt_size <= 0:
            error_msg = "USDT position size must be greater than zero"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Get contract size
        contract_size = okx_client.calculate_contract_size(instrument, usdt_size)
        if contract_size is None:
            error_msg = f"Failed to calculate contract size for {instrument} with {usdt_size} USDT"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Get instrument details for the response
        details = okx_client.get_instrument_details(instrument)
        if not details:
            error_msg = f"Failed to get instrument details for {instrument}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        contract_price, contract_value, lot_size, min_size = details
        
        result = {
            "instrument": instrument,
            "usdtSize": usdt_size,
            "contractSize": contract_size,
            "notionalValue": contract_size * contract_price * contract_value,
            "currentPrice": contract_price,
            "contractValue": contract_value,
            "lotSize": lot_size,
            "minSize": min_size
        }
        
        logger.info(f"Successfully calculated position size: {contract_size} contracts for {usdt_size} USDT on {instrument}")
        return result
        
    except (ConnectionError, OKXError, ValueError) as e:
        logger.error(f"Error calculating position size: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.exception(f"An unexpected error occurred while calculating position size: {e}")
        raise RuntimeError(f"An unexpected error occurred while calculating position size.") from e

# --- Main Execution ---
if __name__ == "__main__":
    # You can adjust the root logger level here if needed, e.g., for more debug info
    # logging.getLogger().setLevel(logging.DEBUG)
    # logger.info("Starting FastMCP server...")
    try:
        mcp.run()
    except Exception as e:
        logger.critical(f"FastMCP server failed to run: {e}", exc_info=True)
        raise SystemExit("FastMCP server encountered a critical error.") from e