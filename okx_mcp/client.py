# okx_mcp/client.py
import os
import time
import hmac
import base64
import requests # Ensure requests is imported
import logging
import json # Ensure json is imported here or globally
from typing import Optional, Dict, Any, List
import hashlib # Added for signature generation

# --- Constants ---
OKX_BASE_URL = "https://www.okx.com"
API_V5_PREFIX = "/api/v5"

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Custom Exception ---
class OKXError(Exception):
    """Custom exception for OKX API errors."""
    def __init__(self, code: str, message: str, http_status: Optional[int] = None):
        self.code = code
        self.message = message
        self.http_status = http_status
        super().__init__(f"OKX API Error (HTTP: {http_status}) - Code: {code}, Message: {message}")

# --- API Client ---
class OKXClient:
    # MOVE cache variables inside the class
    _instrument_details_cache: Dict[str, Dict[str, Any]] = {}
    _CACHE_TTL_SECONDS: int = 60 # Or the value from server.py (using 60 from server.py)

    # PASTE OKXClient.__init__ method here from server.py
    # Ensure it initializes self.api_key, self.secret_key, self.passphrase
    # Adjust logging if necessary (e.g., use logger.info instead of print)
    def __init__(self):
        """Initializes the OKXClient, loading credentials from environment variables."""
        logger.info("Initializing OKX Client...") # Adjusted log message

        # Load credentials securely
        self.api_key = os.environ.get("OKX_API_KEY")
        self.secret_key = os.environ.get("OKX_SECRET_KEY")
        self.passphrase = os.environ.get("OKX_PASSPHRASE")

        self.has_private_access = bool(self.api_key and self.secret_key and self.passphrase)

        if self.has_private_access:
            logger.info("Private API access configured.")
            # logger.debug(f"API Key loaded: {self.api_key}") # Removed sensitive logging
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
                # logger.debug(f"Secret Key loaded: {self.secret_key[:3]}...{self.secret_key[-3:]}") # Removed sensitive logging
            if self.passphrase:
                 # logger.debug(f"Passphrase loaded: {self.passphrase[:3]}...{self.passphrase[-3:]}") # Removed sensitive logging
                 pass # Keep passphrase loading but remove logging
        else:
            logger.warning("Private API access not configured. Only public endpoints are available.")
            logger.warning("Set OKX_API_KEY, OKX_SECRET_KEY, and OKX_PASSPHRASE environment variables for private access.")

        self.base_url = OKX_BASE_URL

    # PASTE _get_timestamp method here from server.py
    def _get_timestamp(self) -> str:
        """Generates the required ISO 8601 timestamp format for OKX API."""
        return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()) # Using .000Z from server.py

    # PASTE _generate_signature method here from server.py
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

    # PASTE _add_auth_headers method here from server.py
    # Ensure it uses self.api_key, self.passphrase and calls self._generate_signature
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

    # PASTE make_request method here from server.py
    # Ensure it uses OKX_BASE_URL, self._add_auth_headers, handles errors, and raises OKXError
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
                raise OKXError(message=error_msg, code=error_code, http_status=response.status_code) # Changed status_code to http_status

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

                raise OKXError(message=error_with_details, code=api_code, http_status=response.status_code) # Changed status_code to http_status

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

    # CREATE get_instrument_details_cached method here
    # Adapt the logic from the original standalone get_instrument_details function
    # Use self.make_request, self._instrument_details_cache, self._CACHE_TTL_SECONDS
    def get_instrument_details_cached(self, instrument_id: str, instrument_type: str = 'SWAP') -> Optional[List[float]]:
        """
        Gets instrument details (last price, contract value, lot size, min size)
        from the OKX API, using an internal cache. Returns [price, ctVal, lotSz, minSz].
        Returns None if fetching fails.
        """
        now = time.time()
        cache_key = f"{instrument_id}_{instrument_type}"
        cached_entry = self._instrument_details_cache.get(cache_key)

        if cached_entry and (now - cached_entry['timestamp']) < self._CACHE_TTL_SECONDS:
            logger.debug(f"Using cached details for {instrument_id} ({instrument_type})")
            return cached_entry['details']

        logger.info(f"Fetching fresh instrument details for {instrument_id} ({instrument_type})")
        try:
            # 1. Get Ticker (for last price)
            ticker_endpoint = f"{API_V5_PREFIX}/market/ticker"
            ticker_params = {"instId": instrument_id}
            ticker_response = self.make_request("GET", ticker_endpoint, params=ticker_params)

            if not ticker_response.get("data"):
                logger.warning(f"No ticker data found for {instrument_id}")
                return None
            contract_price = float(ticker_response["data"][0]["last"])

            # 2. Get Instrument Info (for ctVal, lotSz, minSz)
            # Re-using the logic from the original get_instrument_details which calls get_instruments
            # However, the instruction is to adapt the logic, not necessarily call get_instruments
            # Let's adapt the logic to directly call make_request for instruments endpoint
            instrument_endpoint = f"{API_V5_PREFIX}/public/instruments"
            instrument_params = {"instType": instrument_type, "instId": instrument_id}
            instrument_response = self.make_request("GET", instrument_endpoint, params=instrument_params)


            if not instrument_response.get("data"):
                logger.warning(f"No instrument data found for {instrument_id} ({instrument_type})")
                return None
            instrument_info = instrument_response["data"][0]

            # Extract details, providing defaults if keys are missing (though they should exist for valid instruments)
            contract_value = float(instrument_info.get("ctVal", 1.0)) # Value of 1 contract in USDT (for linear) or coin (for inverse)
            lot_size = float(instrument_info.get("lotSz", 1.0))       # Number of contracts per lot (usually 1 for SWAP/FUTURES)
            min_size = float(instrument_info.get("minSz", 1.0))       # Minimum order size in contracts

            details = [contract_price, contract_value, lot_size, min_size]

            # Update cache
            self._instrument_details_cache[cache_key] = {'timestamp': now, 'details': details}
            logger.debug(f"Cached instrument details for {instrument_id} ({instrument_type}): {details}")
            return details

        except (KeyError, IndexError, ValueError, TypeError, ConnectionError, OKXError) as e:
             logger.error(f"Error fetching/processing instrument details for {instrument_id} ({instrument_type}): {e}", exc_info=True)
             return None # Return None on failure as per original logic
        except Exception as e:
             logger.exception(f"Unexpected error fetching instrument details for {instrument_id} ({instrument_type}): {e}")
             # Decide if you want to raise a generic error or return None
             return None # Sticking to None for consistency

    # --- End of OKXClient class ---