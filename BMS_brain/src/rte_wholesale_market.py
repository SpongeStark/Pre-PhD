import base64
import os
import logging
from datetime import datetime, timedelta
import requests
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RTEWholesaleMarketClient:
    """
    A Python client to interact with the RTE (Réseau de Transport d'Électricité) 
    Wholesale Market API (v3) to recover wholesale market power prices in France.
    """
    
    TOKEN_PATH = "/token/oauth/"
    PRICES_PATH = "/open_api/wholesale_market/v3/france_power_exchanges"
    
    def __init__(self, client_id=None, client_secret=None, base_url="https://digital.iservices.rte-france.com"):
        """
        Initialize the RTE Wholesale Market client.
        
        Parameters:
        -----------
        client_id : str, optional
            RTE developer API Client ID. If not provided, looks for RTE_CLIENT_ID env var.
        client_secret : str, optional
            RTE developer API Client Secret. If not provided, looks for RTE_CLIENT_SECRET env var.
        base_url : str, optional
            Base URL of the RTE API. Defaults to 'https://digital.iservices.rte-france.com'.
        """
        self.base_url = base_url.rstrip('/')
        self.client_id = client_id or os.environ.get("RTE_CLIENT_ID") or "20d74094-8042-493d-a764-ce01d80c6afd"
        self.client_secret = client_secret or os.environ.get("RTE_CLIENT_SECRET") or "64a24921-ecdd-445b-a467-4cad61aac041"
        
        self._access_token = None
        self._token_expires_at = None

    def _format_datetime(self, dt):
        """
        Format datetime objects to ISO 8601 format with timezone offset as required by RTE API.
        If the datetime is naive, it defaults to UTC (appending 'Z').
        """
        if isinstance(dt, str):
            return dt
            
        if not isinstance(dt, (datetime, pd.Timestamp)):
            raise TypeError("Expected string, datetime, or pandas Timestamp object.")

        if dt.tzinfo is None:
            # Naive datetime: assume UTC
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            
        return dt.isoformat()

    def get_access_token(self, force_refresh=False):
        """
        Request a new access token using the OAuth2 Client Credentials flow.
        
        Returns:
        --------
        str
            The OAuth2 access token.
        """
        if self._access_token and not force_refresh:
            # Check if token is still valid (with 60 seconds buffer)
            if self._token_expires_at and datetime.now() < self._token_expires_at - timedelta(seconds=60):
                return self._access_token

        if not self.client_id or not self.client_secret:
            raise ValueError(
                "RTE Client ID and Client Secret must be provided either "
                "during initialization or via environment variables RTE_CLIENT_ID and RTE_CLIENT_SECRET."
            )

        token_url = f"{self.base_url}{self.TOKEN_PATH}"
        
        # Credentials formatting for Basic Auth
        credentials = f"{self.client_id}:{self.client_secret}"
        encoded_creds = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
        
        headers = {
            "Authorization": f"Basic {encoded_creds}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        data = {
            "grant_type": "client_credentials"
        }
        
        logger.info(f"Requesting OAuth2 access token from {token_url}...")
        try:
            response = requests.post(token_url, headers=headers, data=data, timeout=10)
            response.raise_for_status()
            
            token_info = response.json()
            self._access_token = token_info.get("access_token")
            expires_in = token_info.get("expires_in", 7200)  # Default: 2 hours
            self._token_expires_at = datetime.now() + timedelta(seconds=expires_in)
            
            logger.info("Access token successfully retrieved and cached.")
            return self._access_token
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to obtain access token: {e}")
            if 'response' in locals() and response is not None:
                logger.error(f"Response status: {response.status_code}, Body: {response.text}")
            raise

    def fetch_wholesale_prices(self, start_date, end_date):
        """
        Fetch the wholesale market power exchange prices in France from the RTE API.
        
        Parameters:
        -----------
        start_date : str or datetime
            Start date of the interval (e.g. datetime or ISO 8601 string '2026-07-13T00:00:00Z').
        end_date : str or datetime
            End date of the interval (e.g. datetime or ISO 8601 string '2026-07-14T00:00:00Z').
            
        Returns:
        --------
        pd.DataFrame
            A pandas DataFrame containing wholesale market price data.
        """
        start_str = self._format_datetime(start_date)
        end_str = self._format_datetime(end_date)
        
        token = self.get_access_token()
        api_url = f"{self.base_url}{self.PRICES_PATH}"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }
        
        params = {
            "start_date": start_str,
            "end_date": end_str
        }
        
        logger.info(f"Requesting prices from {start_str} to {end_str} via {api_url}...")
        try:
            response = requests.get(api_url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            df = self.parse_prices(data)
            return self.apply_french_taxes_and_fees(df)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch wholesale prices: {e}")
            if 'response' in locals() and response is not None:
                logger.error(f"Response status: {response.status_code}, Body: {response.text}")
            raise

    def parse_prices(self, json_data):
        """
        Parse the JSON response from the RTE Wholesale Market API into a clean DataFrame.
        Provides robust fallbacks to accommodate potential minor schema changes.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns ['start_time', 'end_time', 'price_eur_mwh', 'currency']
        """
        records = []
        
        # Retrieve main list
        exchanges = json_data.get("france_power_exchanges", [])
        
        # Fallback 1: case-insensitive match or contains-check
        if not exchanges:
            for key, val in json_data.items():
                if "france" in key.lower() and "exchange" in key.lower() and isinstance(val, list):
                    exchanges = val
                    break
        
        # Fallback 2: first list element in JSON structure
        if not exchanges:
            for key, val in json_data.items():
                if isinstance(val, list):
                    exchanges = val
                    break
                    
        # Parse items
        for item in exchanges:
            if not isinstance(item, dict):
                continue
                
            values = item.get("values", [])
            # Fallback 3: if structure is flat rather than nested
            if not values and "price" in item:
                values = [item]
                
            for val in values:
                if not isinstance(val, dict):
                    continue
                
                # Check different naming variations
                start_date = val.get("start_date") or val.get("startDate") or val.get("start")
                end_date = val.get("end_date") or val.get("endDate") or val.get("end")
                
                # Check price explicitly first to handle 0.0 correctly
                price = val.get("price")
                if price is None:
                    price = val.get("value")
                    
                currency = val.get("price_currency") or val.get("currency") or "EUR"
                
                if start_date and price is not None:
                    try:
                        records.append({
                            "start_time": pd.to_datetime(start_date),
                            "end_time": pd.to_datetime(end_date) if end_date else None,
                            "price_eur_mwh": float(price),
                            "currency": currency
                        })
                    except Exception as parse_err:
                        logger.warning(f"Skipping record due to parsing error: {val}. Error: {parse_err}")
                        
        if not records:
            logger.warning("No price records could be successfully parsed.")
            return pd.DataFrame()
            
        df = pd.DataFrame(records)
        df = df.sort_values(by="start_time").reset_index(drop=True)
        return df

    def apply_french_taxes_and_fees(self, df):
        """
        Applies French regulatory taxes (Accise/CSPE) and network fees (TURPE) 
        to the wholesale electricity prices.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing at least 'start_time' and 'price_eur_mwh' columns.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with additional columns:
            - 'price_eur_kwh': Wholesale market price in EUR/kWh
            - 'accise_eur_kwh': Accise sur l'électricité tax in EUR/kWh
            - 'turpe_eur_kwh': TURPE network fee in EUR/kWh
            - 'total_price_eur_kwh': Total price including wholesale, tax, and network fee in EUR/kWh
        """
        if df.empty:
            return df
            
        df = df.copy()
        
        # 1. Convert wholesale price from EUR/MWh to EUR/kWh
        df['price_eur_kwh'] = df['price_eur_mwh'] / 1000.0
        
        # Ensure start_time is pandas DatetimeIndex (handling timezone if present)
        times = pd.to_datetime(df['start_time'])
        
        # 2. Calculate Accise sur l'électricité (CSPE / TICFE)
        # - 2022-2023: 0.0005 €/kWh
        # - Feb 1, 2024 to Dec 31, 2024: 0.021 €/kWh
        # - 2025: 0.025 €/kWh average
        # - 2026+: 0.03085 €/kWh
        accise = pd.Series(0.0005, index=df.index)
        
        # Use timezone-aware timestamps if the times series has a timezone
        tz = times.dt.tz
        feb_1_2024 = pd.Timestamp('2024-02-01', tz=tz)
        jan_1_2025 = pd.Timestamp('2025-01-01', tz=tz)
        jan_1_2026 = pd.Timestamp('2026-01-01', tz=tz)
        
        accise.loc[(times >= feb_1_2024) & (times < jan_1_2025)] = 0.021
        accise.loc[(times >= jan_1_2025) & (times < jan_1_2026)] = 0.025
        accise.loc[times >= jan_1_2026] = 0.03085
        
        df['accise_eur_kwh'] = accise
        
        # 3. Calculate TURPE network fee
        # - 2022: 0.020 €/kWh
        # - 2023: 0.021 €/kWh
        # - 2024: 0.022 €/kWh
        # - 2025: 0.0235 €/kWh
        # - 2026+: 0.025 €/kWh
        turpe = pd.Series(0.020, index=df.index)
        
        years = times.dt.year
        turpe.loc[years == 2023] = 0.021
        turpe.loc[years == 2024] = 0.022
        turpe.loc[years == 2025] = 0.0235
        turpe.loc[years >= 2026] = 0.025
        
        df['turpe_eur_kwh'] = turpe
        
        # 4. Calculate total price
        df['total_price_eur_kwh'] = df['price_eur_kwh'] + df['accise_eur_kwh'] + df['turpe_eur_kwh']
        
        return df

    def get_representative_24h_profile(self):
        """
        Fetches the latest available prices and extracts a clean 24-hour profile
        (96 steps of 15-minute intervals) of total prices in EUR/kWh.
        
        Returns:
        --------
        dict
            A dictionary mapping (hour, minute) tuple to total_price_eur_kwh,
            or None if the fetch fails.
        """
        try:
            # Fetch prices for the last 2 days to ensure we cover a full 24-hour period
            end = datetime.now()
            start = end - timedelta(days=2)
            df = self.fetch_wholesale_prices(start, end)
            
            if df.empty:
                return None
                
            # Group by hour and minute and take the mean total price
            times = pd.to_datetime(df['start_time'])
            df['hour'] = times.dt.hour
            df['minute'] = times.dt.minute
            
            profile = df.groupby(['hour', 'minute'])['total_price_eur_kwh'].mean().to_dict()
            return profile
        except Exception as e:
            logger.warning(f"Could not retrieve representative 24h price profile: {e}")
            return None

if __name__ == "__main__":
    import sys
    print("=========================================================================")
    print("RTE Wholesale Market Price Retriever")
    print("=========================================================================")
    print("Usage Instructions:")
    print("1. Set your RTE API credentials as environment variables:")
    print("   On Windows (PowerShell):")
    print("     $env:RTE_CLIENT_ID = 'your_client_id_here'")
    print("     $env:RTE_CLIENT_SECRET = 'your_client_secret_here'")
    print("   On Linux/macOS:")
    print("     export RTE_CLIENT_ID='your_client_id_here'")
    print("     export RTE_CLIENT_SECRET='your_client_secret_here'")
    print("\n2. Run this script directly to test the connection:")
    print("     python rte_wholesale_market.py")
    print("=========================================================================")
    
    # Simple test invocation
    client_id = os.environ.get("RTE_CLIENT_ID") or "20d74094-8042-493d-a764-ce01d80c6afd"
    client_secret = os.environ.get("RTE_CLIENT_SECRET") or "64a24921-ecdd-445b-a467-4cad61aac041"
    
    if not os.environ.get("RTE_CLIENT_ID") or not os.environ.get("RTE_CLIENT_SECRET"):
        print("\n[NOTE] Environment variables RTE_CLIENT_ID/RTE_CLIENT_SECRET not found.")
        print("Using provided default credentials for the live demo.")
        
    try:
        # Fetching for the last 2 days
        client = RTEWholesaleMarketClient(client_id=client_id, client_secret=client_secret)
        end = datetime.now()
        start = end - timedelta(days=2)
        
        print(f"\nAttempting live fetch from RTE API for range: {start.date()} to {end.date()}...")
        df = client.fetch_wholesale_prices(start, end)
        
        if not df.empty:
            print("\nSuccessfully fetched wholesale prices:")
            print(df.head(10))
            print(f"\nTotal records retrieved: {len(df)}")
        else:
            print("\nAPI query completed successfully, but returned an empty dataset.")
            
    except Exception as e:
        print(f"\n[ERROR] An error occurred during the live demonstration: {e}")
