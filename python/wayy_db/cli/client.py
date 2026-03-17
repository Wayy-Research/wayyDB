"""HTTP client for the WayyDB service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NoReturn, Optional

import httpx

from wayy_db.cli.config import get_server_url

# The API uses /api/v1/{db_name}/... for OLTP routes but db_name is unused
# server-side (single global db). We hardcode "default" for forward compat.
_DB_NAME = "default"


class WayyClientError(Exception):
    """Raised when the WayyDB service returns an error."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class WayyClient:
    """Synchronous HTTP client for the WayyDB REST API."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0) -> None:
        self.base_url = (base_url or get_server_url()).rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Make an HTTP request and return JSON response."""
        try:
            resp = self._client.request(method, path, **kwargs)
        except httpx.ConnectError:
            raise WayyClientError(0, f"Cannot connect to {self.base_url}")
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise WayyClientError(resp.status_code, detail)
        if resp.status_code == 204 or not resp.content:
            return {}
        return resp.json()

    # --- Health ---

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def info(self) -> dict[str, Any]:
        return self._request("GET", "/")

    # --- Tables ---

    def list_tables(self) -> list[str]:
        data = self._request("GET", "/tables")
        return data.get("tables", [])

    def get_table_info(self, name: str) -> dict[str, Any]:
        return self._request("GET", f"/tables/{name}")

    def get_table_data(
        self, name: str, limit: int = 100, offset: int = 0
    ) -> dict[str, Any]:
        return self._request(
            "GET", f"/tables/{name}/data", params={"limit": limit, "offset": offset}
        )

    def create_table(
        self,
        name: str,
        columns: list[dict[str, str]],
        primary_key: Optional[str] = None,
        sorted_by: Optional[str] = None,
    ) -> dict[str, Any]:
        payload = {
            "name": name,
            "columns": columns,
            "primary_key": primary_key,
            "sorted_by": sorted_by,
        }
        return self._request("POST", f"/api/v1/{_DB_NAME}/tables", json=payload)

    def drop_table(self, name: str) -> dict[str, Any]:
        return self._request("DELETE", f"/tables/{name}")

    def upload_table(self, table_data: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/tables/upload", json=table_data)

    def append_rows(self, name: str, columns: list[dict[str, Any]]) -> dict[str, Any]:
        return self._request("POST", f"/tables/{name}/append", json={"columns": columns})

    # --- OLTP ---

    def insert_row(self, table: str, data: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "POST", f"/api/v1/{_DB_NAME}/tables/{table}/rows", json={"data": data}
        )

    def get_row(self, table: str, pk: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/{_DB_NAME}/tables/{table}/rows/{pk}")

    def update_row(self, table: str, pk: str, data: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "PUT", f"/api/v1/{_DB_NAME}/tables/{table}/rows/{pk}", json={"data": data}
        )

    def delete_row(self, table: str, pk: str) -> dict[str, Any]:
        return self._request("DELETE", f"/api/v1/{_DB_NAME}/tables/{table}/rows/{pk}")

    def filter_rows(
        self, table: str, filters: Optional[dict[str, str]] = None, limit: int = 500
    ) -> dict[str, Any]:
        params = dict(filters or {})
        params["limit"] = str(limit)
        return self._request(
            "GET", f"/api/v1/{_DB_NAME}/tables/{table}/rows", params=params
        )

    # --- Aggregations ---

    def aggregate(self, table: str, column: str, op: str) -> dict[str, Any]:
        return self._request("GET", f"/tables/{table}/agg/{column}/{op}")

    # --- Joins ---

    def as_of_join(
        self, left: str, right: str, on: list[str], as_of: str
    ) -> dict[str, Any]:
        payload = {"left_table": left, "right_table": right, "on": on, "as_of": as_of}
        return self._request("POST", "/join/aj", json=payload)

    def window_join(
        self,
        left: str,
        right: str,
        on: list[str],
        as_of: str,
        before: int,
        after: int,
    ) -> dict[str, Any]:
        payload = {
            "left_table": left,
            "right_table": right,
            "on": on,
            "as_of": as_of,
            "window_before": before,
            "window_after": after,
        }
        return self._request("POST", "/join/wj", json=payload)

    # --- Window functions ---

    def window_function(
        self,
        table: str,
        column: str,
        operation: str,
        window: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "table": table,
            "column": column,
            "operation": operation,
        }
        if window is not None:
            payload["window"] = window
        if alpha is not None:
            payload["alpha"] = alpha
        return self._request("POST", "/window", json=payload)

    # --- Streaming ---

    def ingest_tick(self, table: str, tick: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/ingest/{table}", json=tick)

    def ingest_batch(self, table: str, ticks: list[dict[str, Any]]) -> dict[str, Any]:
        return self._request("POST", f"/ingest/{table}/batch", json={"ticks": ticks})

    def get_streaming_stats(self) -> dict[str, Any]:
        return self._request("GET", "/streaming/stats")

    def get_quote(self, table: str, symbol: str) -> dict[str, Any]:
        return self._request("GET", f"/streaming/quote/{table}/{symbol}")

    def get_all_quotes(self, table: str) -> dict[str, Any]:
        return self._request("GET", f"/streaming/quotes/{table}")

    # --- KV Store ---

    def kv_get(self, key: str) -> Any:
        data = self._request("GET", f"/kv/{key}")
        return data.get("value")

    def kv_set(self, key: str, value: Any, ttl: Optional[float] = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"value": value}
        if ttl is not None:
            payload["ttl"] = ttl
        return self._request("POST", f"/kv/{key}", json=payload)

    def kv_delete(self, key: str) -> dict[str, Any]:
        return self._request("DELETE", f"/kv/{key}")

    def kv_list(self, pattern: Optional[str] = None) -> list[str]:
        params = {}
        if pattern:
            params["pattern"] = pattern
        data = self._request("GET", "/kv", params=params)
        return data.get("keys", [])

    # --- Checkpoint ---

    def checkpoint(self) -> dict[str, Any]:
        return self._request("POST", f"/api/v1/{_DB_NAME}/checkpoint")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "WayyClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def upload_csv(
    client: WayyClient, name: str, file_path: Path, sorted_by: Optional[str] = None
) -> dict[str, Any]:
    """Read a CSV file and upload it as a table.

    Uses stdlib csv to avoid requiring pandas in CLI.
    """
    import csv

    with open(file_path, newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV file is empty (no data rows)")

    columns: list[dict[str, Any]] = []
    for i, header in enumerate(headers):
        raw_values = [row[i] for row in rows]
        dtype, data = _infer_column(raw_values)
        columns.append({"name": header, "dtype": dtype, "data": data})

    payload = {"name": name, "columns": columns, "sorted_by": sorted_by}
    return client.upload_table(payload)


def _infer_column(values: list[str]) -> tuple[str, list[Any]]:
    """Infer column dtype from string values. Returns (dtype_name, typed_data)."""
    non_empty = [v for v in values if v.strip()]
    if not non_empty:
        return ("float64", [0.0] * len(values))

    # Try int64
    try:
        data = [int(v) if v.strip() else 0 for v in values]
        return ("int64", data)
    except (ValueError, OverflowError):
        pass

    # Try float64 (handles empty cells as NaN)
    try:
        data = [float(v) if v.strip() else float("nan") for v in values]
        return ("float64", data)
    except (ValueError, OverflowError):
        pass

    raise ValueError(
        f"Non-numeric column detected. Values: {values[:3]}... "
        "CSV upload currently supports numeric columns only. "
        "Use the Python API with from_pandas() for string/symbol columns."
    )


def upload_json_ticks(
    client: WayyClient, table: str, file_path: Path
) -> dict[str, Any]:
    """Read a JSON file of ticks and batch-ingest them."""
    with open(file_path) as f:
        data = json.load(f)

    if isinstance(data, list):
        ticks = data
    elif isinstance(data, dict) and "ticks" in data:
        ticks = data["ticks"]
    else:
        raise ValueError("JSON must be a list of ticks or {\"ticks\": [...]}")

    return client.ingest_batch(table, ticks)
