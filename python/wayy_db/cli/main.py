"""WayyDB CLI — command-line interface for the WayyDB service.

Usage:
    wayy status                          Check server health
    wayy connect <url>                   Set server URL
    wayy tables                          List all tables
    wayy create <name> --schema '{}'     Create a table with schema
    wayy query <table>                   Query table data
    wayy upload <name> --file data.csv   Upload CSV as a table
    wayy agg <table> <col> <op>          Run aggregation
    wayy stream <table>                  Subscribe to live updates
    wayy ingest <table> --file ticks.json  Batch ingest ticks
    wayy kv get/set/del <key>            Key-value operations
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NoReturn, Optional

import typer

from wayy_db.cli.client import WayyClient, WayyClientError, upload_csv, upload_json_ticks
from wayy_db.cli.config import get_server_url, load_config, save_config
from wayy_db.cli.deploy import deploy_app
from wayy_db.cli.output import (
    console,
    print_error,
    print_info,
    print_json_data,
    print_kv,
    print_rows,
    print_success,
    print_table_data,
)

app = typer.Typer(
    name="wayy",
    help="WayyDB CLI — high-performance columnar time-series database",
    no_args_is_help=True,
    add_completion=False,
)


def _handle_error(e: WayyClientError) -> NoReturn:
    if e.status_code == 0:
        print_error(f"Connection failed: {e.detail}")
    else:
        print_error(f"Error {e.status_code}: {e.detail}")
    raise typer.Exit(1)


# --- Connection ---


@app.command()
def connect(url: str = typer.Argument(..., help="WayyDB server URL")) -> None:
    """Set the WayyDB server URL."""
    url = url.rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"

    try:
        with WayyClient(base_url=url) as client:
            info = client.health()
    except WayyClientError as e:
        print_error(f"Cannot reach {url}: {e.detail}")
        raise typer.Exit(1)

    config = load_config()
    config["server_url"] = url
    save_config(config)
    print_success(f"Connected to {url}")
    print_info("Tables", info.get("tables", 0))


@app.command()
def status() -> None:
    """Check server health and connection info."""
    url = get_server_url()
    print_info("Server", url)

    try:
        with WayyClient() as client:
            info = client.info()
            health = client.health()
    except WayyClientError as e:
        _handle_error(e)

    print_info("Service", info.get("service", "?"))
    print_info("Version", info.get("version", "?"))
    print_info("Status", health.get("status", "?"))
    print_info("Tables", health.get("tables", 0))


# --- Tables ---


@app.command()
def tables() -> None:
    """List all tables in the database."""
    try:
        with WayyClient() as client:
            table_list = client.list_tables()
    except WayyClientError as e:
        _handle_error(e)

    if not table_list:
        console.print("[dim]No tables[/dim]")
        return

    for t in table_list:
        console.print(f"  {t}")


@app.command()
def create(
    name: str = typer.Argument(..., help="Table name"),
    schema: str = typer.Option(
        ..., "--schema", "-s",
        help='Column schema as JSON: \'{"ts": "timestamp", "price": "float64"}\'',
    ),
    primary_key: Optional[str] = typer.Option(None, "--pk", help="Primary key column"),
    sorted_by: Optional[str] = typer.Option(None, "--sorted-by", help="Sorted index column"),
) -> None:
    """Create a new table with a typed schema."""
    try:
        schema_dict = json.loads(schema)
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON schema: {e}")
        raise typer.Exit(1)

    columns = [{"name": k, "dtype": v} for k, v in schema_dict.items()]

    try:
        with WayyClient() as client:
            result = client.create_table(name, columns, primary_key=primary_key, sorted_by=sorted_by)
    except WayyClientError as e:
        _handle_error(e)

    print_success(f"Created table '{name}' with columns: {result.get('columns', [])}")


@app.command()
def drop(name: str = typer.Argument(..., help="Table name to delete")) -> None:
    """Drop a table."""
    try:
        with WayyClient() as client:
            client.drop_table(name)
    except WayyClientError as e:
        _handle_error(e)

    print_success(f"Dropped table '{name}'")


@app.command()
def info(name: str = typer.Argument(..., help="Table name")) -> None:
    """Get table metadata."""
    try:
        with WayyClient() as client:
            data = client.get_table_info(name)
    except WayyClientError as e:
        _handle_error(e)

    print_info("Name", data.get("name"))
    print_info("Rows", data.get("num_rows"))
    print_info("Columns", data.get("num_columns"))
    print_info("Column names", ", ".join(data.get("columns", [])))
    print_info("Sorted by", data.get("sorted_by") or "none")


@app.command()
def query(
    table: str = typer.Argument(..., help="Table name"),
    limit: int = typer.Option(100, "--limit", "-n", help="Max rows to return"),
    offset: int = typer.Option(0, "--offset", help="Row offset"),
    where: Optional[list[str]] = typer.Option(None, "--where", "-w", help="Filter as col=val"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Query table data."""
    try:
        with WayyClient() as client:
            if where:
                filters = {}
                for w in where:
                    if "=" not in w:
                        print_error(f"Invalid filter: {w} (expected col=val)")
                        raise typer.Exit(1)
                    k, v = w.split("=", 1)
                    filters[k] = v

                result = client.filter_rows(table, filters=filters, limit=limit)

                if output_json:
                    print_json_data(result)
                else:
                    print_rows(result.get("data", []), title=f"{table} ({result.get('count', 0)} rows)")
            else:
                result = client.get_table_data(table, limit=limit, offset=offset)

                if output_json:
                    print_json_data(result)
                else:
                    data = result.get("data", {})
                    total = result.get("total_rows", 0)
                    shown = len(next(iter(data.values()))) if data else 0
                    print_table_data(data, title=f"{table} ({shown}/{total} rows)")

    except WayyClientError as e:
        _handle_error(e)


@app.command()
def upload(
    name: str = typer.Argument(..., help="Table name"),
    file: Path = typer.Option(..., "--file", "-f", help="CSV file to upload"),
    sorted_by: Optional[str] = typer.Option(None, "--sorted-by", help="Sorted index column"),
) -> None:
    """Upload a CSV file as a new table."""
    if not file.exists():
        print_error(f"File not found: {file}")
        raise typer.Exit(1)

    try:
        with WayyClient() as client:
            result = upload_csv(client, name, file, sorted_by=sorted_by)
    except WayyClientError as e:
        _handle_error(e)
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)

    print_success(f"Uploaded '{name}': {result.get('rows', 0)} rows, columns: {result.get('columns', [])}")


# --- Aggregations ---


@app.command()
def agg(
    table: str = typer.Argument(..., help="Table name"),
    column: str = typer.Argument(..., help="Column name"),
    op: str = typer.Argument(..., help="Operation: sum, avg, min, max, std"),
) -> None:
    """Run an aggregation on a table column."""
    try:
        with WayyClient() as client:
            result = client.aggregate(table, column, op)
    except WayyClientError as e:
        _handle_error(e)

    console.print(f"[bold]{op}[/bold]({table}.{column}) = [cyan]{result.get('result')}[/cyan]")


# --- Streaming ---


@app.command()
def stream(
    table: str = typer.Argument(..., help="Table name to subscribe to"),
    symbols: Optional[str] = typer.Option(None, "--symbols", "-s", help="Comma-separated symbol filter"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output raw JSON"),
) -> None:
    """Subscribe to real-time streaming updates via WebSocket."""
    import asyncio

    async def _stream() -> None:
        import websockets

        url = get_server_url().replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{url}/ws/subscribe/{table}"

        console.print(f"[dim]Connecting to {ws_url}...[/dim]")

        async with websockets.connect(ws_url) as ws:
            if symbols:
                symbol_list = [s.strip() for s in symbols.split(",")]
                await ws.send(json.dumps({"symbols": symbol_list}))
                console.print(f"[dim]Filtering: {symbol_list}[/dim]")

            console.print("[green]Connected.[/green] Press Ctrl+C to disconnect.\n")

            try:
                async for message in ws:
                    data = json.loads(message)
                    if output_json:
                        print_json_data(data)
                    else:
                        if "batch" in data:
                            for tick in data["batch"]:
                                _print_tick(tick)
                        else:
                            _print_tick(data)
            except asyncio.CancelledError:
                pass

    try:
        asyncio.run(_stream())
    except KeyboardInterrupt:
        console.print("\n[dim]Disconnected.[/dim]")


def _print_tick(tick: dict[str, Any]) -> None:
    """Format a single tick for display."""
    sym = tick.get("symbol", "?")
    price = tick.get("price", "?")
    vol = tick.get("volume", "")
    bid = tick.get("bid", "")
    ask = tick.get("ask", "")

    parts = [f"[bold]{sym}[/bold]", f"[cyan]{price}[/cyan]"]
    if bid and ask:
        parts.append(f"[dim]{bid}/{ask}[/dim]")
    if vol:
        parts.append(f"vol={vol}")

    console.print(" ".join(parts))


# --- Ingestion ---


@app.command()
def ingest(
    table: str = typer.Argument(..., help="Table name"),
    file: Path = typer.Option(..., "--file", "-f", help="JSON file with ticks"),
) -> None:
    """Batch ingest ticks from a JSON file."""
    if not file.exists():
        print_error(f"File not found: {file}")
        raise typer.Exit(1)

    try:
        with WayyClient() as client:
            result = upload_json_ticks(client, table, file)
    except WayyClientError as e:
        _handle_error(e)
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)

    print_success(f"Ingested {result.get('ingested', 0)} ticks into '{table}'")


# --- KV Store ---

kv_app = typer.Typer(name="kv", help="Key-value store operations", no_args_is_help=True)
app.add_typer(kv_app)


@kv_app.command("get")
def kv_get(key: str = typer.Argument(..., help="Key to get")) -> None:
    """Get a value by key."""
    try:
        with WayyClient() as client:
            value = client.kv_get(key)
    except WayyClientError as e:
        _handle_error(e)

    print_kv(key, value)


@kv_app.command("set")
def kv_set(
    key: str = typer.Argument(..., help="Key to set"),
    value: str = typer.Argument(..., help="Value (JSON or string)"),
    ttl: Optional[float] = typer.Option(None, "--ttl", help="TTL in seconds"),
) -> None:
    """Set a key-value pair."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value

    try:
        with WayyClient() as client:
            client.kv_set(key, parsed, ttl=ttl)
    except WayyClientError as e:
        _handle_error(e)

    print_success(f"Set '{key}'")


@kv_app.command("del")
def kv_del(key: str = typer.Argument(..., help="Key to delete")) -> None:
    """Delete a key."""
    try:
        with WayyClient() as client:
            client.kv_delete(key)
    except WayyClientError as e:
        _handle_error(e)

    print_success(f"Deleted '{key}'")


@kv_app.command("list")
def kv_list(pattern: Optional[str] = typer.Argument(None, help="Glob pattern filter")) -> None:
    """List keys, optionally filtered by pattern."""
    try:
        with WayyClient() as client:
            keys = client.kv_list(pattern)
    except WayyClientError as e:
        _handle_error(e)

    if not keys:
        console.print("[dim]No keys[/dim]")
        return

    for k in keys:
        console.print(f"  {k}")


# --- Joins ---

join_app = typer.Typer(name="join", help="Join operations", no_args_is_help=True)
app.add_typer(join_app)


@join_app.command("aj")
def join_aj(
    left: str = typer.Argument(..., help="Left table"),
    right: str = typer.Argument(..., help="Right table"),
    on: str = typer.Option(..., "--on", help="Join keys (comma-separated)"),
    as_of: str = typer.Option(..., "--as-of", help="Temporal column"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """As-of join: find most recent right row for each left row."""
    on_cols = [c.strip() for c in on.split(",")]

    try:
        with WayyClient() as client:
            result = client.as_of_join(left, right, on_cols, as_of)
    except WayyClientError as e:
        _handle_error(e)

    if output_json:
        print_json_data(result)
    else:
        print_table_data(result.get("data", {}), title=f"aj({left}, {right}) — {result.get('rows', 0)} rows")


@join_app.command("wj")
def join_wj(
    left: str = typer.Argument(..., help="Left table"),
    right: str = typer.Argument(..., help="Right table"),
    on: str = typer.Option(..., "--on", help="Join keys (comma-separated)"),
    as_of: str = typer.Option(..., "--as-of", help="Temporal column"),
    before: int = typer.Option(..., "--before", help="Window before (nanoseconds)"),
    after: int = typer.Option(..., "--after", help="Window after (nanoseconds)"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Window join: find all right rows within time window."""
    on_cols = [c.strip() for c in on.split(",")]

    try:
        with WayyClient() as client:
            result = client.window_join(left, right, on_cols, as_of, before, after)
    except WayyClientError as e:
        _handle_error(e)

    if output_json:
        print_json_data(result)
    else:
        print_table_data(result.get("data", {}), title=f"wj({left}, {right}) — {result.get('rows', 0)} rows")


# --- Window Functions ---


@app.command("window")
def window_fn(
    table: str = typer.Argument(..., help="Table name"),
    column: str = typer.Argument(..., help="Column name"),
    op: str = typer.Argument(..., help="Operation: mavg, msum, mstd, mmin, mmax, ema, diff, pct_change"),
    window: Optional[int] = typer.Option(None, "--window", "-w", help="Window size"),
    alpha: Optional[float] = typer.Option(None, "--alpha", help="EMA alpha"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    """Apply a window function to a column."""
    try:
        with WayyClient() as client:
            result = client.window_function(table, column, op, window=window, alpha=alpha)
    except WayyClientError as e:
        _handle_error(e)

    if output_json:
        print_json_data(result)
    else:
        values = result.get("result", [])
        console.print(f"[bold]{op}[/bold]({table}.{column}) — {len(values)} values")
        if len(values) <= 20:
            for v in values:
                console.print(f"  {v}")
        else:
            for v in values[:5]:
                console.print(f"  {v}")
            console.print(f"  ... ({len(values) - 10} more)")
            for v in values[-5:]:
                console.print(f"  {v}")


# --- Checkpoint ---


@app.command()
def checkpoint() -> None:
    """Flush WAL and save all tables to disk."""
    try:
        with WayyClient() as client:
            client.checkpoint()
    except WayyClientError as e:
        _handle_error(e)

    print_success("Checkpoint complete")


app.add_typer(deploy_app)


if __name__ == "__main__":
    app()
