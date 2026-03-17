"""Output formatting for the WayyDB CLI."""

from __future__ import annotations

import json
import sys
from typing import Any

from rich.console import Console
from rich.json import JSON
from rich.table import Table

console = Console()
err_console = Console(stderr=True)


def print_json_data(data: Any) -> None:
    """Pretty-print JSON data."""
    console.print(JSON(json.dumps(data, default=str)))


def print_table_data(data: dict[str, list[Any]], title: str = "") -> None:
    """Render columnar data as a rich table."""
    if not data:
        console.print("[dim]No data[/dim]")
        return

    table = Table(title=title, show_lines=False)
    columns = list(data.keys())
    for col in columns:
        table.add_column(col, style="cyan")

    num_rows = len(next(iter(data.values())))
    for i in range(num_rows):
        row = [str(data[col][i]) for col in columns]
        table.add_row(*row)

    console.print(table)


def print_rows(rows: list[dict[str, Any]], title: str = "") -> None:
    """Render a list of row dicts as a rich table."""
    if not rows:
        console.print("[dim]No rows[/dim]")
        return

    columns = list(rows[0].keys())
    table = Table(title=title, show_lines=False)
    for col in columns:
        table.add_column(col, style="cyan")

    for row in rows:
        table.add_row(*[str(row.get(col, "")) for col in columns])

    console.print(table)


def print_kv(key: str, value: Any) -> None:
    """Print a KV pair."""
    console.print(f"[bold]{key}[/bold] = ", end="")
    if isinstance(value, (dict, list)):
        print_json_data(value)
    else:
        console.print(str(value))


def print_success(msg: str) -> None:
    console.print(f"[green]{msg}[/green]")


def print_error(msg: str) -> None:
    err_console.print(f"[red]{msg}[/red]")


def print_info(label: str, value: Any) -> None:
    console.print(f"[bold]{label}:[/bold] {value}")
