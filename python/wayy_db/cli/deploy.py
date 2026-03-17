"""Deployment commands for the WayyDB CLI.

Supports:
- Cloud: deploy to Wayy Cloud (Fly.io) — the production path
- Local: start uvicorn directly
- Docker: build and run container locally
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from wayy_db.cli.config import load_config, save_config
from wayy_db.cli.output import console, print_error, print_info, print_success

deploy_app = typer.Typer(
    name="deploy",
    help="Deploy WayyDB service",
    no_args_is_help=True,
)

CLOUD_APP_NAME = "wayydb"
CLOUD_REGION = "ewr"
CLOUD_URL = "https://api.wayydb.com"


def _find_project_root() -> Path:
    """Walk up from cwd looking for pyproject.toml with wayy-db."""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        toml = parent / "pyproject.toml"
        if toml.exists() and "wayy-db" in toml.read_text():
            return parent
    raise FileNotFoundError(
        "Cannot find wayyDB project root (no pyproject.toml with wayy-db found). "
        "Run this command from within the wayyDB repo."
    )


def _run(
    cmd: list[str], cwd: Optional[Path] = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with live output."""
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    return subprocess.run(cmd, cwd=cwd, check=check, text=True)


def _fly_cmd() -> str:
    """Find the fly CLI binary."""
    for name in ("fly", "flyctl"):
        path = shutil.which(name)
        if path:
            return path
    raise FileNotFoundError("fly CLI not found. Install: https://fly.io/docs/flyctl/install/")


# --- Cloud (Fly.io) ---


@deploy_app.command("cloud")
def deploy_cloud(
    app_name: str = typer.Option(CLOUD_APP_NAME, "--app", "-a", help="Fly app name"),
    region: str = typer.Option(CLOUD_REGION, "--region", "-r", help="Fly.io region"),
    create: bool = typer.Option(False, "--create", help="Create the app if it doesn't exist"),
    vm_size: str = typer.Option("shared-cpu-2x", "--vm", help="VM size"),
    memory: str = typer.Option("1gb", "--memory", "-m", help="Memory allocation"),
) -> None:
    """Deploy WayyDB to Wayy Cloud (Fly.io).

    First time: wayy deploy cloud --create
    Updates:    wayy deploy cloud
    """
    try:
        fly = _fly_cmd()
    except FileNotFoundError as e:
        print_error(str(e))
        raise typer.Exit(1)

    try:
        root = _find_project_root()
    except FileNotFoundError as e:
        print_error(str(e))
        raise typer.Exit(1)

    if create:
        console.print(f"[bold]Creating Fly app '{app_name}' in {region}...[/bold]")

        # Create the app
        _run([fly, "apps", "create", app_name, "--org", "personal"], cwd=root, check=False)

        # Create persistent volume for data
        console.print("[bold]Creating persistent volume...[/bold]")
        _run(
            [fly, "volumes", "create", "wayydb_data",
             "--app", app_name,
             "--region", region,
             "--size", "1",
             "--yes"],
            cwd=root,
            check=False,
        )

    # Deploy
    console.print(f"[bold]Deploying to {app_name}.fly.dev...[/bold]")

    # Convert memory like "1gb" to megabytes for fly CLI
    mem_mb = memory
    if memory.lower().endswith("gb"):
        mem_mb = str(int(memory[:-2]) * 1024)
    elif memory.lower().endswith("mb"):
        mem_mb = memory[:-2]

    try:
        _run(
            [fly, "deploy",
             "--app", app_name,
             "--vm-size", vm_size,
             "--vm-memory", mem_mb,
             "--wait-timeout", "300",
             "--yes"],
            cwd=root,
        )
    except subprocess.CalledProcessError:
        print_error("Deploy failed. Check fly logs for details.")
        raise typer.Exit(1)

    url = f"https://{app_name}.fly.dev"
    print_success(f"Deployed to Wayy Cloud")
    print_info("URL", url)
    print_info("Health", f"{url}/health")

    # Auto-save the cloud URL to config
    config = load_config()
    config["server_url"] = url
    config["cloud_app"] = app_name
    save_config(config)

    console.print(f"\n[dim]Connected. Try: wayy status[/dim]")


@deploy_app.command("cloud-status")
def cloud_status(
    app_name: str = typer.Option(CLOUD_APP_NAME, "--app", "-a", help="Fly app name"),
) -> None:
    """Check Wayy Cloud deployment status."""
    try:
        fly = _fly_cmd()
    except FileNotFoundError as e:
        print_error(str(e))
        raise typer.Exit(1)

    _run([fly, "status", "--app", app_name])


@deploy_app.command("cloud-logs")
def cloud_logs(
    app_name: str = typer.Option(CLOUD_APP_NAME, "--app", "-a", help="Fly app name"),
) -> None:
    """Stream logs from Wayy Cloud."""
    try:
        fly = _fly_cmd()
    except FileNotFoundError as e:
        print_error(str(e))
        raise typer.Exit(1)

    try:
        _run([fly, "logs", "--app", app_name])
    except KeyboardInterrupt:
        pass


@deploy_app.command("cloud-scale")
def cloud_scale(
    app_name: str = typer.Option(CLOUD_APP_NAME, "--app", "-a", help="Fly app name"),
    count: int = typer.Argument(..., help="Number of machines"),
    vm_size: str = typer.Option("shared-cpu-2x", "--vm", help="VM size"),
    memory: str = typer.Option("1gb", "--memory", "-m", help="Memory per machine"),
) -> None:
    """Scale Wayy Cloud deployment."""
    try:
        fly = _fly_cmd()
    except FileNotFoundError as e:
        print_error(str(e))
        raise typer.Exit(1)

    _run([fly, "scale", "count", str(count), "--app", app_name])
    _run([fly, "scale", "vm", vm_size, "--app", app_name])
    _run([fly, "scale", "memory", memory, "--app", app_name])
    print_success(f"Scaled {app_name} to {count}x {vm_size} ({memory})")


@deploy_app.command("destroy")
def deploy_destroy(
    app_name: str = typer.Option(CLOUD_APP_NAME, "--app", "-a", help="Fly app name"),
) -> None:
    """Destroy the Wayy Cloud deployment."""
    try:
        fly = _fly_cmd()
    except FileNotFoundError as e:
        print_error(str(e))
        raise typer.Exit(1)

    console.print(f"[bold red]This will destroy {app_name} and all its data.[/bold red]")
    confirm = typer.confirm("Are you sure?")
    if not confirm:
        raise typer.Abort()

    _run([fly, "apps", "destroy", app_name, "--yes"])
    print_success(f"Destroyed {app_name}")


# --- Local serve ---


@deploy_app.command("local")
def deploy_local(
    port: int = typer.Option(8080, "--port", "-p", help="Port to serve on"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
    data_path: str = typer.Option("./data/wayydb", "--data-path", "-d", help="Data directory"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of uvicorn workers"),
) -> None:
    """Start WayyDB server locally with uvicorn."""
    os.makedirs(data_path, exist_ok=True)
    os.environ["WAYY_DATA_PATH"] = str(Path(data_path).resolve())
    os.environ["PORT"] = str(port)
    os.environ["CORS_ORIGINS"] = "*"

    print_info("Data path", os.environ["WAYY_DATA_PATH"])
    print_info("Serving on", f"http://{host}:{port}")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    cmd = [
        sys.executable, "-m", "uvicorn", "api.main:app",
        "--host", host,
        "--port", str(port),
        "--workers", str(workers),
    ]

    try:
        _run(cmd)
    except KeyboardInterrupt:
        console.print("\n[dim]Server stopped.[/dim]")
    except subprocess.CalledProcessError:
        print_error("Failed to start server. Is uvicorn installed? (pip install wayy-db[api])")
        raise typer.Exit(1)


# --- Docker ---


@deploy_app.command("docker")
def deploy_docker(
    port: int = typer.Option(8080, "--port", "-p", help="Host port to expose"),
    tag: str = typer.Option("wayydb:latest", "--tag", "-t", help="Docker image tag"),
    data_volume: str = typer.Option("wayydb-data", "--volume", "-v", help="Docker volume name"),
    build: bool = typer.Option(True, "--build/--no-build", help="Build image before running"),
    detach: bool = typer.Option(True, "--detach/--foreground", help="Run in background"),
) -> None:
    """Build and run WayyDB in Docker locally."""
    if not shutil.which("docker"):
        print_error("Docker not found. Install: https://docs.docker.com/get-docker/")
        raise typer.Exit(1)

    try:
        root = _find_project_root()
    except FileNotFoundError as e:
        print_error(str(e))
        raise typer.Exit(1)

    if build:
        console.print("[bold]Building Docker image...[/bold]")
        _run(["docker", "build", "-t", tag, "."], cwd=root)
        print_success(f"Built {tag}")

    _run(["docker", "volume", "create", data_volume], check=False)
    _run(["docker", "rm", "-f", "wayydb"], check=False)

    cmd = [
        "docker", "run",
        "--name", "wayydb",
        "-p", f"{port}:8080",
        "-v", f"{data_volume}:/data/wayydb",
        "-e", "CORS_ORIGINS=*",
    ]

    if detach:
        cmd.append("-d")
    cmd.append(tag)

    _run(cmd)

    if detach:
        print_success(f"WayyDB running at http://localhost:{port}")
        print_info("Container", "wayydb")
        print_info("Volume", data_volume)
        console.print("[dim]Stop with: wayy deploy stop[/dim]")


@deploy_app.command("stop")
def deploy_stop(
    name: str = typer.Option("wayydb", "--name", "-n", help="Container name"),
) -> None:
    """Stop a running WayyDB Docker container."""
    if not shutil.which("docker"):
        print_error("Docker not found")
        raise typer.Exit(1)

    _run(["docker", "stop", name], check=False)
    _run(["docker", "rm", name], check=False)
    print_success(f"Stopped {name}")


@deploy_app.command("logs")
def deploy_logs(
    name: str = typer.Option("wayydb", "--name", "-n", help="Docker container name"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    tail: int = typer.Option(100, "--tail", help="Number of lines to show"),
) -> None:
    """View logs from a local Docker container."""
    if not shutil.which("docker"):
        print_error("Docker not found")
        raise typer.Exit(1)

    cmd = ["docker", "logs", "--tail", str(tail)]
    if follow:
        cmd.append("-f")
    cmd.append(name)

    try:
        _run(cmd, check=False)
    except KeyboardInterrupt:
        pass
