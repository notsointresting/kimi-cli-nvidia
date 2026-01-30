"""NVIDIA API integration for kimi-cli."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import SecretStr
from rich.console import Console
from rich.prompt import Prompt

from kimi_cli.config import (
    LLMModel,
    LLMProvider,
    load_config,
    save_config,
)
from kimi_cli.share import get_share_dir

cli = typer.Typer(
    help="NVIDIA API integration commands.",
    add_completion=False,
)

# NVIDIA API configuration
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "moonshotai/kimi-k2.5"
NVIDIA_PROVIDER_KEY = "nvidia"
NVIDIA_MODEL_KEY = "nvidia/kimi-k2.5"


def get_env_file() -> Path:
    """Get the .env file path in the share directory."""
    return get_share_dir() / ".env"


def load_nvidia_api_key() -> str | None:
    """Load NVIDIA API key from .env file."""
    env_file = get_env_file()
    if not env_file.exists():
        return None

    try:
        content = env_file.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                if key.strip() == "NVIDIA_API_KEY":
                    # Remove quotes if present
                    value = value.strip()
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    return value
    except Exception:
        pass
    return None


def save_nvidia_api_key(api_key: str) -> None:
    """Save NVIDIA API key to .env file."""
    env_file = get_env_file()
    lines: list[str] = []
    key_found = False

    if env_file.exists():
        try:
            content = env_file.read_text(encoding="utf-8")
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("NVIDIA_API_KEY="):
                    lines.append(f'NVIDIA_API_KEY="{api_key}"')
                    key_found = True
                else:
                    lines.append(line)
        except Exception:
            pass

    if not key_found:
        lines.append(f'NVIDIA_API_KEY="{api_key}"')

    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_nvidia_provider(api_key: str) -> None:
    """Configure NVIDIA as a provider in the kimi config."""
    config = load_config()

    # Add NVIDIA provider
    # Uses custom "nvidia" type for proper reasoning_content handling
    config.providers[NVIDIA_PROVIDER_KEY] = LLMProvider(
        type="nvidia",
        base_url=NVIDIA_BASE_URL,
        api_key=SecretStr(api_key),
    )

    # Add NVIDIA Kimi K2.5 model with thinking enabled
    config.models[NVIDIA_MODEL_KEY] = LLMModel(
        provider=NVIDIA_PROVIDER_KEY,
        model=NVIDIA_MODEL,
        max_context_size=262144,  # 256K context as per NVIDIA docs
        capabilities={"thinking", "image_in", "video_in"},
    )

    # Set as default model if no default is set
    if not config.default_model:
        config.default_model = NVIDIA_MODEL_KEY

    save_config(config)


def remove_nvidia_provider() -> None:
    """Remove NVIDIA provider from config."""
    config = load_config()

    # Remove provider
    if NVIDIA_PROVIDER_KEY in config.providers:
        del config.providers[NVIDIA_PROVIDER_KEY]

    # Remove models using this provider
    models_to_remove = [
        key for key, model in config.models.items()
        if model.provider == NVIDIA_PROVIDER_KEY
    ]
    for key in models_to_remove:
        del config.models[key]

    # Reset default model if it was using NVIDIA
    if config.default_model in models_to_remove:
        config.default_model = next(iter(config.models), "")

    save_config(config)


@cli.command()
def login(
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            "-k",
            help="NVIDIA API key. If not provided, will prompt for input.",
        ),
    ] = None,
) -> None:
    """Login to NVIDIA API with your API key.

    This configures kimi-cli to use NVIDIA's API endpoint for the Kimi K2.5 model.
    Your API key will be stored in ~/.kimi/.env file.

    Get your API key from: https://build.nvidia.com/
    """
    console = Console()

    # Check if already configured
    existing_key = load_nvidia_api_key()
    if existing_key:
        console.print(
            "[yellow]NVIDIA API is already configured.[/yellow]"
        )
        overwrite = Prompt.ask(
            "Do you want to overwrite the existing API key?",
            choices=["y", "n"],
            default="n",
        )
        if overwrite.lower() != "y":
            console.print("Login cancelled.")
            raise typer.Exit()

    # Get API key
    if api_key is None:
        console.print(
            "\n[bold]NVIDIA API Login[/bold]\n"
            "Get your API key from: [link=https://build.nvidia.com/]https://build.nvidia.com/[/link]\n"
        )
        api_key = Prompt.ask("Enter your NVIDIA API key", password=True)

    if not api_key or not api_key.strip():
        console.print("[red]Error: API key cannot be empty.[/red]")
        raise typer.Exit(code=1)

    api_key = api_key.strip()

    # Validate API key format (NVIDIA keys typically start with nvapi-)
    if not api_key.startswith("nvapi-"):
        console.print(
            "[yellow]Warning: NVIDIA API keys typically start with 'nvapi-'. "
            "Proceeding anyway...[/yellow]"
        )

    # Save API key to .env
    console.print("Saving API key to ~/.kimi/.env...")
    save_nvidia_api_key(api_key)

    # Configure provider
    console.print("Configuring NVIDIA provider...")
    configure_nvidia_provider(api_key)

    console.print(
        "\n[green]Success![/green] NVIDIA API configured.\n"
        f"  Provider: {NVIDIA_PROVIDER_KEY}\n"
        f"  Model: {NVIDIA_MODEL}\n"
        f"  Base URL: {NVIDIA_BASE_URL}\n"
        "\nYou can now use kimi with the NVIDIA API:\n"
        f"  kimi --model {NVIDIA_MODEL_KEY}\n"
    )


@cli.command()
def logout() -> None:
    """Logout from NVIDIA API and remove configuration."""
    console = Console()

    existing_key = load_nvidia_api_key()
    if not existing_key:
        console.print("[yellow]NVIDIA API is not configured.[/yellow]")
        raise typer.Exit()

    # Remove from .env
    env_file = get_env_file()
    if env_file.exists():
        try:
            content = env_file.read_text(encoding="utf-8")
            lines = [
                line for line in content.splitlines()
                if not line.strip().startswith("NVIDIA_API_KEY=")
            ]
            env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as e:
            console.print(f"[red]Error removing API key from .env: {e}[/red]")
            raise typer.Exit(code=1)

    # Remove provider from config
    remove_nvidia_provider()

    console.print("[green]Successfully logged out from NVIDIA API.[/green]")


@cli.command()
def status() -> None:
    """Check NVIDIA API configuration status."""
    console = Console()

    api_key = load_nvidia_api_key()
    config = load_config()

    has_provider = NVIDIA_PROVIDER_KEY in config.providers
    has_model = NVIDIA_MODEL_KEY in config.models

    if api_key and has_provider and has_model:
        # Mask API key for display
        masked_key = api_key[:10] + "..." + api_key[-4:] if len(api_key) > 14 else "***"
        console.print(
            "[green]NVIDIA API is configured.[/green]\n"
            f"  API Key: {masked_key}\n"
            f"  Provider: {NVIDIA_PROVIDER_KEY}\n"
            f"  Model: {NVIDIA_MODEL_KEY}\n"
            f"  Base URL: {NVIDIA_BASE_URL}\n"
        )
    elif api_key:
        console.print(
            "[yellow]API key found but provider not fully configured.[/yellow]\n"
            "Run 'kimi nvidia login' to reconfigure."
        )
    else:
        console.print(
            "[yellow]NVIDIA API is not configured.[/yellow]\n"
            "Run 'kimi nvidia login' to configure."
        )
