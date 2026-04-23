"""Main entry point for the evaltron_core package."""

import typer

app = typer.Typer()


@app.command()
def main() -> None:
    """Main entry point for evaltron_core."""
    typer.echo("Valtron Core - LLM call optimization")
    typer.echo("Project initialized successfully!")


if __name__ == "__main__":
    app()
