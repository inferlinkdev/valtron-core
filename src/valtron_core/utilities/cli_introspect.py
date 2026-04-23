#!/usr/bin/env python3
"""CLI tool to introspect any codebase for LLM API calls."""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from valtron_core.utilities.code_introspection import CodeIntrospector

app = typer.Typer(help="Introspect codebases to find LLM API calls")
console = Console()


@app.command()
def scan(
    directory: str,
    output: str = typer.Option(None, "--output", "-o", help="Output JSON file path"),
    exclude: str = typer.Option("", "--exclude", "-e", help="Comma-separated patterns to exclude"),
    show_details: bool = typer.Option(False, "--details", "-d", help="Show detailed results"),
):
    """
    Scan a directory for LLM API calls across all major providers.

    Examples:
        # Scan current directory
        python -m evaltron_core.cli_introspect scan .

        # Scan specific project with output
        python -m evaltron_core.cli_introspect scan /path/to/project -o report.json

        # Show detailed results
        python -m evaltron_core.cli_introspect scan ./src --details

        # Custom exclude patterns
        python -m evaltron_core.cli_introspect scan . -e "*/tests/*,*/docs/*"
    """
    # Convert string paths to Path objects
    directory_path = Path(directory)
    output_path = Path(output) if output else None

    # Parse exclude patterns
    if exclude:
        exclude_patterns = [p.strip() for p in exclude.split(",") if p.strip()]
    else:
        exclude_patterns = ["*/venv/*", "*/.venv/*", "*/env/*", "*/node_modules/*", "*/.git/*"]

    console.print(f"\n[bold blue]🔍 Scanning {directory_path} for LLM API calls...[/bold blue]\n")

    # Initialize introspector
    introspector = CodeIntrospector()

    # Find all LLM calls
    with console.status("[cyan]Analyzing files...", spinner="dots"):
        instances = introspector.find_llm_calls_in_directory(
            directory_path,
            exclude_patterns=exclude_patterns,
        )

    # Generate report
    report = introspector.generate_report(instances)

    # Display results
    console.print(f"[bold green]✓ Scan Complete![/bold green]\n")
    console.print(f"[yellow]Total LLM API calls found:[/yellow] {report['total_calls']}\n")

    if report['total_calls'] == 0:
        console.print("[dim]No LLM API calls detected in this codebase.[/dim]\n")
        return

    # Display by provider
    if report["by_provider"]:
        console.print("[bold magenta]Calls by Provider:[/bold magenta]")
        provider_table = Table(show_header=True, header_style="bold cyan")
        provider_table.add_column("Provider", style="cyan")
        provider_table.add_column("Count", style="yellow", justify="right")

        for provider, count in sorted(
            report["by_provider"].items(), key=lambda x: x[1], reverse=True
        ):
            provider_table.add_row(provider, str(count))

        console.print(provider_table)
        console.print()

    # Display by library
    if report["by_library"]:
        console.print("[bold magenta]Calls by Library:[/bold magenta]")
        library_table = Table(show_header=True, header_style="bold cyan")
        library_table.add_column("Library", style="cyan")
        library_table.add_column("Count", style="yellow", justify="right")

        for library, count in sorted(
            report["by_library"].items(), key=lambda x: x[1], reverse=True
        ):
            library_table.add_row(library, str(count))

        console.print(library_table)
        console.print()

    # Display by call type
    if report["by_call_type"]:
        console.print("[bold magenta]Calls by Type:[/bold magenta]")
        type_table = Table(show_header=True, header_style="bold cyan")
        type_table.add_column("Call Type", style="cyan")
        type_table.add_column("Count", style="yellow", justify="right")

        for call_type, count in sorted(
            report["by_call_type"].items(), key=lambda x: x[1], reverse=True
        ):
            type_table.add_row(call_type, str(count))

        console.print(type_table)
        console.print()

    # Display by file
    if report["by_file"]:
        console.print("[bold magenta]Calls by File:[/bold magenta]")
        file_table = Table(show_header=True, header_style="bold cyan")
        file_table.add_column("File", style="cyan", no_wrap=False)
        file_table.add_column("Count", style="yellow", justify="right")

        for file_path, count in sorted(
            report["by_file"].items(), key=lambda x: x[1], reverse=True
        )[:20]:  # Show top 20 files
            # Make path relative to scan directory
            try:
                rel_path = Path(file_path).relative_to(directory_path)
            except ValueError:
                rel_path = Path(file_path)
            file_table.add_row(str(rel_path), str(count))

        console.print(file_table)
        if len(report["by_file"]) > 20:
            console.print(f"[dim]... and {len(report['by_file']) - 20} more files[/dim]")
        console.print()

    # Display detailed instances if requested
    if show_details and instances:
        console.print("[bold magenta]Detailed Call Instances:[/bold magenta]\n")

        details_table = Table(show_header=True, header_style="bold cyan", show_lines=True)
        details_table.add_column("File", style="cyan", no_wrap=False, max_width=30)
        details_table.add_column("Line", style="yellow", justify="right")
        details_table.add_column("Provider", style="green")
        details_table.add_column("Library", style="blue")
        details_table.add_column("Function", style="magenta")
        details_table.add_column("Prompt", style="white", no_wrap=False, max_width=60)
        details_table.add_column("Prompt Confidence", style="yellow")

        for instance in sorted(instances, key=lambda x: (x.file_path, x.line_number))[:50]:
            # Shorten file path for display
            try:
                short_path = Path(instance.file_path).relative_to(directory_path)
            except ValueError:
                short_path = Path(instance.file_path)

            # Format prompt display
            if instance.extracted_prompt:
                prompt_display = instance.extracted_prompt
                if len(prompt_display) > 80:
                    prompt_display = prompt_display[:77] + "..."
            else:
                prompt_display = f"[dim]{instance.extraction_notes or 'N/A'}[/dim]"

            details_table.add_row(
                str(short_path),
                str(instance.line_number),
                instance.provider,
                instance.library,
                instance.function_name,
                prompt_display,
                instance.prompt_confidence,
            )

        console.print(details_table)
        if len(instances) > 50:
            console.print(f"\n[dim]... and {len(instances) - 50} more instances[/dim]")
        console.print()

    # Export to JSON if requested
    if output_path:
        introspector.export_to_json(instances, output_path)
        console.print(f"[green]✓ Report exported to:[/green] {output_path}\n")

    # Summary
    console.print("[bold]Summary:[/bold]")
    console.print(f"  Providers detected: {len(report['providers_used'])}")
    console.print(f"  Libraries used: {len(report['libraries_used'])}")
    console.print(f"  Files with LLM calls: {len(report['by_file'])}")
    console.print()


@app.command()
def list_providers():
    """List all supported LLM providers and libraries."""
    from valtron_core.utilities.code_introspection import LLM_CALL_PATTERNS

    console.print("\n[bold blue]Supported LLM Providers & Libraries[/bold blue]\n")

    # Group by provider
    providers = {}
    for pattern in LLM_CALL_PATTERNS:
        if pattern.provider not in providers:
            providers[pattern.provider] = set()
        providers[pattern.provider].add(pattern.library)

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Provider", style="cyan")
    table.add_column("Libraries Detected", style="yellow")
    table.add_column("Call Types", style="green")

    for provider in sorted(providers.keys()):
        libraries = sorted(providers[provider])
        call_types = set()
        for pattern in LLM_CALL_PATTERNS:
            if pattern.provider == provider:
                call_types.add(pattern.call_type)

        table.add_row(
            provider,
            ", ".join(libraries),
            ", ".join(sorted(call_types)),
        )

    console.print(table)
    console.print()


if __name__ == "__main__":
    app()
