# Development Guide

## Running with Docker

### Start the server container
```bash
docker compose run --rm server bash
```

### Run evaluation with report generation
```bash
docker compose run --rm server python examples/incremental_evaluation.py
```

### Run tests (with coverage)
```bash
docker compose run --rm pipeline-test
```

HTML coverage report is generated at `test-reports/htmlcov/index.html`.

## Building

### Rebuild Docker image
```bash
docker compose build server
```

### Clean up dangling Docker images
```bash
docker image prune -f
```

## CI/CD

### Test the Bitbucket pipeline locally
```bash
docker compose run --rm pipeline-test
```

## Report Output

Reports are generated in `examples/report_output/`:
- `evaluation_report.html` - Interactive HTML report with ECharts
- `evaluation_report.pdf` - Static PDF report for sharing
