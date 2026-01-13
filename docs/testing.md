# Testing Guide

This project uses `pytest` for both unit and integration coverage. Integration checks can be executed against the Kalshi sandbox when credentials are available.

## Environment Variables

Integration tests read the following variables to authenticate against the sandbox API:

| Variable | Purpose |
| --- | --- |
| `KALSHI_SANDBOX_API_KEY` | Required bearer token for sandbox API access |
| `KALSHI_SANDBOX_EMAIL` | Optional helper for documenting credentials (not used directly by tests) |
| `KALSHI_SANDBOX_PASSWORD` | Optional helper for documenting credentials (not used directly by tests) |
| `KALSHI_SANDBOX_API_BASE` | Override for the sandbox API hostname (defaults to `https://demo-api.kalshi.com`) |

Export the values before invoking the test suite:

```bash
export KALSHI_SANDBOX_API_KEY="<your sandbox key>"
export KALSHI_SANDBOX_EMAIL="<your sandbox email>"
export KALSHI_SANDBOX_PASSWORD="<your sandbox password>"
# Optional if you use a non-default sandbox host
export KALSHI_SANDBOX_API_BASE="https://demo-api.kalshi.com"
```

## Running the Tests

Run the unit tests and offline checks:

```bash
pytest
```

Execute the sandbox integration checks with retries and detailed output:

```bash
pytest --live-sandbox -vv --durations=10
```

The `--live-sandbox` flag enables fixtures that authenticate with the sandbox and will skip gracefully if credentials are missing. Use `--maxfail=1` when debugging to stop on the first failure.

## Capturing Metrics and Flaky Failures

All sandbox tests log start/finish timestamps inside the test body. To capture additional runtime statistics for reporting, combine pytest's reporting flags:

```bash
pytest --live-sandbox -vv --durations=10 --maxfail=1 | tee sandbox-test.log
```

The generated log will contain duration summaries and explicit warnings when retries are exhausted. Review the log for messages marked `ERROR` to spot flaky API responses (for example, HTTP 429 rate limits). Re-run the tests if transient errors occur; the shared `sandbox_retry` fixture already spaces out up to three attempts per call.
