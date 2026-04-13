# Architecture notes

This reference file exists to help the skill stay consistent across repositories.

## Primary objective

Give Codex a reliable, repeatable interface for:
- test execution
- debugging output
- UI navigation
- screenshots and other evidence artifacts

## Default recommendation

Use a sidecar streamable HTTP MCP server for most repositories.

Choose an embedded implementation only when it is clearly simpler and still easy to keep out of production artifacts.

## Universal adapter pattern

Design the server in two layers.

### Layer 1: generic MCP tool contracts
Stable tool names and input/output schemas that Codex can rely on.

### Layer 2: project adapters
Thin wrappers that translate those generic calls into project-specific commands such as:
- `npm test -- --grep ...`
- `pytest -k ...`
- `cargo test ...`
- `go test ./...`
- `dotnet test ...`
- a Playwright route navigation helper
- a framework-specific log reader

This split keeps the external interface stable while allowing the internal implementation to match the repository.

## Recommended file organization

One acceptable pattern is:

- `tools/dev-mcp/` or `devtools/mcp/`
- `server/` for the MCP transport and schemas
- `adapters/` for test, logs, browser, artifacts
- `tests/` for MCP-specific verification
- `README.md` for operator instructions

## Artifact handling

Prefer deterministic artifact directories and filenames. Good examples:
- `.codex/dev-mcp/artifacts/screenshots/`
- `.codex/dev-mcp/artifacts/logs/`
- `.codex/dev-mcp/artifacts/test-runs/`

Return artifact paths from tools instead of embedding large binary payloads.

## Browser automation

For UI repositories:
- reuse Playwright, Cypress, Selenium, Puppeteer, or the repo's existing tooling
- do not install a second browser framework without a clear reason
- prefer tools that accept URL or route, wait strategy, viewport, and output path
- make screenshots reproducible

## Safety boundaries

Keep tool execution bounded by:
- repository root
- explicit allowlists or task registries where possible
- timeouts
- documented environment variables
- no hidden production side effects

## Extensibility rule

When a missing developer workflow shows up repeatedly, add a reusable tool and test it.
Do not add one-off tools that are too narrow to justify their existence.
