# Integration checklist

Use this checklist when applying the skill to a real repository.

## Discovery
- Identify language(s), framework(s), and package/build system(s)
- Detect default test commands
- Detect browser tooling and whether the project has a UI
- Detect existing log sources or debug scripts

## Architecture
- Choose sidecar vs embedded
- For compiled repos, prefer sidecar, separate dev target, or compile-time flag
- Keep the feature off by default

## Server
- Implement streamable HTTP transport
- Use `/mcp` for the MCP endpoint
- Add `/` or `/healthz` for a simple health response
- Add bounded timeouts and stable JSON responses

## Core tools
- `health`
- `project_info`
- `run_tests`
- `run_task` or safe equivalent
- `read_log_excerpt`
- `list_artifacts`

## UI tools when relevant
- `navigate`
- `capture_screenshot`
- optional DOM or console helpers

## Validation
- Startup smoke test
- Tool smoke test
- At least one failing-path test
- Documentation for local usage

## Documentation
- where the code lives
- how to start it
- how to stop it
- how to verify it
- how to disable it
- why it is dev-only

## Codex integration
- add a project-scoped `.codex/config.toml` snippet or example
- add an AGENTS.md snippet telling Codex to use the dev MCP server when it is enabled
- keep global user configuration untouched unless the user explicitly asks for it
