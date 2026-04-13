# Configuration and prompt examples

## Codex project config example

Use this as a starting point in `.codex/config.toml` or as a snippet in project docs.

```toml
[mcp_servers.project_dev]
url = "http://127.0.0.1:8787/mcp"
enabled = false
required = false
startup_timeout_sec = 20
tool_timeout_sec = 120
```

Notes:
- Keep `enabled = false` by default if the server is optional.
- Flip it on only while developing or debugging.
- Add `enabled_tools` only after the tool surface stabilizes.

## Skill disable example

```toml
[[skills.config]]
path = "/absolute/path/to/universal-dev-mcp/SKILL.md"
enabled = false
```

## Suggested AGENTS.md snippet

```md
When the project development MCP server is enabled and healthy, use it first for test execution, log inspection, local app navigation, and screenshots. If a repeated debugging workflow is missing, extend the development MCP server with a reusable tool, then add tests and documentation. Keep the server dev-only and easy to disable.
```

## Suggested repository README note

State clearly that:
- the MCP server is for local development and debugging
- it is optional
- it is excluded from release flows by default
- compiled projects should keep it in a sidecar, separate dev target, or build-flagged path
