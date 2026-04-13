---
name: universal-dev-mcp
description: Design, scaffold, or maintain a project-local, dev-only MCP streamable HTTP server and its repo wiring so Codex can run tests, inspect logs, navigate local apps, and capture screenshots across any language or framework. Use for development and debugging workflows. Do not use for release packaging or production-only changes.
---

# Purpose

Use this skill when the user wants Codex to create, improve, or standardize a local development MCP server for a repository.

The server this skill produces must help Codex:
- run tests and targeted validation repeatedly
- inspect logs and other debugging output
- navigate a local web app when one exists
- capture screenshots or other artifacts when UI validation is needed
- evolve the tool surface over time as new debugging needs appear

This skill is intentionally cross-language and cross-framework. Do not assume Node, Python, Java, Rust, Go, .NET, or any browser stack until you inspect the repository.

# Do not use this skill when

- the request is only about application logic and does not involve developer tooling
- the user wants a production integration, release packaging, or public connector before the dev workflow exists
- the repository already has an adequate MCP server and only needs a small bug fix unrelated to its role
- the user explicitly wants zero extra tooling in the repository

# Operating rules

1. Keep the implementation dev-only, optional, and easy to disable.
2. Prefer the least invasive architecture that gives Codex reliable test and debugging access.
3. Reuse the project's existing tools instead of replacing them.
4. Expose a small, generic, structured tool surface first. Expand only when a repeated debugging need appears.
5. Keep all filesystem access rooted in the repository or its declared artifact directories.
6. Add or update tests for the MCP server itself and for any adapters it depends on.
7. Document how to start it, stop it, verify it, and disable it.
8. Avoid hidden autonomy. Any "self-improvement" means improving the development MCP tooling inside version control with explicit code changes, tests, and documentation.

# Architecture decision order

Follow this order unless the user asks for something else.

## 1. Choose sidecar vs embedded

Default to a sidecar dev server when:
- the project is in a compiled language
- production binaries should remain clean
- the repo is polyglot
- the fastest path is a small Node or Python tool folder
- browser automation is easier outside the main application runtime

Embedded implementations are acceptable when:
- the primary language already has good HTTP and MCP support
- the team clearly wants a single-process developer experience
- the implementation can still be excluded from production builds cleanly

## 2. Prefer existing capabilities

Before adding dependencies, inspect the repo for:
- test runners and package scripts
- browser tooling such as Playwright, Cypress, Selenium, Puppeteer, or equivalent
- log files, structured logs, or existing debug commands
- dev server commands
- integration or E2E test harnesses

Wrap what already exists. Do not add a second test runner or second browser stack unless necessary.

## 3. Define a minimal first tool set

Implement the smallest useful set first. Usually:
- `health`
- `project_info`
- `run_tests`
- `run_task` or `run_command` with project-root restrictions
- `read_log_excerpt`
- `list_artifacts`

If the repo has a browser UI, also implement:
- `navigate`
- `capture_screenshot`
- optionally `snapshot_dom` or `get_console_messages`

Only add lifecycle tools like `start_dev_server` or `stop_dev_server` when the workflow genuinely needs Codex to manage the app process.

# Transport and shape

Use a streamable HTTP MCP server.

Recommended defaults:
- expose the MCP transport at `/mcp`
- add a simple health route such as `/` or `/healthz`
- keep request handling stateless when practical
- return structured JSON with stable keys
- save generated artifacts to a deterministic project-local directory such as `.codex/dev-mcp/artifacts/` or another clearly documented path

# Tool design standards

For every tool:
- define a tight schema
- validate inputs
- include timeouts
- return machine-readable status fields
- include paths to logs or artifacts instead of embedding large blobs whenever practical
- never assume one package manager, one test runner, or one browser tool

Recommended return fields:
- `ok`
- `summary`
- `exit_code` when relevant
- `stdout_path`
- `stderr_path`
- `artifact_paths`
- `detected_stack`
- `next_suggested_actions`

# Continuous improvement loop

When Codex repeatedly needs a manual step that the MCP server does not expose:
1. confirm that the need is recurring and useful
2. add or refine a tool instead of hardcoding one-off instructions
3. add tests for that new capability
4. update documentation and examples
5. keep the tool generic enough to remain useful across future debugging tasks

Good examples:
- repeated need to tail a framework-specific log file
- repeated need to open a route and take a screenshot after a UI change
- repeated need to run a filtered subset of tests with standard output capture

Bad examples:
- adding a tool for one single bug with no reusable value
- adding a production deployment tool to a development-only MCP server
- creating a tool that bypasses repo safety boundaries

# Guidance for compiled projects

For compiled repos, prefer one of these patterns:

## Pattern A: Sidecar tool folder
Create a small developer tool in a separate folder such as:
- `tools/dev-mcp/`
- `devtools/mcp/`
- `.internal/dev-mcp/`

Advantages:
- production binaries stay unchanged
- faster iteration
- easy to delete or ignore in release flows

## Pattern B: Compile-time flag
If the server must live in the main codebase, guard it behind a compile-time or build-time switch and keep it off by default.

## Pattern C: Separate developer binary
Build a dedicated developer binary or target that is not part of the normal release artifact.

In all compiled-project patterns:
- keep the feature disabled by default
- do not wire it into release packaging
- document how to enable and disable it locally

# Repository outputs to create

Unless the repo already has equivalents, aim to add:

- the MCP server code
- adapter code that wraps the repo's existing test, log, and browser workflows
- tests for the MCP layer
- a short README for the server
- a project configuration snippet showing how Codex can connect to it
- an AGENTS.md snippet telling Codex to prefer the project dev MCP server when it is enabled and healthy

# Implementation workflow

1. Inspect the repo and detect its stack, test runner, and UI or non-UI nature.
2. Choose sidecar or embedded architecture with a brief justification.
3. Define the minimum initial tool catalog.
4. Implement the streamable HTTP MCP server.
5. Wire tools to existing repo commands and browser automation.
6. Add tests and at least one smoke test.
7. Add documentation for setup, run, disable, and troubleshooting.
8. Add optional Codex config examples, but avoid changing user-global config unless explicitly asked.
9. Summarize what was added, how to enable it, how to disable it, and which future tools would be the best next additions.

# Quality bar

Do not finish until the repository has:
- a clear enable path
- a clear disable path
- at least one validation path for server startup
- at least one validation path for the core tools
- documentation that explains the dev-only nature of the feature

# Final response format

When you use this skill, end with:
- architecture choice
- files created or changed
- commands to start and test the MCP server
- how to enable or disable it
- open follow-up opportunities for the next most reusable MCP capabilities
