# Suggested initial tool catalog

The exact set should be adjusted to the repository, but this is a good universal starting point.

## 1. `health`
Purpose:
- confirm the server is up
- expose version and capability summary

Suggested output:
- `ok`
- `server_name`
- `server_version`
- `project_root`
- `capabilities`

## 2. `project_info`
Purpose:
- detect language, framework, test runner, browser tooling, package manager, and main dev commands

Suggested output:
- `detected_stack`
- `detected_test_commands`
- `detected_ui_stack`
- `recommended_next_tools`

## 3. `run_tests`
Purpose:
- run the default test suite or a filtered subset

Suggested inputs:
- `target`
- `filter`
- `cwd_relative`
- `timeout_sec`
- `update_snapshots` if relevant and safe

Suggested output:
- `ok`
- `summary`
- `exit_code`
- `stdout_path`
- `stderr_path`
- `artifact_paths`

## 4. `run_task`
Purpose:
- execute a named repo task such as `dev`, `lint`, `typecheck`, `test:e2e`

Prefer named tasks over free-form shell when possible.

Suggested inputs:
- `task_name`
- `args`
- `cwd_relative`
- `timeout_sec`

## 5. `run_command`
Optional.
Use only when the project genuinely needs ad hoc command execution and there is a safe repo-root boundary.

## 6. `read_log_excerpt`
Purpose:
- return a bounded excerpt from a configured log source

Suggested inputs:
- `source`
- `max_lines`
- `tail`
- `contains`

## 7. `list_artifacts`
Purpose:
- enumerate screenshots, test outputs, and logs created by MCP tools

## 8. `navigate`
For UI repositories.
Purpose:
- open a route or full URL in the project browser harness

Suggested inputs:
- `url`
- `wait_for`
- `timeout_sec`
- `viewport`

## 9. `capture_screenshot`
For UI repositories.
Purpose:
- create deterministic screenshots for debugging and regression review

Suggested inputs:
- `url`
- `path`
- `full_page`
- `wait_for`
- `timeout_sec`

## 10. `snapshot_dom` or `get_console_messages`
Optional.
Useful when screenshots are insufficient for diagnosis.

# Naming advice

Use simple snake_case names. Keep them stable. Do not expose framework names in the public tool name unless the tool is intentionally framework-specific.
