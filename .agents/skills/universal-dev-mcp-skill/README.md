# Universal Dev MCP skill for Codex

This bundle provides a reusable Codex skill that tells the agent how to design and implement a project-local, dev-only MCP streamable HTTP server for:

- tests
- logs
- navigation
- screenshots
- repeatable debugging workflows

## Files

- `SKILL.md` — the actual skill manifest and operating instructions
- `agents/openai.yaml` — optional metadata plus an OpenAI Docs MCP dependency
- `references/architecture.md` — architecture guidance
- `references/tool-catalog.md` — suggested generic tool surface
- `references/integration-checklist.md` — rollout checklist
- `references/config-examples.md` — Codex config and AGENTS.md examples

## Installation

Put this folder in either:
- `~/.agents/skills/universal-dev-mcp/` for personal use across repositories
- `.agents/skills/universal-dev-mcp/` inside a repository for team use

Then invoke it explicitly from Codex as `$universal-dev-mcp`.

## Intent

The skill is optimized for dev-only tooling. For compiled projects, the preferred default is a sidecar or separate developer target so the normal release binary stays unchanged.
