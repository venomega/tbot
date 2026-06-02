# Skill Format Specification v1

A cross-chatbot skill system. Each skill is a **directory** containing a
`SKILL.md` manifest and a handler file. Any chatbot implementing this spec
can load and execute the same skills.

## Directory Structure

```
~/.config/tbot/skills/
  hello/
    SKILL.md
    run.py
  counter/
    SKILL.md
    run.py
  ...
```

## SKILL.md (manifest)

Uses YAML frontmatter (between `---` delimiters):

```yaml
---
name: "hello"                        # skill identifier (exposed as skill_<name>)
description: "Greets the user"       # tool description for the AI
schema:                              # JSON Schema for call arguments
  type: object
  properties:
    name:
      type: string
      description: Name to greet
  required: [name]
handler: run.py                      # file that contains run()
---

Markdown documentation after the frontmatter.
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | yes | Skill identifier. Exposed to AI as `skill_<name>` |
| `description` | yes | Human/AI description of what the skill does |
| `schema` | no | JSON Schema object (OpenAI function calling format). Defaults to `{"input": "string"}` |
| `handler` | no | Filename of the handler script (default: `run.py`) |

## Handler Script

A Python file that exports `run(args: dict) -> str`:

```python
def run(args):
    """Execute the skill. args is a dict matching 'schema'."""
    return "result string"
```

### Optional lifecycle hooks

```python
def setup():
    """Called once when the skill is loaded."""

def teardown():
    """Called when skills are reloaded."""
```

## Execution Contract

1. Chatbot reads `SKILL.md`, parses frontmatter, imports `handler`.
2. Calls `setup()` once on load.
3. When AI calls `skill_<name>`, calls `run(args)` with parsed JSON arguments.
4. Return value `str` is sent back to the AI as the tool result.
5. Exceptions are caught and reported as error strings.
6. Results are truncated to 8000 characters.

## Naming

- Directory name: `[a-zA-Z_][a-zA-Z0-9_]*`
- Names starting with `_` are ignored.
- Exposed as `skill_<name>` to avoid conflicts with built-in tools.

## Chatbot Commands (Recommended)

| Command | Description |
|---------|-------------|
| `/skills` | List installed skills |
| `/skill add <name>` | Create a new skill directory with template |
| `/skill rm <name>` | Delete a skill directory |
| `/skill show <name>` | Print all files in the skill directory |
