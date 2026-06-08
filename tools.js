/**
 * Tool calls del proyecto opencode (packages/opencode/src/tool/registry.ts)
 *
 * Builtin tools (ordenados según aparecen en el builtin array):
 */
export const tools = [
  {
    id: "invalid",
    description: "Do not use",
    file: "invalid.ts",
    parameters: { tool: "string", error: "string" },
  },
  {
    id: "question",
    description: "Questions to ask the user",
    file: "question.ts",
    parameters: { questions: "Question[]" },
    enabled: 'client in ["app","cli","desktop"] || flags.enableQuestionTool',
  },
  {
    id: "bash",
    description: "Shell command execution",
    file: "shell.ts",
    parameters: { command: "string", description: "string", timeout: "number?", workdir: "string?" },
  },
  {
    id: "read",
    description: "Read file or directory from local filesystem",
    file: "read.ts",
    parameters: { filePath: "string", offset: "number?", limit: "number?" },
  },
  {
    id: "glob",
    description: "Fast file pattern matching via glob",
    file: "glob.ts",
    parameters: { pattern: "string", path: "string?" },
  },
  {
    id: "grep",
    description: "Fast content search via regex",
    file: "grep.ts",
    parameters: { pattern: "string", path: "string?", include: "string?" },
  },
  {
    id: "edit",
    description: "Edit files via exact string replacement",
    file: "edit.ts",
    parameters: { filePath: "string", oldString: "string", newString: "string", replaceAll: "boolean?" },
  },
  {
    id: "write",
    description: "Write a file to the local filesystem",
    file: "write.ts",
    parameters: { filePath: "string", content: "string" },
  },
  {
    id: "task",
    description: "Launch a subagent for complex multistep tasks",
    file: "task.ts",
    parameters: { description: "string", prompt: "string", subagent_type: "string", task_id: "string?", command: "string?" },
  },
  {
    id: "webfetch",
    description: "Fetch content from a URL",
    file: "webfetch.ts",
    parameters: { url: "string", format: "'markdown'|'text'|'html'", timeout: "number?" },
  },
  {
    id: "todowrite",
    description: "Create and manage a structured task list",
    file: "todo.ts",
    parameters: { todos: "TodoItem[]" },
  },
  {
    id: "websearch",
    description: "Real-time web search",
    file: "websearch.ts",
    parameters: { query: "string", numResults: "number?", livecrawl: "'fallback'|'preferred'?", type: "'auto'|'fast'|'deep'?", contextMaxCharacters: "number?" },
    enabled: 'provider === "opencode" || flags.exa || flags.parallel',
  },
  {
    id: "skill",
    description: "Load a specialized skill",
    file: "skill.ts",
    parameters: { name: "string" },
  },
  {
    id: "apply_patch",
    description: "Apply a patch to the filesystem",
    file: "apply_patch.ts",
    parameters: { patchText: "string" },
    enabled: 'modelID.includes("gpt-") && !modelID.includes("oss") && !modelID.includes("gpt-4")',
  },
  {
    id: "lsp",
    description: "LSP operations (experimental)",
    file: "lsp.ts",
    parameters: { operation: "'goToDefinition'|'findReferences'|'hover'|'documentSymbol'|'workspaceSymbol'|'goToImplementation'|'prepareCallHierarchy'|'incomingCalls'|'outgoingCalls'", filePath: "string", line: "number", character: "number", query: "string?" },
    enabled: "flags.experimentalLspTool",
  },
  {
    id: "plan_exit",
    description: "Exit plan mode and switch to build agent",
    file: "plan.ts",
    parameters: {},
    enabled: "flags.experimentalPlanMode && flags.client === 'cli'",
  },
]

/**
 * Tools experimentales condicionales (scout):
 *   repo_clone, repo_overview  →  enabled: flags.experimentalScout
 */
