# Security

AgentScape is a local agent application, not an OS or network sandbox.

- Shell commands run with your user account's filesystem and network privileges. Review every approval; use a disposable VM and a restricted account for untrusted tasks.
- The default `ask` policy requests approval for shell commands and file writes. `allow` removes these prompts. `deny` blocks both; read, fetch, and search remain available when enabled.
- Workspace writes use descriptor-relative directory traversal without following symlinks, and atomic file replacement. Concurrent writes to the same file still use last-writer-wins behavior. These controls do not isolate the host shell.
- Shell process groups are terminated before the tool returns, including on cancellation and timeout. Commands that deliberately detach into a new session can escape group cleanup; use OS isolation for untrusted execution.
- Workspace path checks apply to the read/write tools. Fetch scope checks URL hostnames and redirects, but does not pin DNS or restrict shell networking. An empty scope permits all HTTP(S) hosts, including local services.
- Treat prompts, model output, and retrieved content as untrusted. Keep credentials outside the workspace and avoid sensitive environment variables: child shell processes inherit the environment.
- Keep Ollama bound to a trusted local interface. Protect remote endpoints with transport encryption, authentication, and network controls.
- Configuration, run logs, evidence, workspace files, virtual environments, and common credential files are excluded from Git. Review staged files before every push; ignore rules are not a security boundary.
- CI runs unit tests and checks the pinned dependency list for known advisories weekly. A clean scan does not guarantee that software is vulnerability-free.

Report suspected vulnerabilities privately to the repository owner through GitHub before disclosing sensitive details in an issue. Never include live credentials in reports.
