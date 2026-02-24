# OWASP Top 10 for Agentic AI Applications (ASI) — December 2025

**Released:** Black Hat Europe, December 10, 2025
**Source:** https://owasp.org/www-project-agentic-security/
**Identifier prefix:** ASI (Agentic Security Issue)

---

## ASI-01: Prompt Injection in Agentic Contexts
Malicious inputs manipulate autonomous agents to perform unauthorized actions, bypass safety
controls, or exfiltrate data during multi-step task execution. Unlike standard prompt injection,
agentic attacks can chain through multiple steps and tool calls.

## ASI-02: Inadequate Human Oversight
Agents taking high-impact or irreversible actions without sufficient human-in-the-loop checkpoints.
The principle of "least agency" applies: grant only the minimum autonomy required for safe, bounded tasks.

## ASI-03: Excessive Tool Access
Agents granted broader tool or API access than required for their task scope, violating least privilege.
A compromised or manipulated agent can then cause disproportionate damage.

## ASI-04: Memory and Context Manipulation
Attacks targeting agent memory systems (short-term context, long-term memory, RAG stores) to
corrupt the agent's reasoning or inject persistent malicious instructions.

## ASI-05: Multi-Agent Trust Exploitation
In multi-agent orchestration, malicious agents impersonating trusted orchestrators or sub-agents
to inject malicious instructions into the pipeline without detection.

## ASI-06: Unbounded Resource Consumption
Agents entering infinite loops, spawning excessive sub-agents, or consuming disproportionate
compute or API resources — either through attack or runaway behavior.

## ASI-07: Sensitive Data Exfiltration via Agent Actions
Agents covertly exfiltrating sensitive information through tool calls, API requests, or
generated outputs, often triggered by indirect prompt injection.

## ASI-08: Insecure Inter-Agent Communication
Lack of authentication and integrity checks between agents in multi-agent systems, allowing
message spoofing, replay attacks, and unauthorized instruction injection.

## ASI-09: Cascading Failures in Agent Pipelines
A failure or compromise in one agent propagating through an entire agentic workflow,
amplifying impact across systems that depend on the compromised agent's outputs.

## ASI-10: Goal Misalignment and Reward Hacking
Agents pursuing their objectives in unintended ways that technically satisfy the goal specification
but violate the underlying human intent, sometimes with harmful side effects.
