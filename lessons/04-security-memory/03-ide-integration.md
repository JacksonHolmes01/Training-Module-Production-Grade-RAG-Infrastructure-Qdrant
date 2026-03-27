# Lesson 4.3 — Using Your Security Memory in an IDE (VS Code / Cursor)

> **What you're building:** not another chatbot, but a real workflow where your IDE assistant retrieves the right security references and uses them to produce grounded, citeable fixes to actual lab files.

> **Upgraded Capabilities.** Up until now, the AI in this lab could answer cybersecurity questions, explain frameworks, and help you understand concepts. With the memory and API you built in Lessons 4.1 and 4.2, it can now do something more powerful: you can point it at real files in your project, such as a Dockerfile, a config, or an API handler, and it will review them against actual security standards and suggest specific, sourced improvements. This lesson is where you put that capability to use.

This lesson teaches a workflow that works in both VS Code and Cursor, even if you have not configured any special integrations.

---

## Learning Outcomes
By the end of this lesson, you will:

- Understand what "grounded" security review means and why it matters
- Know how to open and use the AI chat panels in VS Code and Cursor
- Use `/memory/query` to retrieve relevant security guidance for a specific file
- Feed that guidance into your IDE assistant using the included prompt library
- Know how to ask follow-up questions and get the most out of the AI
- Apply the workflow to review and propose fixes to real lab files
- Know how to keep your memory up to date as standards evolve
- Understand what "major integration" means in the context of this project

## Important Note regarding Lesson 4.2
You may be wondering where the chatbot you built in this lab fits in. In this lesson, the AI doing the review is Copilot or Cursor through your chosen IDE, not your chatbot. However, your contribution is the retrieval layer -- the `/memory/query` endpoint you built in Lesson 4.2 is what fetches the security chunks. Without that, you would just be asking Copilot generic security questions with no grounded references. Think of it as a division of labour: your API finds the right standards, and the IDE AI uses them to review the code.

If you completed the optional section in Lesson 4.2, your own chatbot can take over the entire workflow. Instead of manually running a curl command and pasting chunks into Copilot, you would just open your chatbot's chat interface at `http://localhost:7860` (only exists if completed the optional section in 4.2) and ask it to review the file directly. Behind the scenes, your `/chat` endpoint would automatically detect that the question is security-related, call `/memory/query` to fetch the relevant chunks, inject them into the prompt, and send everything to Ollama to generate a grounded response. The end result is the same, a security review backed by real standards, but your chatbot is handling every step rather than you doing it manually. This is the more production-ready version of the workflow and is what a real security tool would look like in practice.

# Chat Pipeline Observations

It is possible that when using the chat endpoint you may get sources from example.org or answers from the sample data rather than recently ingested documents

## Things to Improve

- The sources are coming from the main `LabDoc` collection (sample articles like "Supply Chain Risk"), not from `ExpandedVSCodeMemory` — the memory chunks are being injected into the prompt but the RAG sources shown are from the general collection
- The answer is vague and generic, which means the security memory chunks either aren't reaching the model effectively or the model (`llama3.2:1b`) is too small to use them well

## Notes

The source display is expected behavior — `/chat` shows sources from the main RAG retrieval, not from the memory injection. The memory chunks are injected silently into the prompt context.

Overall this is working correctly. The quality of the answer is a model size limitation — `llama3.2:1b` is very small. If you want better answers try pulling a larger model:

```bash
docker exec -i ollama ollama pull llama3.2:3b
```

Then update your `.env`:

```
OLLAMA_MODEL=llama3.2:3b
```
> **Note:** The Gradio UI at `http://localhost:7860` is for demonstrating the chat/RAG pipeline to end users. The `/memory/query` endpoint is a developer tool — it is meant to be called programmatically or from the terminal as part of the workflow shown in these lessons. To test memory retrieval directly, always use the curl commands in the terminal.


## If you want gradio to use /memory/query change gradio-ui/app.py to this:
```
import os
import httpx
import gradio as gr

API_BASE_URL = os.getenv("API_BASE_URL", "http://nginx:8088")
EDGE_API_KEY = os.getenv("EDGE_API_KEY", "")


def call_api(path: str, payload: dict):
    if not EDGE_API_KEY:
        return {"error": "EDGE_API_KEY is not set for the UI container."}
    headers = {"X-API-Key": EDGE_API_KEY}
    with httpx.Client(timeout=120) as client:
        r = client.post(f"{API_BASE_URL}{path}", json=payload, headers=headers)
        r.raise_for_status()
        return r.json()


def chat_with_followups(message, history):
    """
    Extended chat function that returns the answer and extracts
    follow-up suggestions from the API response.
    """
    data = call_api("/chat", {"message": message})
    if "error" in data:
        answer = f"Error: {data['error']}"
        return answer, [], [], []

    answer = (data.get("answer") or "").strip()
    sources = data.get("sources") or []
    followups = data.get("followups") or []

    if sources:
        answer += "\n\n---\n**Sources (retrieved from Qdrant):**\n"
        for i, s in enumerate(sources, start=1):
            title = s.get("title") or "Untitled"
            url = s.get("url") or ""
            dist = s.get("distance")
            answer += f"{i}. {title} — {url} (distance={dist})\n"

    # Pad followups to always return 3 values (empty string = hide button)
    while len(followups) < 3:
        followups.append("")

    return (
        answer,
        gr.update(value=followups[0], visible=bool(followups[0])),
        gr.update(value=followups[1], visible=bool(followups[1])),
        gr.update(value=followups[2], visible=bool(followups[2])),
    )


def memory_query_fn(query, tags_str, top_k):
    if not query.strip():
        return "Enter a query above and click Search."
    tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str.strip() else []
    payload = {"query": query, "top_k": int(top_k)}
    if tags:
        payload["tags"] = tags
    data = call_api("/memory/query", payload)
    if "error" in data:
        return f"Error: {data['error']}"
    results = data.get("results") or []
    if not results:
        return "No results returned. Check that ingestion has run and the collection is populated."
    output = f"**{len(results)} chunks retrieved from ExpandedVSCodeMemory**\n\n"
    for i, r in enumerate(results, start=1):
        score = round(r.get("score", 0), 4)
        title = r.get("title") or "Untitled"
        source = r.get("source") or ""
        tags_out = r.get("tags") or []
        text = (r.get("text") or "").strip()
        output += f"---\n**{i}. {title}** | source: `{source}` | tags: `{tags_out}` | score: `{score}`\n\n{text}\n\n"
    return output


def memory_health_fn():
    if not EDGE_API_KEY:
        return "UI misconfigured: EDGE_API_KEY is missing."
    headers = {"X-API-Key": EDGE_API_KEY}
    try:
        with httpx.Client(timeout=10) as client:
            r = client.get(f"{API_BASE_URL}/memory/health", headers=headers)
            return r.text
    except Exception as e:
        return f"Memory health check failed: {e}"


def health_text():
    if not EDGE_API_KEY:
        return "UI misconfigured: EDGE_API_KEY is missing."
    headers = {"X-API-Key": EDGE_API_KEY}
    try:
        with httpx.Client(timeout=10) as client:
            r = client.get(f"{API_BASE_URL}/health", headers=headers)
            return f"{r.status_code}: {r.text}"
    except Exception as e:
        return f"Health check failed: {e}"


with gr.Blocks(title="Lab 1 — Chat with Qdrant (RAG)") as demo:
    gr.Markdown(
        "# Lab 1 — Chat with Your Dataset (RAG)\n"
        "This UI chats with your dataset using:\n"
        "- Retrieval from Qdrant\n"
        "- A local LLM via Ollama\n"
        "- An API layer behind an authenticated NGINX proxy\n"
    )

    with gr.Tabs():

        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="Chat", height=400)
                    msg_input = gr.Textbox(
                        label="Your message",
                        placeholder="Ask a question...",
                        lines=2,
                    )
                    send_btn = gr.Button("Send", variant="primary")

                    gr.Markdown("### Suggested follow-up questions")
                    with gr.Row():
                        followup_btn_1 = gr.Button("", visible=False)
                        followup_btn_2 = gr.Button("", visible=False)
                        followup_btn_3 = gr.Button("", visible=False)

                with gr.Column(scale=1):
                    gr.Markdown("## System status")
                    health_btn = gr.Button("Refresh health")
                    health_out = gr.Textbox(label="API /health output", lines=10)
                    health_btn.click(fn=health_text, outputs=health_out)

            history_state = gr.State([])

            def handle_message(message, history):
                if not message.strip():
                    return history, "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                answer, f1, f2, f3 = chat_with_followups(message, history)
                history = history + [(message, answer)]
                return history, "", f1, f2, f3

            def use_followup(question, history):
                if not question:
                    return history, "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                answer, f1, f2, f3 = chat_with_followups(question, history)
                history = history + [(question, answer)]
                return history, "", f1, f2, f3

            send_btn.click(
                fn=handle_message,
                inputs=[msg_input, history_state],
                outputs=[chatbot, msg_input, followup_btn_1, followup_btn_2, followup_btn_3],
            ).then(lambda h: h, inputs=chatbot, outputs=history_state)

            msg_input.submit(
                fn=handle_message,
                inputs=[msg_input, history_state],
                outputs=[chatbot, msg_input, followup_btn_1, followup_btn_2, followup_btn_3],
            ).then(lambda h: h, inputs=chatbot, outputs=history_state)

            followup_btn_1.click(
                fn=use_followup,
                inputs=[followup_btn_1, history_state],
                outputs=[chatbot, msg_input, followup_btn_1, followup_btn_2, followup_btn_3],
            ).then(lambda h: h, inputs=chatbot, outputs=history_state)

            followup_btn_2.click(
                fn=use_followup,
                inputs=[followup_btn_2, history_state],
                outputs=[chatbot, msg_input, followup_btn_1, followup_btn_2, followup_btn_3],
            ).then(lambda h: h, inputs=chatbot, outputs=history_state)

            followup_btn_3.click(
                fn=use_followup,
                inputs=[followup_btn_3, history_state],
                outputs=[chatbot, msg_input, followup_btn_1, followup_btn_2, followup_btn_3],
            ).then(lambda h: h, inputs=chatbot, outputs=history_state)

        with gr.Tab("Security Memory"):
            gr.Markdown(
                "## Query Security Memory\n"
                "Search the `ExpandedVSCodeMemory` collection directly. "
                "These are the chunks injected into the prompt when you ask a security question in Chat.\n"
            )
            with gr.Row():
                with gr.Column():
                    mem_query = gr.Textbox(
                        label="Query",
                        placeholder="e.g. containers running as root",
                        lines=2,
                    )
                    mem_tags = gr.Textbox(
                        label="Tags (comma separated, optional)",
                        placeholder="e.g. cis, docker",
                    )
                    mem_topk = gr.Slider(
                        minimum=1, maximum=10, value=3, step=1, label="Top K"
                    )
                    mem_btn = gr.Button("Search Memory")
                    mem_health_btn = gr.Button("Check Memory Health")
                    mem_health_out = gr.Textbox(label="Memory health", lines=3)
                    mem_health_btn.click(fn=memory_health_fn, outputs=mem_health_out)

            mem_results = gr.Markdown(label="Results")
            mem_btn.click(
                fn=memory_query_fn,
                inputs=[mem_query, mem_tags, mem_topk],
                outputs=mem_results,
            )

demo.launch(server_name="0.0.0.0", server_port=7860)
```

---

## 1) What Does "Grounded" Mean and Why Does It Matter?

Before jumping into the workflow, it is worth understanding what we are trying to achieve here.

When you ask an AI assistant "is this Dockerfile secure?", it will give you an answer, but that answer is based entirely on its training data. It might be outdated, vague, or confidently wrong. There is no way to know which specific standard it is drawing from, and you cannot audit it.

**Grounded** means the AI's answer is tied to a specific, retrievable source. Instead of "the AI said so," you get "the AI said so, and here is the CIS Docker Benchmark section it pulled from." That is the difference between a guess and a citation.

This matters in security work because security standards are versioned and change over time, you want to know which version you are working from. In a real organization, "we followed CIS Benchmark v1.6 section 4.1" is auditable; "the AI said it was fine" is not. It also trains you to think in terms of controls and frameworks, not just instinct.

Everything you built in Lessons 4.1 and 4.2 was in service of making this possible.

---

## 2) Opening the AI Chat Panel in Your IDE

You are already familiar with VS Code as a code editor. What you may not have used yet is its built-in AI chat panel. Both VS Code and Cursor have one, and they work similarly, you type a message, the AI responds, and you can have a back-and-forth conversation while your code files are open alongside it.

### In VS Code

VS Code's AI features are powered by **GitHub Copilot**. If your course account has Copilot enabled, here is how to open the chat panel:

1. Open VS Code and make sure you are signed into your GitHub account (bottom left corner; click the account icon if you are not signed in)
2. Open the chat panel in one of these ways:
   - Press `Ctrl+Alt+I` (Windows/Linux) or `Cmd+Option+I` (Mac)
   - Click the chat bubble icon in the left sidebar
   - Go to **View > Chat** in the top menu
3. The chat panel will appear on the right side of your screen. You will see a text box at the bottom where you type your messages.

You can also open **inline chat** directly inside a file by pressing `Ctrl+I` (Windows/Linux) or `Cmd+I` (Mac) while your cursor is in the editor. This is useful for asking about a specific line or block of code without switching to the side panel.

One useful VS Code Copilot feature: you can type `@workspace` at the start of a message to tell Copilot to consider all the files in your project, not just the one you have open. For example: `@workspace review docker-compose.yml for security issues`. This gives the AI more context about your project structure.

### In Cursor

Cursor is a fork of VS Code that has AI built in more deeply. The interface looks almost identical to VS Code, the same file explorer, the same editor layout, but with a more capable AI panel.

1. Open Cursor and make sure your project folder is open (`File > Open Folder`)
2. Open the AI chat panel in one of these ways:
   - Press `Ctrl+L` (Windows/Linux) or `Cmd+L` (Mac) to open the chat sidebar
   - Press `Ctrl+K` (Windows/Linux) or `Cmd+K` (Mac) to open inline chat directly in the editor
3. The chat panel appears on the right. You will see your conversation history at the top and a text input at the bottom.

Cursor has a feature called **Add to context**, you can drag a file from the file explorer directly into the chat box, or type `@` followed by a filename to reference it. For example, typing `@docker-compose.yml` in the chat will attach that file's contents so the AI can read it directly. This is more reliable than copy-pasting large files.

### Tips for getting good responses from either IDE

These tips apply whether you are in VS Code or Cursor:

**Be specific about what you want.** "Review this file" is vague. "Review this file for secrets in environment variables, exposed ports, and missing resource limits" gives the AI a clear checklist to work through.

**Paste context before asking.** If you have retrieved security chunks from memory (covered in the next section), paste them into the chat before your question. Tell the AI explicitly: "Use only the references below when identifying issues." This keeps answers grounded.

**Ask follow-up questions.** The AI chat is a conversation, not a one-shot query. If a finding is unclear, ask "why does that matter?" or "can you show me what the fix would look like as a diff?" If you disagree with something, push back: "are you sure that is required by CIS, or is that your own inference?" This is how you distinguish grounded answers from hallucination.

**Ask it to explain its reasoning.** If the AI proposes a fix, ask "which specific control does this address?" This forces it to connect the fix back to a standard, and if it cannot, that is a signal the suggestion is not grounded.

**Use the file context features.** In Cursor, use `@filename` to attach files. In VS Code, use `@workspace` for project-wide context. These prevent the AI from making assumptions about code it has not actually read.

---

## 3) The Core Workflow

This is the workflow you will use throughout this lesson. It has two steps: retrieve context from your memory API, then use it in your IDE chat.

### Step 1, Retrieve Security Context from Memory

Open the integrated terminal in your IDE. In both VS Code and Cursor, you can open it with `Ctrl+`` ` (backtick) or by going to **Terminal > New Terminal** in the menu. 

Before running any commands, make sure you are in the repo root folder. The repo root is the top-level folder of the project, the one that contains files like `docker-compose.yml` and `.env`. If you opened the project correctly in your IDE (`File > Open Folder` and selected the repo folder), your terminal will already start there. You can confirm by running:

```bash
pwd
```

The output should end with your repo folder name. If it does not, navigate there with:

```bash
cd /path/to/your/repo
```

Replace `/path/to/your/repo` with the actual path where you cloned the project. If you are not sure where that is, right-click the repo folder in your IDE file explorer and look for "Copy Path" or "Reveal in Finder/Explorer" to find it.


First, load your API key:

```bash
EDGE_API_KEY=$(grep -E '^EDGE_API_KEY=' .env | cut -d= -f2-)
```

Then query the memory API for the topic you want to review. Here is an example for reviewing a `docker-compose.yml`:

```bash
curl -sS -X POST http://localhost:8088/memory/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $EDGE_API_KEY" \
  -d '{
    "query": "docker compose security best practices secrets ports privileged mounts",
    "tags": ["docker", "cis"],
    "top_k": 8
  }' > /tmp/security_context.json
```

Breaking this down:

The `query` field is a natural language description of what you are looking for. Be specific: mention the kinds of issues you are concerned about (secrets, exposed ports, privileged containers, etc.). The more specific your query, the more relevant the chunks you will get back.

*Since this is a query related to docker-compose the query relates to docker concepts. If you want different information, use a different query*

`tags` narrows the search to specific frameworks. `"docker"` and `"cis"` will limit results to your Docker and CIS Benchmark documents. Leave `tags` out entirely if you want to search across everything.

`top_k: 8` means return the 8 most relevant chunks. For a thorough review, 6 to 10 is a good range.

`> /tmp/security_context.json` saves the results to a temporary file, meaning you do not have to copy-paste the output, you can read from the file or attach it in the next step.

To preview what came back, run:

```bash
cat /tmp/security_context.json | python -m json.tool
```

Look at the `text` fields in each result, these are the actual security reference chunks your AI will use. If they look irrelevant, adjust your `query` to be more specific and run it again.

### Step 2 -- Use the Context in Your IDE Chat

First, open the file you want to review. In VS Code or Cursor, look at the file explorer panel on the left side of the screen. Find `docker-compose.yml` in the list and click it once. It will open in the editor in the center of the screen.

Next, open the AI chat panel. If you are in VS Code, press `Ctrl+Alt+I` (Windows/Linux) or `Cmd+Option+I` (Mac). If you are in Cursor, press `Ctrl+L` (Windows/Linux) or `Cmd+L` (Mac). The chat panel will appear on the right side of your screen with a text box at the bottom where you type.

Now you need to give the AI two things in one message: the security reference chunks you retrieved, and the file you want reviewed. In the chat text box, paste this prompt first:
```
I am going to give you a set of security reference chunks retrieved from a vector database of security standards. After the references, I will share a file for you to review.

Your job:
- Identify security issues in the file
- For each issue, cite which reference it comes from
- Propose a minimal fix that keeps the lab functional
- If a finding is not supported by the references provided, say so explicitly -- do not invent citations

References:
[paste your retrieved chunks here]

File to review:
[paste the contents of docker-compose.yml here]
```

To get the chunk text, go back to your terminal and run:
```bash
cat /tmp/security_context.json | python -m json.tool
```

To get chunk text related to the query without associated scores and other information run:
```bash
cat /tmp/security_context.json | python -m json.tool | python3 -c "
import json, sys
data = json.load(open('/tmp/security_context.json'))
for r in data['results']:
    print(r['text'])
    print('---')
"
```

In the output, find each result block and copy the text inside the `"text": "..."` field. Paste all of those text values one after another where it says `[paste your retrieved chunks here]`.

Then open `docker-compose.yml` in the editor, select all the text with `Ctrl+A` (Windows/Linux) or `Cmd+A` (Mac), copy it, and paste it where it says `[paste the contents of docker-compose.yml here]`.

Once you have both in the chat, hit enter and the AI will review the file against the security references.

**Why "say so explicitly: do not invent citations"?** Without this instruction the AI will happily cite frameworks it was not given, or make up section numbers. Explicitly telling it to flag unsupported claims forces honesty and helps you separate grounded findings from general suggestions.

Once the AI responds, do not just accept the output. Use the follow-up techniques from Section 2 to dig in, ask why a finding matters, ask for the fix as a diff, or challenge anything you are unsure about.

---

## 4) The Prompt Library

Rather than constructing prompts from scratch every time, this repo includes ready-made prompt templates for the most common review scenarios. They live at:

```
security-memory/prompts/
```

Each template is a `.md` file you open in VS Code or Cursor, read the instructions at the top, then copy the prompt body into your IDE chat along with your retrieved memory chunks.

### What each template covers

**`01-dockerfile-review.md`**

Use this when reviewing a `Dockerfile`. The template instructs the AI to check for insecure base images (for example, using `latest` tags or unverified images), processes running as the `root` user inside the container, secrets or credentials hardcoded in `ENV` or `ARG` instructions, missing `HEALTHCHECK` directives, and unnecessary packages or capabilities being installed. Findings are mapped to CIS Docker Benchmark and OWASP controls.

To use it: run a memory query with `tags: ["docker", "cis"]`, open `01-dockerfile-review.md` in your IDE, copy the prompt body, paste it into your IDE chat, then follow it with your retrieved chunks and the contents of your `Dockerfile`.

**`02-compose-review.md`**

Use this when reviewing a `docker-compose.yml`. The template instructs the AI to check for services exposing ports directly to the host that should stay internal, containers running in `privileged` mode, missing resource limits (`mem_limit`, `cpu_shares`), insecure volume mounts that expose sensitive host paths, and secrets passed as plain environment variables instead of using Docker secrets.

To use it: run a memory query with `tags: ["docker", "cis"]`, open `02-compose-review.md`, copy the prompt, paste it into your IDE chat, then follow it with your retrieved chunks and your `docker-compose.yml` contents.

**`03-nginx-review.md`**

Use this when reviewing an NGINX configuration file. The template instructs the AI to check for missing HTTP security headers (`X-Frame-Options`, `Content-Security-Policy`, `Strict-Transport-Security`, etc.), weak or missing TLS configuration, missing rate limiting on public endpoints, and overly permissive CORS settings.

To use it: run a memory query with `tags: ["owasp"]` or without tags to search broadly, open `03-nginx-review.md`, copy the prompt, paste it with your retrieved chunks and NGINX config.

**`04-api-auth-review.md`**

Use this when reviewing API authentication and authorization code or configuration. The template instructs the AI to check against the OWASP API Security Top 10, specifically broken object-level authorization, broken authentication, excessive data exposure, lack of rate limiting, and missing function-level authorization.

To use it: run a memory query with `tags: ["owasp"]`, open `04-api-auth-review.md`, and paste the relevant auth-related code or config alongside your retrieved chunks.

**`05-dependency-risk-review.md`**

Use this when reviewing dependency files like `requirements.txt` or `package.json`. The template instructs the AI to flag packages with known CVEs, packages pinned to a version with a known vulnerability, unpinned packages (using `*` or no version constraint), and packages that have been abandoned or transferred to unknown maintainers.

To use it: run a memory query without tags or with `tags: ["owasp"]` for broader coverage, open `05-dependency-risk-review.md`, and paste your dependency file alongside the retrieved chunks.

### Step-by-step for any template

1. Decide which file you want to review and open it in your IDE
2. Run the appropriate memory query in your terminal (example queries are at the top of each template file)
3. Preview the results with `cat /tmp/security_context.json | python -m json.tool` to confirm the chunks look relevant
4. Open the prompt template from `security-memory/prompts/` in VS Code by clicking it in the file explorer
5. Read the instructions at the top of the template, then copy the prompt body
6. Open your IDE chat panel (see Section 2), paste the prompt, then paste or attach your memory chunks and the file being reviewed
7. Read through the AI's findings carefully, do not implement everything without thinking. Ask follow-up questions for anything unclear
8. Implement the fixes one at a time
9. After each fix, re-run the lab with `docker compose up` and confirm it still works before moving to the next fix

Step 9 is not optional: security hardening that breaks the lab is not a success. Fixing one issue at a time lets you pinpoint which change caused a problem if something stops working.

### Getting more out of the AI with follow-up questions

After the AI gives its initial review, these follow-ups tend to be useful:

"Which CIS Benchmark section specifically says that?", forces the AI to be precise about its citation, or to admit it does not have one.

"Show me that fix as a unified diff", instead of a prose description, you get the exact lines to change, which is easier to implement correctly.

"If I make that change, will anything else in the lab break?", prompts the AI to think through dependencies before you commit to a fix.

"Are there any fixes you recommended that are nice-to-have versus actually required by the standard?", helps you prioritize.

"Rank these findings by severity", useful when there are many findings and you do not have time to fix everything at once.

---

## 5) What "Major Integration" Means for This Project

You may have heard this project described as a "major integration." Here is what that actually means in practice, so you understand what you are building toward and why each lesson matters.

A major integration in this context means the repo has four things working together:

**A working vector database memory**: Qdrant is running, the `ExpandedVSCodeMemory` collection exists, and it contains real security reference documents (NIST, CIS, MITRE, OWASP, etc.). This is what you built in Lesson 4.1.

**A stable, queryable API tool**: the `/memory/query` endpoint is live, authenticated, and returns structured results that can be fed directly into prompts. This is what you built in Lesson 4.2.

**Instructions and prompts that make it useful**: without a clear workflow and prompt library, the memory is just a database nobody uses. The prompts and workflow in this lesson are what turn it into a practical security tool.

**Optionally, IDE or tool integration**: the most advanced step is configuring your IDE to call the memory API automatically, so retrieval happens without any manual steps. This is covered in Section 6 and is a nice-to-have, not a requirement.

You are not being asked to invent new AI or build something from scratch. You are being asked to connect the pieces you have already built into something genuinely useful for security work, and to practice using it on real lab files.

---

## 6) Optional: MCP / Automatic Tool Integration (Advanced)

Right now the workflow requires you to manually run a curl command and paste the results into your IDE. That works, but it adds friction. The next level is configuring your IDE to call `/memory/query` automatically whenever you ask a security question, no manual retrieval step needed.

This is done through **MCP (Model Context Protocol)**, a standard that Cursor supports for connecting external tools to the AI assistant. When configured, the IDE calls your memory API as a tool in the background, retrieves the relevant chunks, and injects them into the prompt without any manual steps.

The endpoints you already built are exactly the right shape for this: `/memory/query` is the retrieval tool and `/chat` is the RAG generation tool.

If you want to explore this, the repo includes optional notes and configuration examples under:

```
security-memory/mcp/
```

This is not required to complete the lesson. The manual workflow in Section 3 achieves the same learning goal, MCP just removes the friction once you are comfortable with the concepts.

---

## 7) Keeping Your Memory Up to Date

Security standards evolve. CIS Benchmarks get new versions, OWASP updates its Top 10, and MITRE ATT&CK adds new techniques. If you never update your memory, it will gradually go stale and start producing outdated guidance.

### How to add or update documents

1. Add or replace `.md` or `.txt` files under `security-memory/data/` in the appropriate subfolder (e.g., `security-memory/data/cis/` for a new CIS document)
2. Re-run ingestion:

```bash
docker exec -i ingestion-api python -m app.security_memory.ingest
```

Ingestion uses **upserts**, which means it will update existing chunks if the content has changed and add new ones without creating duplicates. You can re-run this safely as many times as needed.

### When do you need to create a new collection entirely?

Most updates are safe to ingest into the existing `ExpandedVSCodeMemory` collection. However, if you change any of the following, you must create a new collection:

**The embedding model**: different models produce vectors in different spaces, so old and new vectors are not comparable. Mixing them in one collection will produce nonsense search results.

**The embedding dimension**: if the new model outputs vectors of a different length, Qdrant will reject them.

**The distance metric**: if you switch from cosine to dot product or Euclidean distance, existing scores become meaningless.

If you do need a new collection, create it with a versioned name (e.g., `ExpandedVSCodeMemory_v2`) and update the `SECURITY_COLLECTION` variable in your `.env` file. This keeps the old collection intact in case you need to roll back.

---

## 8) What You Are Actually Learning Here

It is worth stepping back and connecting this lesson to the broader skills you are developing, because the workflow can feel mechanical without that context.

When you run a memory query, pick a prompt template, and ask your IDE to review a `docker-compose.yml`, you are practicing things that matter in real security work.

**Retrieval grounding**: you are learning to always anchor security claims to a specific source. This is how professional security assessments work: every finding cites a control, a benchmark, or a standard.

**Controls frameworks**: NIST, CIS, OWASP, and MITRE are not just acronyms. They are the shared vocabulary that security teams use to communicate, prioritize, and audit. Working with them in a hands-on context is much more effective than reading about them in a textbook.

**Secure configuration review**: reviewing a `Dockerfile` or `nginx.conf` against a benchmark is a real skill used in penetration testing, cloud security audits, and DevSecOps pipelines. You are practicing it in a low-stakes environment where you can make mistakes and see the results.

**Change management**: the requirement to re-run the lab after every fix is not bureaucracy. It is teaching you that security changes have to be validated. Hardening that breaks functionality is not hardening, it is an outage.

---

## Checkpoint
You are done when you can:

- Open the AI chat panel in VS Code or Cursor
- Retrieve relevant security standards via `/memory/query` for a specific file
- Use the prompt library to produce a grounded security review in your IDE
- Implement at least one fix and confirm the lab still runs correctly after the change


---
*License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) — Jackson Holmes*
