# LangGraph Job Agent

A companion repo to [Ai_Agent_new](https://github.com/alumnx-ai-labs/Ai_Agent_new), teaching **LangGraph** instead of LangChain's `create_agent()`.

**Important framing:** despite the repo name, this example is not "agents talking to agents." Every node here is a deterministic step — some plain Python (duplicate check, routing rules), some a single LLM call for one specific job (extraction, scoring, rewriting). None of them reason over their own tools in a loop the way Day 1's agent did. The lesson is **LangGraph as a workflow engine**: explicit typed state, conditional branching, bounded retry loops, and a human-in-the-loop pause — not multi-agent orchestration.

## The Use Case

Recruiters submit messy job descriptions (JDs) — incomplete, duplicate, vague, or biased. Rather than let a raw JD hit the database unchecked, this pipeline cleans, scores, deduplicates, and only then publishes it — asking a human for help when it can't proceed alone.

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/alumnx-ai-labs/LangGraph_Multi_Agent.git
cd LangGraph_Multi_Agent
```

### 2. Create and activate a virtual environment
```bash
python -m venv myenv
myenv\Scripts\activate      # Windows
# source myenv/bin/activate # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get a free Gemini API key
[aistudio.google.com/apikey](https://aistudio.google.com/apikey) — sign in, **Create API key**.

### 5. Get a free MongoDB Atlas cluster
Same as `Ai_Agent_new` — see that repo's README for the full walkthrough (sign up, free M0 cluster, database user, IP allowlist, connection string). This pipeline uses a different database name (`jd_intake_db`), so it won't collide with Day 1's data even on the same cluster.

### 6. Set up `.env`
```bash
cp .env.example .env
```
Fill in:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
MONGO_URI=mongodb+srv://your_username:your_password@cluster0.xxxxx.mongodb.net/
```

## Usage

### Web UI

A minimal Flask chat UI wraps the same pipeline:
```bash
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) and paste a JD into the chat. If required fields are missing, the assistant asks for them one at a time — just reply in the input box. Under the hood, each browser conversation gets its own LangGraph `thread_id`: the first submission starts a fresh run, and every reply while a question is pending resumes that same paused thread via `Command(resume=...)`, all backed by the same `checkpoints.sqlite` the CLI uses below.

### CLI

Submit a JD:
```bash
python jd_intake_pipeline.py "Senior Backend Engineer at Acme, Bangalore, 5+ years, hybrid. Building our core platform APIs."
```

If required fields are missing, the pipeline asks for them **one at a time**, right there in the terminal:
```
[thread_id: 8f2a...]

What is the location?
>
```
Type your answer and press enter; if another field is still missing, it asks for that one next. Once every required field has an answer, it proceeds straight to duplicate checking (no re-parsing needed, since each answer is taken verbatim).

### Proving it survives being killed

The interactive loop above is the normal path — but the whole reason this uses a persistent `SqliteSaver` checkpointer (not `MemorySaver`) is that the pipeline's state doesn't live in that Python process at all; it's in `checkpoints.sqlite`, keyed by `thread_id`. To prove that:

1. Submit a JD with missing fields, get asked a question, then **close the terminal entirely** instead of answering (or `Ctrl+C`).
2. Open a **new** terminal and resume with:
   ```bash
   python jd_intake_pipeline.py --resume 8f2a... "Bangalore"
   ```
   using the `thread_id` printed in step 1. It picks up exactly where it left off - a genuinely different process, not an in-memory loop that only looks persistent.

## State

```python
class JDState(TypedDict):
    raw_jd: str
    parsed: dict                # title, company, location, exp, work_mode
    missing_fields: list
    missing_info_attempts: int
    quality_score: int
    quality_feedback: str
    rewritten_jd: str
    revision_count: int
    duplicate_of: Optional[str]
    status: str                 # published | rejected | needs_input
    job_id: Optional[str]
    log: list                   # audit trail of what each node did
```

## Nodes

| Node | Does |
|---|---|
| `parse_jd` | LLM structured extraction → `parsed`, flags `missing_fields` |
| `request_missing_info` | `interrupt()` — pauses, asks a human for one missing field, sets it directly in `parsed` |
| `check_duplicate` | Plain DB lookup (exact match on lowercased title+company) → sets `duplicate_of` |
| `score_quality` | LLM scores clarity, specificity, bias-free language → `quality_score` + feedback |
| `rewrite_jd` | Rewrites using the feedback → `rewritten_jd`, increments `revision_count` |
| `publish_job` | Writes to MongoDB → `job_id`, status `published` |
| `reject` | Terminal: duplicate found |
| `escalate` | Terminal: missing-info or quality-revision attempts exhausted, status `needs_input` |

## Edges

```
START → parse_jd
parse_jd  ──[fields missing]──→ request_missing_info (asks ONE field)
          └─[complete]────────→ check_duplicate

request_missing_info ──[fields still missing, attempts < 6]──→ request_missing_info   # ask next field
                      ├─[fields still missing, attempts ≥ 6]──→ escalate
                      └─[all fields collected]─────────────────→ check_duplicate

check_duplicate ──[duplicate]──→ reject
                └─[unique]─────→ score_quality

score_quality ──[score ≥ 75]────────────────────→ publish_job
              ├─[score < 75, revisions < 2]─────→ rewrite_jd → score_quality   # self-improvement loop
              └─[score < 75, revisions ≥ 2]─────→ escalate

publish_job → END      reject → END      escalate → END
```

Two things worth noticing: **the duplicate check and both retry caps are plain Python conditionals, not LLM decisions** — a duplicate is a fact, not a judgment call, and a circuit breaker you can't trust an LLM to enforce on itself isn't a circuit breaker. And **both loops (missing-info, quality-rewrite) land on the same `escalate` node** when they run out — this pipeline never auto-rejects a JD just because it couldn't get it polished or complete in time; it hands off to a human instead.

## Design decisions worth knowing

- **MongoDB, not a second database.** The duplicate check is a simple two-field lookup with no joins — doesn't need anything relational, and reuses the Atlas account already set up for Day 1.
- **Exact-match lowercase fields, not regex.** Day 1's job search tool crashed when an unescaped keyword hit MongoDB's `$regex`. `jobs_db.py` here compares precomputed `title_lower`/`company_lower` fields instead — safer and faster.
- **`SqliteSaver`, not `MemorySaver`.** Only a persistent checkpointer survives a process restart, which is the whole point of the pause/resume demo.
- **`escalate` is shared** by both the missing-info cap and the quality-revision cap, rather than two separate "give up" nodes — same outcome (hand off to a human), one node.
- **Missing fields are asked one at a time, not all at once.** Each answer is taken verbatim and written directly into `parsed` — no re-parsing, no loop back through `parse_jd`. Simpler, fewer LLM calls, and it reads like a normal short Q&A instead of one dense "please provide: X, Y, Z" prompt.

## Project Structure

```
LangGraph_Multi_Agent/
├── jd_intake_pipeline.py   # The graph: state, nodes, edges, CLI
├── jobs_db.py              # MongoDB: duplicate check + publish
├── app.py                  # Flask chat UI over the same graph
├── templates/
│   └── index.html          # Chat frontend (vanilla JS, no build step)
├── requirements.txt
├── .env.example
├── .gitignore
└── .env                    # Your API keys (not tracked in git)
```

## Troubleshooting

### MongoDB `ServerSelectionTimeoutError` / SSL handshake failures
Almost always a missing IP allowlist entry, not a connectivity bug — Atlas rejects the TLS handshake for non-allowlisted IPs with an error that looks like a generic SSL failure. Check **Network Access** in Atlas and add your current IP.

### Garbled characters (mojibake) in output
Fixed via `sys.stdout.reconfigure(encoding="utf-8")` at the top of `jd_intake_pipeline.py` — Gemini's replies can contain accented characters, and Windows consoles don't default to UTF-8 stdout.

### "No GOOGLE_API_KEY found" / "MONGO_URI environment variable is not set"
See `Ai_Agent_new`'s README troubleshooting section — same causes, same fixes.

### Every LLM call hangs forever (LangSmith tracing enabled)
If `LANGSMITH_TRACING=true` and requests to `ChatGoogleGenerativeAI` never return — no error, no timeout, just stuck — this is **not** a network problem. Confirmed via thread-dump (`faulthandler.dump_traceback_later`): the hang is in `platform.win32_ver()`, called synchronously by LangChain's tracer (`get_runtime_environment()` on the first traced event) to attach OS metadata to the trace. On Python 3.14 + certain Windows setups, `win32_ver()`'s WMI query never returns — this blocks the main thread before any request reaches LangSmith's servers (which, on their own, are reachable fine).

This code forces tracing off unconditionally at import time (see the comment above `os.environ["LANGCHAIN_TRACING_V2"]` near the top of `jd_intake_pipeline.py`) specifically to avoid this, regardless of what `.env` says.

To get tracing back:
1. Check whether WMI itself is the problem — from an elevated PowerShell: `Get-Service Winmgmt`. If it's not `Running`, that's likely your actual fix; restart it and retest.
2. If WMI is healthy but this still hangs, it may be specific to Python 3.14's `platform.win32_ver()` implementation — try a Python 3.11–3.13 virtualenv instead.
3. Once either is confirmed fixed, remove the `os.environ[...]` overrides in `jd_intake_pipeline.py` to let `.env`'s `LANGSMITH_TRACING` take effect again.

Quick isolated repro, no LangChain involved, to confirm this on your own machine:
```bash
python -c "import platform; print(platform.win32_ver())"
```
If that alone hangs, it's this issue.
