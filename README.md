# LangGraph Job Agent (Claude + MongoDB)

A job-description intake pipeline built with **LangGraph** and **Claude** (Anthropic). A recruiter pastes a messy job description (JD); the pipeline extracts the key fields, asks for anything missing, rejects duplicates, scores and rewrites the text if needed, and finally publishes it to MongoDB.

It is a **workflow**, not a team of agents. Every node is a fixed step: some are plain Python (duplicate check, routing rules), some make a single LLM call for one job (extraction, scoring, rewriting). LangGraph provides typed state, conditional branching, bounded retry loops, and a human-in-the-loop pause that survives a restart.

You can run it two ways: a small **web chat UI** (`app.py`) or the **command line** (`jd_intake_pipeline.py`).

## What you need

| Requirement | Notes |
|---|---|
| Python 3.10 or newer | 3.11 to 3.13 is the safest choice |
| An Anthropic API key | https://console.anthropic.com/settings/keys (API usage is billed) |
| A MongoDB Atlas cluster (free M0 tier) | Used for the duplicate check and for publishing jobs |

## Setup

### 1. Get the code

```bash
git clone https://github.com/alumnx-ai-labs/Langgraph-Job-Agent.git
cd Langgraph-Job-Agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (cmd)
.venv\Scripts\activate.bat
# macOS / Linux
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Get an Anthropic API key

1. Sign in at https://console.anthropic.com.
2. Open **Settings → API keys** and click **Create key**.
3. Copy the key (it starts with `sk-ant-`). You will not be able to see it again.

### 5. Create a free MongoDB Atlas cluster

1. Sign up at https://www.mongodb.com/cloud/atlas/register and create a new project.
2. Click **Create** a cluster and choose the free **M0** tier. Pick any region near you.
3. **Database Access** → **Add New Database User**. Choose a username and password and keep them. Use letters and digits only in the password, or you must URL-encode special characters in the connection string.
4. **Network Access** → **Add IP Address**. Add your current IP (**Add Current IP Address**). For a quick local test only, you can allow `0.0.0.0/0`.
5. **Database** → **Connect** on your cluster → **Drivers** → copy the connection string. It looks like:
   `mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/`
6. Replace `<username>` and `<password>` with the database user from step 3.

The app creates its own database (`jd_intake_db`) and collection (`jobs`) on first use. You do not need to create them.

### 6. Create your `.env` file

```bash
# macOS / Linux
cp .env.example .env
# Windows (PowerShell)
copy .env.example .env
```

Open `.env` and fill in the two required values:

```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
MONGO_URI=mongodb+srv://your_username:your_password@cluster0.xxxxx.mongodb.net/
```

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key. Never commit it. |
| `MONGO_URI` | Yes | Your MongoDB Atlas connection string. |
| `ANTHROPIC_MODEL` | No | Claude model name. Default: `claude-sonnet-5-5`. Set a smaller model to reduce cost. |

`.env` is listed in `.gitignore`, so it will not be committed.

## Run it

### Option A: Web chat UI

```bash
python app.py
```

Open http://127.0.0.1:5000 and paste a job description into the chat. If a required field is missing, the assistant asks for it one question at a time. Just type your answer in the same box.

Try this first:

```
Senior Backend Engineer at Acme, Bangalore, 5+ years, hybrid. Building our core platform APIs.
```

All five fields are present, so it goes straight to the duplicate check and scoring. Then try a post with gaps, for example `We need a backend engineer at Acme.`, and it will ask for the location, experience and work mode.

Submit the **same** JD twice to see the duplicate rejection (`Status: rejected`).

### Option B: Command line

```bash
python jd_intake_pipeline.py "Senior Backend Engineer at Acme, Bangalore, 5+ years, hybrid. Building our core platform APIs."
```

If required fields are missing, it asks one at a time in the terminal:

```
[thread_id: 8f2a...]

What is the location?
>
```

Type the answer and press Enter. When all fields are filled, it carries on automatically. At the end it prints the final status and a log of what each node did. A published job prints its `job_id`.

### Prove that it survives being killed

The pipeline state is saved in `checkpoints.sqlite`, keyed by `thread_id`, not kept in the running process.

1. Submit a JD with a missing field and, at the question, close the terminal (or press `Ctrl+C`). Note the `thread_id` that was printed.
2. Open a new terminal in the project folder, activate the environment again and run:

   ```bash
   python jd_intake_pipeline.py --resume 8f2a... "Bangalore"
   ```

   Use your own `thread_id` and your answer. It carries on from exactly where it stopped.

## How it works

```
START → parse_jd
parse_jd ──[fields missing]──→ request_missing_info
         └─[complete]────────→ check_duplicate

request_missing_info ──[more fields missing, fewer than 6 questions asked]──→ request_missing_info
                     ├─[more fields missing, 6 questions asked]────────────→ escalate
                     └─[all fields collected]──────────────────────────────→ check_duplicate

check_duplicate ──[duplicate]──→ reject
                └─[unique]─────→ score_quality

score_quality ──[score ≥ 75]──────────────────────→ publish_job
              ├─[score < 75, fewer than 2 rewrites]→ rewrite_jd → score_quality
              └─[score < 75, 2 rewrites used]──────→ escalate

publish_job → END     reject → END     escalate → END
```

| Node | Type | What it does |
|---|---|---|
| `parse_jd` | LLM | Extracts `title`, `company`, `location`, `exp`, `work_mode`; flags empty ones as missing |
| `request_missing_info` | Human in the loop | `interrupt()` pauses the graph and asks for ONE missing field |
| `check_duplicate` | Plain Python + MongoDB | Exact match on lowercased title and company |
| `score_quality` | LLM | Scores clarity, specificity and bias-free language (0 to 100) |
| `rewrite_jd` | LLM | Rewrites the JD using the feedback |
| `publish_job` | Plain Python + MongoDB | Saves the job, status `published` |
| `reject` | Terminal | Duplicate found, status `rejected` |
| `escalate` | Terminal | Needs a person (too many questions or two failed rewrites), status `needs_input` |

Constants you can change at the top of `jd_intake_pipeline.py`: `REQUIRED_FIELDS`, `QUALITY_THRESHOLD` (75), `MAX_REVISIONS` (2) and `MAX_MISSING_INFO_ATTEMPTS` (6).

### State

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

### Design decisions

- **Rules, not the LLM, decide the path.** Duplicate detection and both retry limits are plain Python conditionals. A duplicate is a fact, and a stop rule that the LLM enforces on itself is not a real limit.
- **`SqliteSaver`, not `MemorySaver`.** Only a persistent checkpointer survives a process restart.
- **`escalate` is shared** by the missing-info limit and the rewrite limit. The pipeline never silently throws a JD away because it could not be polished in time. It hands off to a person.
- **Missing fields are asked one at a time.** Each answer is written straight into `parsed` with no extra LLM call.
- **Exact-match lowercase fields, not regex,** for the duplicate check, which avoids crashes on special characters.

## Project structure

```
Langgraph-Job-Agent/
├── jd_intake_pipeline.py   The graph: state, nodes, edges, CLI
├── jobs_db.py              MongoDB: duplicate check and publish
├── app.py                  Flask chat UI over the same graph
├── templates/
│   └── index.html          Chat page (plain JavaScript, no build step)
├── requirements.txt
├── .env.example            Template for your .env
└── .env                    Your real keys (not committed)
```

`checkpoints.sqlite` is created the first time you run it. It is local run history and is not committed.

## Troubleshooting

### `ModuleNotFoundError`
The virtual environment is not active, or dependencies are not installed. Activate it and run `pip install -r requirements.txt` again.

### `authentication_error` / `invalid x-api-key`
`ANTHROPIC_API_KEY` is missing or wrong. Check `.env` (no quotes or spaces around the key) and make sure you run the command from the project folder, because `.env` is read from the current directory.

### `model: ... not found`
Your account cannot use the default model. Set `ANTHROPIC_MODEL` in `.env` to a model you have access to.

### `MONGO_URI environment variable is not set`
Add `MONGO_URI` to `.env`.

### MongoDB `ServerSelectionTimeoutError` or SSL handshake errors
Almost always a missing IP allowlist entry. Atlas rejects the connection from IP addresses it does not know, and the error looks like a generic SSL failure. In Atlas open **Network Access** and add your current IP. Also confirm the username and password in the connection string.

### Garbled characters in the terminal
The script already switches stdout to UTF-8. If your terminal still shows broken characters, use a UTF-8 terminal such as Windows Terminal.

### Every LLM call hangs forever (LangSmith tracing)
On some Windows and Python 3.14 setups, LangChain's tracer calls `platform.win32_ver()`, which can block forever on a stuck WMI query. The code therefore forces LangSmith tracing off at import time (see the comment near the top of `jd_intake_pipeline.py`). To check whether your machine is affected:

```bash
python -c "import platform; print(platform.win32_ver())"
```

If that alone hangs, restart the Windows Management Instrumentation service (`Get-Service Winmgmt` in an elevated PowerShell) or use Python 3.11 to 3.13. To re-enable tracing, remove the `os.environ[...]` overrides in `jd_intake_pipeline.py` and set `LANGSMITH_TRACING=true` in `.env`.

### "Address already in use" on port 5000
Another program uses port 5000 (on macOS, AirPlay Receiver often does). Stop it, or change the last line of `app.py` to `app.run(debug=True, port=5001)`.
