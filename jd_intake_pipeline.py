"""
JD Intake Pipeline — a LangGraph workflow, not a multi-agent system.

Every node below is a deterministic step: some are plain Python (duplicate check,
routing decisions), some make a single LLM call for one specific job (extraction,
scoring, rewriting). None of them reason over their own tools in a loop the way
Day 1's `create_agent()` agent did — that's the point of this example. LangGraph
here is used to orchestrate an explicit, typed, resumable *workflow*, with loops
that have circuit breakers and a human-in-the-loop pause for missing information.
"""

import json
import os
import sys
import uuid
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from jobs_db import find_duplicate_job, publish_job_in_db

# LLM replies can contain arbitrary Unicode (e.g. accented characters); on
# Windows consoles the default stdout encoding isn't UTF-8, so those characters
# print as mojibake without this.
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(override=True)

# LangSmith tracing hangs indefinitely on this stack - confirmed via thread-dump,
# it's NOT a network issue (smith.langchain.com is reachable fine). The tracer's
# on_chat_model_start() calls langchain_core.env.get_runtime_environment(), which
# calls platform.win32_ver(); on Python 3.14 + this Windows setup that makes a WMI
# query that never returns. Any traced LLM call hangs forever, right there in the
# main thread, before a single byte reaches LangSmith. See README Troubleshooting
# for how to check whether this affects you and how to re-enable tracing.
# LANGSMITH_TRACING=false in .env isn't enough on its own - some tracing paths key
# off LANGSMITH_API_KEY just being *present*. Force both off here so
# llm.invoke() never takes that path.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ.pop("LANGSMITH_API_KEY", None)
os.environ.pop("LANGCHAIN_API_KEY", None)

# Reads ANTHROPIC_API_KEY from the environment (.env). Override the model with
# ANTHROPIC_MODEL in .env, e.g. a smaller/cheaper Claude model.
llm = ChatAnthropic(
    model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
    max_tokens=2048,
)

REQUIRED_FIELDS = ["title", "company", "location", "exp", "work_mode"]
QUALITY_THRESHOLD = 75
MAX_REVISIONS = 2
MAX_MISSING_INFO_ATTEMPTS = 6  # fields are asked one at a time, so this is a total-question cap, not a round cap


class JDState(TypedDict):
    raw_jd: str
    parsed: dict
    missing_fields: list
    missing_info_attempts: int
    quality_score: int
    quality_feedback: str
    rewritten_jd: str
    revision_count: int
    duplicate_of: Optional[str]
    status: str
    job_id: Optional[str]
    log: list


def extract_text(content) -> str:
    """Chat models may return message content as a list of blocks instead of a plain string."""
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text")
    return content


def _strip_code_fence(text: str) -> str:
    """LLMs sometimes wrap JSON replies in ```json ... ``` fences even when told not to."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


def _log(state: JDState, note: str) -> list:
    return state.get("log", []) + [note]


# ─── Nodes ────────────────────────────────────────────────────────────────────

def parse_jd(state: JDState) -> dict:
    """LLM structured extraction: pulls title/company/location/exp/work_mode out of the raw text."""
    prompt = (
        "Extract these fields from the job description below as strict JSON, "
        f"with keys exactly {REQUIRED_FIELDS}. Use an empty string for any field "
        "that isn't clearly mentioned. Reply with JSON only, no commentary, no markdown fences.\n\n"
        f"Job description:\n{state['raw_jd']}"
    )
    response = llm.invoke(prompt)
    text = _strip_code_fence(extract_text(response.content))
    try:
        raw_parsed = json.loads(text)
    except json.JSONDecodeError:
        raw_parsed = {}

    parsed = {field: str(raw_parsed.get(field, "")).strip() for field in REQUIRED_FIELDS}
    missing = [field for field in REQUIRED_FIELDS if not parsed[field]]

    return {
        "parsed": parsed,
        "missing_fields": missing,
        "log": _log(state, f"parse_jd: parsed={parsed}, missing={missing}"),
    }


def route_after_parse(state: JDState) -> str:
    """Plain Python, not an LLM decision: are required fields present at all?"""
    return "request_missing_info" if state["missing_fields"] else "check_duplicate"


def request_missing_info(state: JDState) -> dict:
    """Pauses the graph (interrupt) and asks a human for ONE missing field at a time.

    The reply is taken verbatim as that field's value - no re-parsing needed, so
    this doesn't loop back through `parse_jd` at all. No side effects happen before
    `interrupt()`: on resume, the node re-runs from the top, so anything before the
    interrupt call would otherwise execute twice.
    """
    field = state["missing_fields"][0]
    reply = interrupt({"field": field, "question": f"What is the {field}?"})

    parsed = {**state["parsed"], field: str(reply).strip()}
    return {
        "parsed": parsed,
        "missing_fields": state["missing_fields"][1:],
        "missing_info_attempts": state.get("missing_info_attempts", 0) + 1,
        "log": _log(state, f"request_missing_info: got '{field}' from the recruiter"),
    }


def route_after_request_missing_info(state: JDState) -> str:
    """Plain Python: still fields left to ask, or have we already asked too many times?"""
    if state["missing_fields"]:
        if state.get("missing_info_attempts", 0) >= MAX_MISSING_INFO_ATTEMPTS:
            return "escalate"
        return "request_missing_info"
    return "check_duplicate"


def check_duplicate(state: JDState) -> dict:
    """Plain database lookup — no LLM. A duplicate is a fact, not a judgment call."""
    existing = find_duplicate_job(state["parsed"]["title"], state["parsed"]["company"])
    duplicate_of = str(existing["_id"]) if existing else None
    return {
        "duplicate_of": duplicate_of,
        "log": _log(state, f"check_duplicate: duplicate_of={duplicate_of}"),
    }


def route_after_duplicate_check(state: JDState) -> str:
    return "reject" if state["duplicate_of"] else "score_quality"


def score_quality(state: JDState) -> dict:
    """LLM scores clarity, specificity, and bias-free language of the current draft."""
    jd_text = state.get("rewritten_jd") or state["raw_jd"]
    prompt = (
        "Score this job description from 0-100 on clarity, specificity, and "
        "bias-free language. Reply as strict JSON, no markdown fences: "
        '{"score": <int 0-100>, "feedback": "<1-2 sentence critique>"}.\n\n'
        f"Job description:\n{jd_text}"
    )
    response = llm.invoke(prompt)
    text = _strip_code_fence(extract_text(response.content))
    try:
        result = json.loads(text)
        score = int(result.get("score", 0))
        feedback = str(result.get("feedback", ""))
    except (json.JSONDecodeError, ValueError, TypeError):
        score, feedback = 0, "Could not parse a quality score from the model's reply."

    return {
        "quality_score": score,
        "quality_feedback": feedback,
        "log": _log(state, f"score_quality: {score}/100 - {feedback}"),
    }


def route_after_score(state: JDState) -> str:
    """Plain Python: the threshold and revision cap are fixed rules, not LLM discretion."""
    if state["quality_score"] >= QUALITY_THRESHOLD:
        return "publish_job"
    if state.get("revision_count", 0) < MAX_REVISIONS:
        return "rewrite_jd"
    return "escalate"


def rewrite_jd(state: JDState) -> dict:
    """LLM rewrites the current draft using the quality feedback."""
    jd_text = state.get("rewritten_jd") or state["raw_jd"]
    prompt = (
        f"Rewrite this job description to address the following feedback: "
        f"\"{state['quality_feedback']}\". Reply with only the rewritten job "
        "description, no commentary, no markdown fences.\n\n"
        f"Job description:\n{jd_text}"
    )
    response = llm.invoke(prompt)
    rewritten = extract_text(response.content).strip()
    revision_count = state.get("revision_count", 0) + 1
    return {
        "rewritten_jd": rewritten,
        "revision_count": revision_count,
        "log": _log(state, f"rewrite_jd: revision #{revision_count}"),
    }


def publish_job(state: JDState) -> dict:
    """Terminal: writes the final job posting to MongoDB."""
    description = state.get("rewritten_jd") or state["raw_jd"]
    job_id = publish_job_in_db(state["parsed"], description, state["quality_score"])
    return {
        "job_id": job_id,
        "status": "published",
        "log": _log(state, f"publish_job: job_id={job_id}"),
    }


def reject(state: JDState) -> dict:
    """Terminal: duplicate postings are auto-rejected, no human judgment needed."""
    return {
        "status": "rejected",
        "log": _log(state, "reject: duplicate of an existing job posting"),
    }


def escalate(state: JDState) -> dict:
    """Terminal: too many missing-info questions asked, or too many failed quality revisions.

    Either way this pipeline gives up and hands off to a human outside the graph —
    it does not auto-reject a JD just because the LLM couldn't polish it in time.
    """
    return {
        "status": "needs_input",
        "log": _log(state, "escalate: needs manual review"),
    }


# ─── Graph assembly ───────────────────────────────────────────────────────────

def build_graph(checkpointer):
    graph = StateGraph(JDState)

    graph.add_node("parse_jd", parse_jd)
    graph.add_node("request_missing_info", request_missing_info)
    graph.add_node("check_duplicate", check_duplicate)
    graph.add_node("score_quality", score_quality)
    graph.add_node("rewrite_jd", rewrite_jd)
    graph.add_node("publish_job", publish_job)
    graph.add_node("reject", reject)
    graph.add_node("escalate", escalate)

    graph.add_edge(START, "parse_jd")
    graph.add_conditional_edges("parse_jd", route_after_parse, {
        "request_missing_info": "request_missing_info",
        "escalate": "escalate",
        "check_duplicate": "check_duplicate",
    })
    graph.add_conditional_edges("request_missing_info", route_after_request_missing_info, {
        "request_missing_info": "request_missing_info",
        "escalate": "escalate",
        "check_duplicate": "check_duplicate",
    })
    graph.add_conditional_edges("check_duplicate", route_after_duplicate_check, {
        "reject": "reject",
        "score_quality": "score_quality",
    })
    graph.add_conditional_edges("score_quality", route_after_score, {
        "publish_job": "publish_job",
        "rewrite_jd": "rewrite_jd",
        "escalate": "escalate",
    })
    graph.add_edge("rewrite_jd", "score_quality")
    graph.add_edge("publish_job", END)
    graph.add_edge("reject", END)
    graph.add_edge("escalate", END)

    return graph.compile(checkpointer=checkpointer)


# ─── CLI runner ────────────────────────────────────────────────────────────────
#
# Default mode is interactive: pauses just prompt for input right there in the
# same process, like any normal CLI tool. `--resume` is kept as a separate,
# explicit mode for the one demo that actually needs two process invocations:
# proving the pipeline survives being killed and resumed later, purely from
# checkpoints.sqlite plus a thread_id. Day-to-day use should never need it.

def report_final(result: dict) -> None:
    print(f"\nStatus: {result['status']}")
    for line in result["log"]:
        print(f"  - {line}")
    if result["status"] == "published":
        print(f"\nPublished job_id: {result['job_id']}")


def run_interactive(app, raw_jd: str, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    print(f"[thread_id: {thread_id}]")
    result = app.invoke({"raw_jd": raw_jd}, config=config)

    while "__interrupt__" in result:
        question = result["__interrupt__"][0].value["question"]
        reply = input(f"\n{question}\n> ")
        result = app.invoke(Command(resume=reply), config=config)

    report_final(result)


if __name__ == "__main__":
    with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
        app = build_graph(checkpointer)

        if len(sys.argv) >= 2 and sys.argv[1] == "--resume":
            if len(sys.argv) < 4:
                print('Usage: python jd_intake_pipeline.py --resume <thread_id> "<your answer>"')
                sys.exit(1)
            thread_id = sys.argv[2]
            reply = " ".join(sys.argv[3:])
            config = {"configurable": {"thread_id": thread_id}}
            result = app.invoke(Command(resume=reply), config=config)

            while "__interrupt__" in result:
                question = result["__interrupt__"][0].value["question"]
                reply = input(f"\n{question}\n> ")
                result = app.invoke(Command(resume=reply), config=config)

            report_final(result)

        elif len(sys.argv) >= 2:
            thread_id = str(uuid.uuid4())
            raw_jd_input = " ".join(sys.argv[1:])
            run_interactive(app, raw_jd_input, thread_id)

        else:
            print('Usage: python jd_intake_pipeline.py "<raw job description text>"')
            print('       python jd_intake_pipeline.py --resume <thread_id> "<your answer>"  (after killing a paused run)')
            sys.exit(1)
