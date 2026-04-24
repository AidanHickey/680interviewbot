from __future__ import annotations

import json
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from google import genai


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_NAME = os.getenv("INTERVIEWIQ_MODEL", "gemini-2.5-flash")
HOST = os.getenv("INTERVIEWIQ_HOST", "127.0.0.1")
PORT = int(os.getenv("INTERVIEWIQ_PORT", "5000"))

BGRADE_RE = re.compile(r"\{[\s\S]*\}")
QUESTION_RE = re.compile(r"Question\s+(\d+)\s*[:\.\)]", re.IGNORECASE)

app = Flask(__name__, static_folder="/mnt/data")
SESSIONS: dict[str, "InterviewSession"] = {}
SESSIONS_LOCK = threading.Lock()


# --------------------------------------------------
# DATA MODELS
# --------------------------------------------------
@dataclass
class InterviewSession:
    session_id: str
    api_key: str
    job_title: str
    industry: str
    difficulty: str
    num_questions: int
    focus_area: str = ""
    conversation: list[dict[str, Any]] = field(default_factory=list)
    system_prompt: str = ""
    question_count: int = 0
    grading: dict[str, Any] | None = None
    complete: bool = False

    @property
    def meta(self) -> str:
        return f"{self.job_title} · {self.industry} · {self.difficulty}"


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def ok(payload: dict[str, Any], status: int = 200):
    return jsonify(payload), status


def err(message: str, status: int = 400, code: str = "bad_request"):
    return jsonify({"ok": False, "error": {"code": code, "message": message}}), status


def normalize_difficulty(value: str) -> str:
    mapping = {
        "entry-level": "Entry Level",
        "mid-level": "Mid Level",
        "senior": "Senior",
        "executive": "Executive",
        "entry level": "Entry Level",
        "mid level": "Mid Level",
    }
    cleaned = (value or "").strip()
    return mapping.get(cleaned.lower(), cleaned or "Mid Level")


def get_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def build_system_prompt(job_title: str, industry: str, difficulty: str, num_questions: int, focus_area: str) -> str:
    focus = f"Focus areas requested: {focus_area}." if focus_area.strip() else ""
    return f"""You are an expert senior interviewer at a top-tier {industry} company.
You are conducting a {difficulty} job interview for: {job_title}.

Your instructions:
1. Ask exactly {num_questions} thoughtful, realistic interview questions appropriate for the role.
2. After each candidate answer, give a brief (1-2 sentence) acknowledgment or follow-up if needed, then ask the next question.
3. Vary question types: behavioral (STAR method), situational, technical/role-specific.
4. Number each question clearly, e.g. \"Question 1:\".
5. After the candidate answers the {num_questions}th question, write EXACTLY this line alone: END_INTERVIEW
   Then immediately output a JSON grading block (no markdown, no backticks).

{focus}

JSON grading format (output this right after END_INTERVIEW, valid JSON only):
{{
  \"overall_score\": <integer 0-100>,
  \"grade\": \"<A|B|C|D>\",
  \"grade_label\": \"<Exceptional|Good|Adequate|Needs Improvement>\",
  \"categories\": {{
    \"Communication\": <0-100>,
    \"Relevance\": <0-100>,
    \"Depth\": <0-100>,
    \"Confidence\": <0-100>,
    \"Problem Solving\": <0-100>
  }},
  \"summary\": \"<2-3 sentence overall summary>\",
  \"strengths\": [\"<strength 1>\", \"<strength 2>\", \"<strength 3>\"],
  \"improvements\": [\"<area 1>\", \"<area 2>\", \"<area 3>\"],
  \"tips\": [\"<actionable tip 1>\", \"<actionable tip 2>\", \"<actionable tip 3>\"]
}}"""


def parse_grading(text: str) -> dict[str, Any] | None:
    match = BGRADE_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def sanitize_grading(grading: dict[str, Any] | None) -> dict[str, Any] | None:
    if not grading:
        return None

    categories = grading.get("categories") or {}
    return {
        "overall_score": int(grading.get("overall_score", 0)),
        "grade": str(grading.get("grade", "C")),
        "grade_label": str(grading.get("grade_label", "Adequate")),
        "categories": {
            "Communication": int(categories.get("Communication", 0)),
            "Relevance": int(categories.get("Relevance", 0)),
            "Depth": int(categories.get("Depth", 0)),
            "Confidence": int(categories.get("Confidence", 0)),
            "Problem Solving": int(categories.get("Problem Solving", 0)),
        },
        "summary": str(grading.get("summary", "")),
        "strengths": list(grading.get("strengths", []))[:3],
        "improvements": list(grading.get("improvements", []))[:3],
        "tips": list(grading.get("tips", []))[:3],
    }


def call_gemini(api_key: str, conversation: list[dict[str, Any]]) -> str:
    client = get_client(api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=conversation,
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("The model returned an empty response.")
    return text


def update_question_count(current_count: int, reply: str, total_q: int) -> int:
    matches = QUESTION_RE.findall(reply)
    if matches:
        return min(max(int(m) for m in matches), total_q)
    if "END_INTERVIEW" in reply:
        return total_q
    return current_count


def build_initial_turn(system_prompt: str) -> str:
    return (
        f"[INTERVIEW CONTEXT]\n{system_prompt}\n\n"
        "Begin with a brief professional greeting (2-3 sentences), then ask Question 1."
    )


def build_end_turn() -> str:
    return (
        "Please end the interview now. Provide the full grading assessment exactly as specified - "
        "write END_INTERVIEW on its own line, then the JSON block."
    )


# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.get("/")
def root():
    if os.path.exists("/mnt/data/interviewiq_frontend.html"):
        return send_from_directory("/mnt/data", "interviewiq_frontend.html")
    return ok({"ok": True, "message": "Backend is running.", "model": MODEL_NAME})


@app.get("/health")
def health():
    return ok({"ok": True, "model": MODEL_NAME})


@app.post("/api/start")
def start_interview():
    data = request.get_json(silent=True) or {}

    api_key = str(data.get("apiKey", "")).strip()
    job_title = str(data.get("jobTitle", "")).strip()
    industry = str(data.get("industry", "Technology")).strip() or "Technology"
    difficulty = normalize_difficulty(str(data.get("difficulty", "Mid Level")))
    focus_area = str(data.get("focusArea", "")).strip()

    try:
        num_questions = int(data.get("numQuestions", 5))
    except (TypeError, ValueError):
        return err("Number of questions must be a valid integer.")

    if not api_key:
        return err("Please enter your Gemini API key.")
    if not job_title:
        return err("Please enter a job title.")
    if num_questions not in {3, 5, 8, 10}:
        return err("Number of questions must be one of: 3, 5, 8, 10.")

    system_prompt = build_system_prompt(job_title, industry, difficulty, num_questions, focus_area)
    session = InterviewSession(
        session_id=str(uuid.uuid4()),
        api_key=api_key,
        job_title=job_title,
        industry=industry,
        difficulty=difficulty,
        num_questions=num_questions,
        focus_area=focus_area,
        system_prompt=system_prompt,
    )

    initial_turn = build_initial_turn(system_prompt)
    session.conversation.append({"role": "user", "parts": [initial_turn]})

    try:
        reply = call_gemini(session.api_key, session.conversation)
    except Exception as exc:
        return err(f"Gemini request failed: {exc}", status=502, code="provider_error")

    session.conversation.append({"role": "model", "parts": [reply]})
    session.question_count = update_question_count(0, reply, session.num_questions)

    grading = None
    reply_text = reply
    if "END_INTERVIEW" in reply:
        before, after = reply.split("END_INTERVIEW", 1)
        reply_text = before.strip()
        grading = sanitize_grading(parse_grading(after))
        session.complete = grading is not None
        session.question_count = session.num_questions
        session.grading = grading

    with SESSIONS_LOCK:
        SESSIONS[session.session_id] = session

    return ok(
        {
            "ok": True,
            "sessionId": session.session_id,
            "reply": reply_text,
            "done": session.complete,
            "grading": session.grading,
            "questionCount": session.question_count,
            "numQuestions": session.num_questions,
            "meta": session.meta,
        }
    )


@app.post("/api/message")
def send_message():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("sessionId", "")).strip()
    message = str(data.get("message", "")).strip()

    if not session_id:
        return err("Session ID is required.")
    if not message:
        return err("Please enter a message.")

    with SESSIONS_LOCK:
        session = SESSIONS.get(session_id)

    if not session:
        return err("Session not found.", status=404, code="session_not_found")
    if session.complete:
        return err("This interview is already complete.", code="session_complete")

    session.conversation.append({"role": "user", "parts": [message]})

    try:
        reply = call_gemini(session.api_key, session.conversation)
    except Exception as exc:
        return err(f"Gemini request failed: {exc}", status=502, code="provider_error")

    session.conversation.append({"role": "model", "parts": [reply]})
    session.question_count = update_question_count(session.question_count, reply, session.num_questions)

    grading = None
    reply_text = reply
    done = False

    if "END_INTERVIEW" in reply:
        before, after = reply.split("END_INTERVIEW", 1)
        reply_text = before.strip()
        grading = sanitize_grading(parse_grading(after))
        done = grading is not None
        session.complete = done
        session.question_count = session.num_questions if done else session.question_count
        session.grading = grading

    return ok(
        {
            "ok": True,
            "reply": reply_text,
            "done": done,
            "grading": grading,
            "questionCount": session.question_count,
            "numQuestions": session.num_questions,
        }
    )


@app.post("/api/end")
def end_interview():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("sessionId", "")).strip()

    if not session_id:
        return err("Session ID is required.")

    with SESSIONS_LOCK:
        session = SESSIONS.get(session_id)

    if not session:
        return err("Session not found.", status=404, code="session_not_found")
    if session.complete:
        return ok({"ok": True, "done": True, "grading": session.grading, "reply": ""})

    session.conversation.append({"role": "user", "parts": [build_end_turn()]})

    try:
        reply = call_gemini(session.api_key, session.conversation)
    except Exception as exc:
        return err(f"Gemini request failed: {exc}", status=502, code="provider_error")

    session.conversation.append({"role": "model", "parts": [reply]})

    reply_text = reply
    grading = None
    if "END_INTERVIEW" in reply:
        before, after = reply.split("END_INTERVIEW", 1)
        reply_text = before.strip()
        grading = sanitize_grading(parse_grading(after))
    else:
        grading = sanitize_grading(parse_grading(reply))

    if not grading:
        return err(
            "Interview completed, but grading JSON could not be parsed.",
            status=502,
            code="grading_parse_failed",
        )

    session.complete = True
    session.question_count = session.num_questions
    session.grading = grading

    return ok({"ok": True, "done": True, "reply": reply_text, "grading": grading})


if __name__ == "__main__":
    print(f"InterviewIQ backend starting on http://{HOST}:{PORT} using model {MODEL_NAME}")
    app.run(host=HOST, port=PORT, debug=True)
