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
from google.genai import types


MODEL_NAME = os.getenv("INTERVIEWIQ_MODEL", "gemini-2.5-flash")
HOST = os.getenv("INTERVIEWIQ_HOST", "127.0.0.1")
PORT = int(os.getenv("INTERVIEWIQ_PORT", "5000"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_CANDIDATES = [
    "interviewiq_frontend.html",
    "interview_bot.html",
]

BGRADE_RE = re.compile(r"\{[\s\S]*\}")
QUESTION_RE = re.compile(r"Question\s+(\d+)\s*[:\.\)]", re.IGNORECASE)

app = Flask(__name__, static_folder=BASE_DIR)
SESSIONS: dict[str, "InterviewSession"] = {}
SESSIONS_LOCK = threading.Lock()


@dataclass
class InterviewMessage:
    role: str
    text: str


@dataclass
class InterviewSession:
    session_id: str
    api_key: str
    job_title: str
    industry: str
    difficulty: str
    num_questions: int
    focus_area: str = ""
    conversation: list[InterviewMessage] = field(default_factory=list)
    system_prompt: str = ""
    question_count: int = 0
    answers_received: int = 0
    grading: dict[str, Any] | None = None
    complete: bool = False

    @property
    def meta(self) -> str:
        return f"{self.job_title} · {self.industry} · {self.difficulty}"


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


def build_system_prompt(job_title: str, industry: str, difficulty: str, num_questions: int, focus_area: str) -> str:
    focus = f"Focus areas requested: {focus_area}." if focus_area.strip() else ""
    return f"""You are an expert senior interviewer at a top-tier {industry} company.
You are conducting a {difficulty} job interview for: {job_title}.

Your instructions:
1. Ask exactly {num_questions} thoughtful, realistic interview questions appropriate for the role.
2. After each candidate answer, give a brief (1-2 sentence) acknowledgment or follow-up if needed, then ask the next question.
3. Vary question types: behavioral (STAR method), situational, technical/role-specific.
4. Number each question clearly, e.g. \"Question 1:\".
5. Never ask more than one numbered question in a single response.
6. Do not write END_INTERVIEW or any grading unless the system explicitly asks you to do grading.

{focus}
"""


def build_initial_turn(system_prompt: str) -> str:
    return (
        f"[INTERVIEW CONTEXT]\n{system_prompt}\n\n"
        "Begin with a brief professional greeting (2-3 sentences), then ask exactly one question labeled Question 1:. "
        "Do not ask more than one numbered question. Do not include grading."
    )


def build_next_turn_instruction(answered_count: int, total_q: int) -> str:
    next_q = answered_count + 1
    return (
        f"[CONTROL]\nThe candidate has answered {answered_count} out of {total_q} questions so far. "
        f"Respond with a brief acknowledgment (1-2 sentences), then ask exactly one new question labeled Question {next_q}:. "
        "Do not ask multiple numbered questions. Do not write END_INTERVIEW. Do not provide grading JSON."
    )


def build_grading_prompt(session: InterviewSession) -> str:
    transcript_lines: list[str] = []
    for message in session.conversation:
        if message.role == "user" and message.text.startswith("[INTERVIEW CONTEXT]"):
            continue
        speaker = "Candidate" if message.role == "user" else "Interviewer"
        transcript_lines.append(f"{speaker}: {message.text}")

    transcript = "\n\n".join(transcript_lines).strip()
    if not transcript:
        transcript = "No interview transcript available."

    return f"""You are evaluating a mock job interview transcript.

Return EXACTLY:
1. A line containing only END_INTERVIEW
2. A valid JSON object with this exact structure and no markdown:
{{
  "overall_score": <integer 0-100>,
  "grade": "<A|B|C|D>",
  "grade_label": "<Exceptional|Good|Adequate|Needs Improvement>",
  "categories": {{
    "Communication": <0-100>,
    "Relevance": <0-100>,
    "Depth": <0-100>,
    "Confidence": <0-100>,
    "Problem Solving": <0-100>
  }},
  "summary": "<2-3 sentence overall summary>",
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "improvements": ["<area 1>", "<area 2>", "<area 3>"],
  "tips": ["<actionable tip 1>", "<actionable tip 2>", "<actionable tip 3>"]
}}

Transcript:
{transcript}
"""


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
        "overall_score": max(0, min(100, int(grading.get("overall_score", 0)))),
        "grade": str(grading.get("grade", "C")),
        "grade_label": str(grading.get("grade_label", "Adequate")),
        "categories": {
            "Communication": max(0, min(100, int(categories.get("Communication", 0)))),
            "Relevance": max(0, min(100, int(categories.get("Relevance", 0)))),
            "Depth": max(0, min(100, int(categories.get("Depth", 0)))),
            "Confidence": max(0, min(100, int(categories.get("Confidence", 0)))),
            "Problem Solving": max(0, min(100, int(categories.get("Problem Solving", 0)))),
        },
        "summary": str(grading.get("summary", "")),
        "strengths": [str(x) for x in list(grading.get("strengths", []))[:3]],
        "improvements": [str(x) for x in list(grading.get("improvements", []))[:3]],
        "tips": [str(x) for x in list(grading.get("tips", []))[:3]],
    }


def to_gemini_contents(conversation: list[InterviewMessage]) -> list[types.Content]:
    contents: list[types.Content] = []
    for message in conversation:
        role = "model" if message.role == "model" else "user"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=message.text)],
            )
        )
    return contents


def call_gemini(api_key: str, conversation: list[InterviewMessage], extra_instruction: str | None = None) -> str:
    messages = list(conversation)
    if extra_instruction:
        messages.append(InterviewMessage(role="user", text=extra_instruction))

    contents = to_gemini_contents(messages)
    with genai.Client(api_key=api_key) as client:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
        )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("The model returned an empty response.")
    return text


def sanitize_single_question_reply(reply: str, expected_question_number: int) -> str:
    text = reply.strip()
    if "END_INTERVIEW" in text:
        text = text.split("END_INTERVIEW", 1)[0].strip()

    matches = list(QUESTION_RE.finditer(text))
    if not matches:
        return text

    first = matches[0]
    second = matches[1] if len(matches) > 1 else None
    trimmed = text[: second.start()].rstrip() if second else text
    normalized = QUESTION_RE.sub(f"Question {expected_question_number}:", trimmed, count=1)
    return normalized.strip()


def has_exactly_one_question(reply: str) -> bool:
    return len(QUESTION_RE.findall(reply)) == 1 and "END_INTERVIEW" not in reply


def generate_question_turn(session: InterviewSession, answered_count: int) -> str:
    expected_question_number = answered_count + 1
    instruction = build_next_turn_instruction(answered_count, session.num_questions)
    reply = call_gemini(session.api_key, session.conversation, extra_instruction=instruction)
    cleaned = sanitize_single_question_reply(reply, expected_question_number)

    if has_exactly_one_question(cleaned):
        return cleaned

    retry_instruction = (
        f"[CORRECTION]\nYour previous answer did not follow instructions. "
        f"Reply again with a brief acknowledgment and exactly one question labeled Question {expected_question_number}:. "
        "No grading. No END_INTERVIEW. No extra numbered questions."
    )
    retry_reply = call_gemini(session.api_key, session.conversation, extra_instruction=retry_instruction)
    retry_cleaned = sanitize_single_question_reply(retry_reply, expected_question_number)

    if has_exactly_one_question(retry_cleaned):
        return retry_cleaned

    return f"Thanks for that answer. Question {expected_question_number}: Can you walk me through a specific example that best demonstrates your fit for this role?"


def generate_grading(session: InterviewSession) -> tuple[str, dict[str, Any] | None]:
    grading_prompt = build_grading_prompt(session)
    reply = call_gemini(session.api_key, [InterviewMessage(role="user", text=grading_prompt)])
    grading = None
    reply_text = reply

    if "END_INTERVIEW" in reply:
        before, after = reply.split("END_INTERVIEW", 1)
        reply_text = before.strip()
        grading = sanitize_grading(parse_grading(after))
    if not grading:
        grading = sanitize_grading(parse_grading(reply))

    return reply_text, grading


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/api/start", methods=["OPTIONS"])
@app.route("/api/message", methods=["OPTIONS"])
@app.route("/api/end", methods=["OPTIONS"])
def api_preflight():
    return ("", 204)


def _find_frontend_file() -> str | None:
    for name in FRONTEND_CANDIDATES:
        if os.path.exists(os.path.join(BASE_DIR, name)):
            return name
    return None


@app.get("/")
@app.get("/app")
def root():
    frontend_name = _find_frontend_file()
    if frontend_name:
        return send_from_directory(BASE_DIR, frontend_name)
    return ok({
        "ok": True,
        "message": "Backend is running, but no frontend HTML file was found next to the Python file.",
        "model": MODEL_NAME,
        "expected_frontend_files": FRONTEND_CANDIDATES,
        "base_dir": BASE_DIR,
    })


@app.get("/health")
def health():
    return ok({"ok": True, "model": MODEL_NAME, "baseDir": BASE_DIR})


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

    session.conversation.append(InterviewMessage(role="user", text=build_initial_turn(system_prompt)))

    try:
        reply = generate_question_turn(session, answered_count=0)
    except Exception as exc:
        return err(f"Gemini request failed: {exc}", status=502, code="provider_error")

    session.conversation.append(InterviewMessage(role="model", text=reply))
    session.question_count = 1

    with SESSIONS_LOCK:
        SESSIONS[session.session_id] = session

    return ok(
        {
            "ok": True,
            "sessionId": session.session_id,
            "reply": reply,
            "done": False,
            "grading": None,
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

    session.conversation.append(InterviewMessage(role="user", text=message))
    session.answers_received += 1

    if session.answers_received >= session.num_questions:
        try:
            reply_text, grading = generate_grading(session)
        except Exception as exc:
            return err(f"Gemini request failed: {exc}", status=502, code="provider_error")

        if not grading:
            return err(
                "Interview completed, but grading JSON could not be parsed.",
                status=502,
                code="grading_parse_failed",
            )

        session.complete = True
        session.question_count = session.num_questions
        session.grading = grading
        return ok(
            {
                "ok": True,
                "reply": reply_text,
                "done": True,
                "grading": grading,
                "questionCount": session.question_count,
                "numQuestions": session.num_questions,
            }
        )

    try:
        reply = generate_question_turn(session, answered_count=session.answers_received)
    except Exception as exc:
        return err(f"Gemini request failed: {exc}", status=502, code="provider_error")

    session.conversation.append(InterviewMessage(role="model", text=reply))
    session.question_count = session.answers_received + 1

    return ok(
        {
            "ok": True,
            "reply": reply,
            "done": False,
            "grading": None,
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

    try:
        reply_text, grading = generate_grading(session)
    except Exception as exc:
        return err(f"Gemini request failed: {exc}", status=502, code="provider_error")

    if not grading:
        return err(
            "Interview completed, but grading JSON could not be parsed.",
            status=502,
            code="grading_parse_failed",
        )

    session.complete = True
    session.question_count = min(session.question_count, session.num_questions)
    session.grading = grading

    return ok({"ok": True, "done": True, "reply": reply_text, "grading": grading})


if __name__ == "__main__":
    print(f"InterviewIQ backend starting on http://{HOST}:{PORT} using model {MODEL_NAME}")
    app.run(host=HOST, port=PORT, debug=True)
