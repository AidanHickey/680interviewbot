from __future__ import annotations

import json
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from google import genai
from google.genai import types
from pypdf import PdfReader


MODEL_NAME = os.getenv("INTERVIEWIQ_MODEL", "gemini-2.5-flash")
HOST = os.getenv("INTERVIEWIQ_HOST", "127.0.0.1")
PORT = int(os.getenv("INTERVIEWIQ_PORT", "5000"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_CANDIDATES = [
    "interviewiq_frontend.html",
    "interview_bot.html",
]
HISTORY_FILE = os.path.join(BASE_DIR, "interviewiq_history.json")

BGRADE_RE = re.compile(r"\{[\s\S]*\}")
QUESTION_RE = re.compile(r"Question\s+(\d+)\s*[:\.\)]", re.IGNORECASE)
PLACEHOLDER_RE = re.compile(r"\[(?:applicant|candidate)\s*name\]|\{\{\s*name\s*\}\}", re.IGNORECASE)
NAME_PATTERNS = [
    re.compile(r"\bmy name is ([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b"),
    re.compile(r"\bI am ([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b"),
    re.compile(r"\bI'm ([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b"),
    re.compile(r"\bThis is ([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b"),
]
BLOCKED_NAME_WORDS = {
    "Software", "Engineer", "Developer", "Analyst", "Manager", "Designer", "Consultant",
    "Question", "Thanks", "Hello", "Hi", "Technology", "Finance", "Healthcare", "Marketing",
}

app = Flask(__name__, static_folder=BASE_DIR)
SESSIONS: dict[str, "InterviewSession"] = {}
SESSIONS_LOCK = threading.Lock()
STORE_LOCK = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    candidate_name: str = ""
    resume_text: str = ""
    resume_filename: str = ""
    conversation: list[InterviewMessage] = field(default_factory=list)
    system_prompt: str = ""
    question_count: int = 0
    answers_received: int = 0
    grading: dict[str, Any] | None = None
    complete: bool = False
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

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


def clean_candidate_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z\s'-]", " ", (value or "")).strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return ""
    words = text.split()
    if len(words) > 3:
        return ""
    if any(word in BLOCKED_NAME_WORDS for word in words):
        return ""
    return " ".join(word.capitalize() for word in words)


def detect_candidate_name(text: str) -> str:
    snippet = (text or "").strip()
    for pattern in NAME_PATTERNS:
        match = pattern.search(snippet)
        if match:
            candidate = clean_candidate_name(match.group(1))
            if candidate:
                return candidate
    return ""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


def extract_resume_text(filename: str, file_bytes: bytes) -> str:
    lower_name = (filename or "").lower()

    if lower_name.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore").strip()

    if lower_name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)

    raise ValueError("Unsupported file type. Please upload a PDF or TXT resume.")


def clean_resume_text(text: str, max_chars: int = 6000) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text[:max_chars]


def build_system_prompt(
    job_title: str,
    industry: str,
    difficulty: str,
    num_questions: int,
    focus_area: str,
    candidate_name: str,
    resume_text: str = "",
) -> str:
    focus = f"Focus areas requested: {focus_area}." if focus_area.strip() else ""
    candidate = f"The candidate's name is {candidate_name}." if candidate_name else "The candidate's name is not known."
    resume_block = (
        f"Candidate resume/background summary:\n{resume_text}\n\n"
        "Use the resume to personalize questions around the candidate's actual projects, skills, tools, education, and experience. "
        "Do not just repeat resume lines back; ask realistic interview questions based on them."
        if resume_text.strip()
        else ""
    )

    return f'''You are an expert senior interviewer at a top-tier {industry} company.
You are conducting a {difficulty} job interview for: {job_title}.
{candidate}

{resume_block}

Your instructions:
1. Ask exactly {num_questions} thoughtful, realistic interview questions appropriate for the role.
2. After each candidate answer, give a brief (1-2 sentence) acknowledgment or follow-up if needed, then ask the next question.
3. Vary question types: behavioral (STAR method), situational, technical/role-specific.
4. Number each question clearly, e.g. "Question 1:".
5. Never ask more than one numbered question in a single response.
6. Do not write END_INTERVIEW or any grading unless the system explicitly asks you to do grading.
7. Never use placeholders such as [Applicant Name], [Candidate Name], {{name}}, or similar. If the candidate name is unknown, use a neutral greeting.
8. Tailor questions to the specific role, difficulty, industry, focus areas, and resume if provided.

{focus}
'''


def build_initial_turn(system_prompt: str, candidate_name: str) -> str:
    greeting_rule = (
        f"Use the candidate's name, {candidate_name}, naturally once in the greeting."
        if candidate_name
        else "Do not invent a candidate name; use a neutral greeting instead."
    )
    return (
        f"[INTERVIEW CONTEXT]\n{system_prompt}\n\n"
        "Begin with a brief professional greeting (2-3 sentences), then ask exactly one question labeled Question 1:. "
        f"{greeting_rule} "
        "Do not ask more than one numbered question. Do not include grading."
    )


def build_next_turn_instruction(session: InterviewSession, answered_count: int, final_turn: bool = False) -> str:
    if final_turn:
        return (
            f"[CONTROL]\nThe candidate has answered all {session.num_questions} interview questions. "
            "Do not ask another question. Provide only a brief closing acknowledgment and no grading."
        )
    next_q = answered_count + 1
    return (
        f"[CONTROL]\nThe candidate has answered {answered_count} out of {session.num_questions} questions so far. "
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

    transcript = "\n\n".join(transcript_lines).strip() or "No interview transcript available."
    candidate_line = f"Candidate name: {session.candidate_name}\n" if session.candidate_name else ""
    focus_line = session.focus_area if session.focus_area else "No explicit focus areas provided."

    return f'''You are evaluating a mock job interview transcript.

Score the candidate specifically for this interview context, not against a generic interview rubric.

Interview context:
- Role: {session.job_title}
- Industry: {session.industry}
- Difficulty level: {session.difficulty}
- Focus areas: {focus_line}
{candidate_line}Scoring guidance:
- Communication: clarity, structure, concision, and professionalism.
- Relevance: how directly the answers match the role, industry, and question asked.
- Depth: depth of examples, technical reasoning, domain knowledge, and detail expected for this level.
- Confidence: poise, ownership, decisiveness, and credibility.
- Problem Solving: how well the candidate reasons through tradeoffs, ambiguity, constraints, and decision making.
- Strongly reward role-specific examples, realistic experience, and level-appropriate judgment.
- Penalize generic, vague, or off-role answers even if they sound polished.
- For senior/executive roles, expect more leadership, strategy, tradeoffs, and scope.
- For entry/mid-level roles, expect sound fundamentals, clear reasoning, and growth potential.

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
  "summary": "<2-3 sentence overall summary that references fit for the target role>",
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "improvements": ["<area 1>", "<area 2>", "<area 3>"],
  "tips": ["<actionable tip 1 tailored to the role>", "<actionable tip 2 tailored to the role>", "<actionable tip 3 tailored to the role>"]
}}

Transcript:
{transcript}
'''


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
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=message.text)]))
    return contents


def call_gemini(api_key: str, conversation: list[InterviewMessage], extra_instruction: str | None = None) -> str:
    messages = list(conversation)
    if extra_instruction:
        messages.append(InterviewMessage(role="user", text=extra_instruction))

    contents = to_gemini_contents(messages)
    with genai.Client(api_key=api_key) as client:
        response = client.models.generate_content(model=MODEL_NAME, contents=contents)
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("The model returned an empty response.")
    return text


def apply_name_safeguards(text: str, candidate_name: str) -> str:
    replacement = candidate_name if candidate_name else "there"
    return PLACEHOLDER_RE.sub(replacement, text)


def sanitize_single_question_reply(reply: str, expected_question_number: int, candidate_name: str) -> str:
    text = apply_name_safeguards(reply.strip(), candidate_name)
    if "END_INTERVIEW" in text:
        text = text.split("END_INTERVIEW", 1)[0].strip()

    matches = list(QUESTION_RE.finditer(text))
    if not matches:
        return text

    second = matches[1] if len(matches) > 1 else None
    trimmed = text[: second.start()].rstrip() if second else text
    normalized = QUESTION_RE.sub(f"Question {expected_question_number}:", trimmed, count=1)
    return normalized.strip()


def has_exactly_one_question(reply: str) -> bool:
    return len(QUESTION_RE.findall(reply)) == 1 and "END_INTERVIEW" not in reply


def generate_question_turn(session: InterviewSession, answered_count: int) -> str:
    expected_question_number = answered_count + 1
    instruction = build_next_turn_instruction(session, answered_count)
    reply = call_gemini(session.api_key, session.conversation, extra_instruction=instruction)
    cleaned = sanitize_single_question_reply(reply, expected_question_number, session.candidate_name)

    if has_exactly_one_question(cleaned):
        return cleaned

    retry_instruction = (
        f"[CORRECTION]\nYour previous answer did not follow instructions. "
        f"Reply again with a brief acknowledgment and exactly one question labeled Question {expected_question_number}:. "
        "No grading. No END_INTERVIEW. No extra numbered questions. Never use name placeholders."
    )
    retry_reply = call_gemini(session.api_key, session.conversation, extra_instruction=retry_instruction)
    retry_cleaned = sanitize_single_question_reply(retry_reply, expected_question_number, session.candidate_name)

    if has_exactly_one_question(retry_cleaned):
        return retry_cleaned

    return f"Thanks for that answer. Question {expected_question_number}: Can you walk me through a specific example that best demonstrates your fit for the {session.job_title} role?"


def generate_grading(session: InterviewSession) -> tuple[str, dict[str, Any] | None]:
    grading_prompt = build_grading_prompt(session)
    reply = call_gemini(session.api_key, [InterviewMessage(role="user", text=grading_prompt)])
    grading = None
    reply_text = reply

    if "END_INTERVIEW" in reply:
        before, after = reply.split("END_INTERVIEW", 1)
        reply_text = apply_name_safeguards(before.strip(), session.candidate_name)
        grading = sanitize_grading(parse_grading(after))
    if not grading:
        grading = sanitize_grading(parse_grading(reply))

    return reply_text, grading


def message_to_dict(message: InterviewMessage) -> dict[str, str]:
    return {"role": message.role, "text": message.text}


def message_from_dict(data: dict[str, Any]) -> InterviewMessage:
    return InterviewMessage(
        role=str(data.get("role", "user")),
        text=str(data.get("text", "")),
    )


def session_to_dict(session: InterviewSession) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "api_key": session.api_key,
        "job_title": session.job_title,
        "industry": session.industry,
        "difficulty": session.difficulty,
        "num_questions": session.num_questions,
        "focus_area": session.focus_area,
        "candidate_name": session.candidate_name,
        "resume_text": session.resume_text,
        "resume_filename": session.resume_filename,
        "conversation": [message_to_dict(m) for m in session.conversation],
        "system_prompt": session.system_prompt,
        "question_count": session.question_count,
        "answers_received": session.answers_received,
        "grading": session.grading,
        "complete": session.complete,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }


def session_from_dict(data: dict[str, Any]) -> InterviewSession:
    return InterviewSession(
        session_id=str(data.get("session_id", "")),
        api_key=str(data.get("api_key", "")),
        job_title=str(data.get("job_title", "")),
        industry=str(data.get("industry", "Technology")),
        difficulty=str(data.get("difficulty", "Mid Level")),
        num_questions=int(data.get("num_questions", 5)),
        focus_area=str(data.get("focus_area", "")),
        candidate_name=str(data.get("candidate_name", "")),
        resume_text=str(data.get("resume_text", "")),
        resume_filename=str(data.get("resume_filename", "")),
        conversation=[message_from_dict(m) for m in list(data.get("conversation", []))],
        system_prompt=str(data.get("system_prompt", "")),
        question_count=int(data.get("question_count", 0)),
        answers_received=int(data.get("answers_received", 0)),
        grading=data.get("grading"),
        complete=bool(data.get("complete", False)),
        created_at=str(data.get("created_at", now_iso())),
        updated_at=str(data.get("updated_at", now_iso())),
    )


def _read_store_unlocked() -> list[dict[str, Any]]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return []

    if isinstance(data, dict) and isinstance(data.get("sessions"), list):
        return data["sessions"]
    if isinstance(data, list):
        return data
    return []


def _write_store_unlocked(records: list[dict[str, Any]]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as fh:
        json.dump({"sessions": records}, fh, indent=2, ensure_ascii=False)


def persist_session(session: InterviewSession) -> None:
    session.updated_at = now_iso()
    record = session_to_dict(session)
    with STORE_LOCK:
        records = _read_store_unlocked()
        replaced = False
        for index, existing in enumerate(records):
            if str(existing.get("session_id")) == session.session_id:
                records[index] = record
                replaced = True
                break
        if not replaced:
            records.append(record)
        _write_store_unlocked(records)


def load_session_from_store(session_id: str) -> InterviewSession | None:
    with STORE_LOCK:
        records = _read_store_unlocked()
    for record in records:
        if str(record.get("session_id")) == session_id:
            return session_from_dict(record)
    return None


def get_session_by_id(session_id: str) -> InterviewSession | None:
    with SESSIONS_LOCK:
        existing = SESSIONS.get(session_id)
        if existing:
            return existing

    loaded = load_session_from_store(session_id)
    if loaded:
        with SESSIONS_LOCK:
            SESSIONS[session_id] = loaded
        return loaded
    return None


def weakest_category_from_grading(grading: dict[str, Any] | None) -> tuple[str | None, int | None]:
    if not grading:
        return None, None
    categories = grading.get("categories") or {}
    numeric_items: list[tuple[str, int]] = []
    for name, value in categories.items():
        try:
            numeric_items.append((str(name), int(value)))
        except (TypeError, ValueError):
            continue
    if not numeric_items:
        return None, None
    numeric_items.sort(key=lambda item: item[1])
    return numeric_items[0][0], numeric_items[0][1]


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/api/start", methods=["OPTIONS"])
@app.route("/api/message", methods=["OPTIONS"])
@app.route("/api/end", methods=["OPTIONS"])
@app.route("/api/parse_resume", methods=["OPTIONS"])
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
    return ok(
        {
            "ok": True,
            "message": "Backend is running, but no frontend HTML file was found next to the Python file.",
            "model": MODEL_NAME,
            "expected_frontend_files": FRONTEND_CANDIDATES,
            "base_dir": BASE_DIR,
        }
    )


@app.get("/health")
def health():
    return ok({"ok": True, "model": MODEL_NAME, "baseDir": BASE_DIR})


@app.get("/api/history")
def get_history():
    with STORE_LOCK:
        records = _read_store_unlocked()

    records.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)

    items: list[dict[str, Any]] = []
    for record in records:
        grading = record.get("grading") or {}
        weakest_name, weakest_score = weakest_category_from_grading(grading)
        items.append(
            {
                "sessionId": str(record.get("session_id", "")),
                "jobTitle": str(record.get("job_title", "")),
                "industry": str(record.get("industry", "")),
                "difficulty": str(record.get("difficulty", "")),
                "numQuestions": int(record.get("num_questions", 0)),
                "questionCount": int(record.get("question_count", 0)),
                "answersReceived": int(record.get("answers_received", 0)),
                "complete": bool(record.get("complete", False)),
                "candidateName": str(record.get("candidate_name", "")),
                "createdAt": str(record.get("created_at", "")),
                "updatedAt": str(record.get("updated_at", "")),
                "overallScore": grading.get("overall_score"),
                "grade": grading.get("grade"),
                "summary": grading.get("summary", ""),
                "weakestArea": weakest_name,
                "weakestScore": weakest_score,
                "resumeUploaded": bool(record.get("resume_filename")),
            }
        )

    return ok({"ok": True, "items": items})


@app.get("/api/history/<session_id>")
def get_history_session(session_id: str):
    session = get_session_by_id(session_id)
    if not session:
        return err("Session not found.", status=404, code="session_not_found")

    visible_conversation = [
        {"role": message.role, "text": message.text}
        for message in session.conversation
        if not (message.role == "user" and message.text.startswith("[INTERVIEW CONTEXT]"))
    ]

    return ok(
        {
            "ok": True,
            "session": {
                "sessionId": session.session_id,
                "candidateName": session.candidate_name,
                "jobTitle": session.job_title,
                "industry": session.industry,
                "difficulty": session.difficulty,
                "numQuestions": session.num_questions,
                "questionCount": session.question_count,
                "answersReceived": session.answers_received,
                "complete": session.complete,
                "grading": session.grading,
                "conversation": visible_conversation,
                "focusArea": session.focus_area,
                "resumeFilename": session.resume_filename,
                "meta": session.meta,
                "createdAt": session.created_at,
                "updatedAt": session.updated_at,
            }
        }
    )


@app.post("/api/parse_resume")
def parse_resume():
    if "resume" not in request.files:
        return err("Resume file is required.")

    uploaded = request.files["resume"]
    if not uploaded or not uploaded.filename:
        return err("Please choose a resume file.")

    try:
        file_bytes = uploaded.read()
        resume_text = clean_resume_text(extract_resume_text(uploaded.filename, file_bytes))
    except ValueError as exc:
        return err(str(exc))
    except Exception as exc:
        return err(f"Could not read resume: {exc}", status=400, code="resume_parse_failed")

    if not resume_text:
        return err("Resume file was read, but no text could be extracted.", status=400, code="empty_resume")

    return ok(
        {
            "ok": True,
            "filename": uploaded.filename,
            "resumeText": resume_text,
            "preview": resume_text[:400],
        }
    )


@app.post("/api/start")
def start_interview():
    data = request.get_json(silent=True) or {}

    api_key = str(data.get("apiKey", "")).strip()
    job_title = str(data.get("jobTitle", "")).strip()
    industry = str(data.get("industry", "Technology")).strip() or "Technology"
    difficulty = normalize_difficulty(str(data.get("difficulty", "Mid Level")))
    focus_area = str(data.get("focusArea", "")).strip()
    candidate_name = clean_candidate_name(str(data.get("candidateName", "")).strip())
    resume_text = clean_resume_text(str(data.get("resumeText", "")).strip())
    resume_filename = str(data.get("resumeFilename", "")).strip()

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

    system_prompt = build_system_prompt(
        job_title,
        industry,
        difficulty,
        num_questions,
        focus_area,
        candidate_name,
        resume_text,
    )

    session = InterviewSession(
        session_id=str(uuid.uuid4()),
        api_key=api_key,
        job_title=job_title,
        industry=industry,
        difficulty=difficulty,
        num_questions=num_questions,
        focus_area=focus_area,
        candidate_name=candidate_name,
        resume_text=resume_text,
        resume_filename=resume_filename,
        system_prompt=system_prompt,
    )

    session.conversation.append(InterviewMessage(role="user", text=build_initial_turn(system_prompt, candidate_name)))

    try:
        reply = generate_question_turn(session, answered_count=0)
    except Exception as exc:
        return err(f"Gemini request failed: {exc}", status=502, code="provider_error")

    session.conversation.append(InterviewMessage(role="model", text=reply))
    session.question_count = 1

    with SESSIONS_LOCK:
        SESSIONS[session.session_id] = session

    persist_session(session)

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
            "candidateName": session.candidate_name,
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

    session = get_session_by_id(session_id)

    if not session:
        return err("Session not found.", status=404, code="session_not_found")
    if session.complete:
        return err("This interview is already complete.", code="session_complete")

    inferred_name = detect_candidate_name(message)
    if inferred_name and not session.candidate_name:
        session.candidate_name = inferred_name

    session.conversation.append(InterviewMessage(role="user", text=message))
    session.answers_received += 1
    persist_session(session)

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
        persist_session(session)

        return ok(
            {
                "ok": True,
                "reply": reply_text,
                "done": True,
                "grading": grading,
                "questionCount": session.question_count,
                "numQuestions": session.num_questions,
                "candidateName": session.candidate_name,
            }
        )

    try:
        reply = generate_question_turn(session, answered_count=session.answers_received)
    except Exception as exc:
        return err(f"Gemini request failed: {exc}", status=502, code="provider_error")

    session.conversation.append(InterviewMessage(role="model", text=reply))
    session.question_count = session.answers_received + 1
    persist_session(session)

    return ok(
        {
            "ok": True,
            "reply": reply,
            "done": False,
            "grading": None,
            "questionCount": session.question_count,
            "numQuestions": session.num_questions,
            "candidateName": session.candidate_name,
        }
    )


@app.post("/api/end")
def end_interview():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("sessionId", "")).strip()

    if not session_id:
        return err("Session ID is required.")

    session = get_session_by_id(session_id)

    if not session:
        return err("Session not found.", status=404, code="session_not_found")
    if session.complete:
        return ok(
            {
                "ok": True,
                "done": True,
                "grading": session.grading,
                "reply": "",
                "candidateName": session.candidate_name,
            }
        )

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
    persist_session(session)

    return ok(
        {
            "ok": True,
            "done": True,
            "reply": reply_text,
            "grading": grading,
            "candidateName": session.candidate_name,
        }
    )


if __name__ == "__main__":
    print(f"InterviewIQ backend starting on http://{HOST}:{PORT} using model {MODEL_NAME}")
    app.run(host=HOST, port=PORT, debug=True)
