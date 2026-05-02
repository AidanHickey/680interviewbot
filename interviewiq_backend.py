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
    "interviewiq_frontend_dualmode.html",
    "interviewiq_frontend.html",
    "interview_bot.html",
]

QUESTION_RE = re.compile(r"Question\s+(\d+)\s*[:\.\)]", re.IGNORECASE)
JSON_RE = re.compile(r"\{[\s\S]*\}")
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


@dataclass
class InterviewMessage:
    role: str
    text: str


@dataclass
class InterviewSession:
    session_id: str
    api_key: str
    mode: str
    job_title: str
    industry: str
    difficulty: str
    num_questions: int
    focus_area: str = ""
    candidate_name: str = ""
    conversation: list[InterviewMessage] = field(default_factory=list)
    system_prompt: str = ""
    question_count: int = 0
    answers_received: int = 0
    grading: dict[str, Any] | None = None
    employer_summary: dict[str, Any] | None = None
    complete: bool = False

    @property
    def meta(self) -> str:
        prefix = "Interviewee" if self.mode == "interviewee" else "Employer"
        name = f"{self.candidate_name} · " if self.candidate_name else ""
        return f"{prefix} · {name}{self.job_title} · {self.industry} · {self.difficulty}"


def ok(payload: dict[str, Any], status: int = 200):
    return jsonify(payload), status


def err(message: str, status: int = 400, code: str = "bad_request"):
    return jsonify({"ok": False, "error": {"code": code, "message": message}}), status


def normalize_difficulty(value: str) -> str:
    mapping = {
        "entry-level": "Entry Level",
        "mid-level": "Mid Level",
        "entry level": "Entry Level",
        "mid level": "Mid Level",
        "senior": "Senior",
        "executive": "Executive",
    }
    cleaned = (value or "").strip()
    return mapping.get(cleaned.lower(), cleaned or "Mid Level")


def normalize_mode(value: str) -> str:
    cleaned = (value or "interviewee").strip().lower()
    if cleaned in {"employer", "interviewer", "hiring_manager"}:
        return "employer"
    return "interviewee"


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


def apply_name_safeguards(text: str, candidate_name: str) -> str:
    replacement = candidate_name if candidate_name else "there"
    return PLACEHOLDER_RE.sub(replacement, text)


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

    with genai.Client(api_key=api_key) as client:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=to_gemini_contents(messages),
        )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("The model returned an empty response.")
    return text


def build_interviewee_system_prompt(job_title: str, industry: str, difficulty: str, num_questions: int, focus_area: str, candidate_name: str) -> str:
    focus = f"Focus areas requested: {focus_area}." if focus_area.strip() else ""
    candidate = f"The candidate's name is {candidate_name}." if candidate_name else "The candidate's name is not known."
    return f"""You are an expert senior interviewer at a top-tier {industry} company.
You are conducting a {difficulty} job interview for: {job_title}.
{candidate}

Your instructions:
1. Ask exactly {num_questions} thoughtful, realistic interview questions appropriate for the role.
2. After each candidate answer, give a brief (1-2 sentence) acknowledgment or follow-up if needed, then ask the next question.
3. Vary question types: behavioral (STAR method), situational, technical/role-specific.
4. Number each question clearly, e.g. "Question 1:".
5. Never ask more than one numbered question in a single response.
6. Do not write END_INTERVIEW or any grading unless the system explicitly asks you to do grading.
7. Never use placeholders such as [Applicant Name], [Candidate Name], {{name}}, or similar. If the candidate name is unknown, use a neutral greeting.
8. Tailor questions to the specific role, difficulty, industry, and any focus areas provided.

{focus}
"""


def build_employer_system_prompt(job_title: str, industry: str, difficulty: str, focus_area: str, candidate_name: str) -> str:
    focus = f"Special focus areas for the candidate persona: {focus_area}." if focus_area.strip() else ""
    candidate = f"Your simulated candidate name is {candidate_name}." if candidate_name else "Do not invent a formal name if none is provided; introduce yourself naturally."
    return f"""You are simulating a job candidate interviewing for a {difficulty} {job_title} role in the {industry} industry.
The human user is the employer or interviewer and will ask the questions.
{candidate}

Your instructions:
1. Answer as a realistic candidate for this specific role and level.
2. Keep answers professional, believable, and role-appropriate.
3. Use concrete examples, tradeoffs, and outcomes when useful, but do not claim impossible experience.
4. Do not ask numbered questions back to the user.
5. Do not output END_INTERVIEW or grading unless the system explicitly asks you to summarize the session.
6. Never use placeholders such as [Applicant Name], [Candidate Name], {{name}}, or similar.
7. Keep answers focused on the employer's question.
8. If the employer asks something vague, answer reasonably and note the assumption briefly.

{focus}
"""


def build_initial_turn(session: InterviewSession) -> str:
    if session.mode == "employer":
        greeting_rule = (
            f"Introduce yourself once as {session.candidate_name}."
            if session.candidate_name else
            "Introduce yourself without inventing a placeholder name."
        )
        return (
            f"[SESSION CONTEXT]\n{session.system_prompt}\n\n"
            "Start the session with a brief candidate introduction (2-3 sentences max) and invite the interviewer to begin asking questions. "
            f"{greeting_rule} Do not ask numbered questions. Do not grade."
        )

    greeting_rule = (
        f"Use the candidate's name, {session.candidate_name}, naturally once in the greeting."
        if session.candidate_name else
        "Do not invent a candidate name; use a neutral greeting instead."
    )
    return (
        f"[INTERVIEW CONTEXT]\n{session.system_prompt}\n\n"
        "Begin with a brief professional greeting (2-3 sentences), then ask exactly one question labeled Question 1:. "
        f"{greeting_rule} Do not ask more than one numbered question. Do not include grading."
    )


def build_next_turn_instruction(session: InterviewSession, answered_count: int) -> str:
    next_q = answered_count + 1
    return (
        f"[CONTROL]\nThe candidate has answered {answered_count} out of {session.num_questions} questions so far. "
        f"Respond with a brief acknowledgment (1-2 sentences), then ask exactly one new question labeled Question {next_q}:. "
        "Do not ask multiple numbered questions. Do not write END_INTERVIEW. Do not provide grading JSON."
    )


def sanitize_single_question_reply(reply: str, expected_question_number: int, candidate_name: str) -> str:
    text = apply_name_safeguards(reply.strip(), candidate_name)
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


def generate_initial_reply(session: InterviewSession) -> str:
    reply = call_gemini(session.api_key, [InterviewMessage(role="user", text=build_initial_turn(session))])
    if session.mode == "interviewee":
        reply = sanitize_single_question_reply(reply, 1, session.candidate_name)
        if not has_exactly_one_question(reply):
            correction = (
                "[CORRECTION]\nReply again with a short greeting and exactly one question labeled Question 1:. "
                "No grading. No END_INTERVIEW. No extra numbered questions. Never use name placeholders."
            )
            retry = call_gemini(session.api_key, [InterviewMessage(role="user", text=build_initial_turn(session))], correction)
            reply = sanitize_single_question_reply(retry, 1, session.candidate_name)
        if not has_exactly_one_question(reply):
            base = f"Welcome{' ' + session.candidate_name if session.candidate_name else ''}. Let's begin. Question 1: Can you tell me about yourself and why you're interested in this {session.job_title} role?"
            return base
    else:
        reply = apply_name_safeguards(reply, session.candidate_name)
        reply = reply.replace("END_INTERVIEW", "").strip()
    return reply


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


def generate_employer_turn(session: InterviewSession) -> str:
    instruction = (
        f"[CONTROL]\nAnswer the interviewer's latest question as a realistic {session.difficulty} candidate for the {session.job_title} role. "
        "Do not ask numbered questions back. Do not grade. Keep the answer focused and natural."
    )
    reply = call_gemini(session.api_key, session.conversation, extra_instruction=instruction)
    cleaned = apply_name_safeguards(reply, session.candidate_name).replace("END_INTERVIEW", "").strip()
    if QUESTION_RE.search(cleaned):
        correction = (
            "[CORRECTION]\nDo not ask numbered questions or behave like the interviewer. "
            "Reply only as the candidate answering the employer's last question."
        )
        retry = call_gemini(session.api_key, session.conversation, extra_instruction=correction)
        cleaned = apply_name_safeguards(retry, session.candidate_name).replace("END_INTERVIEW", "").strip()
    return cleaned or f"For this {session.job_title} role, I would focus on the responsibilities, collaboration style, and measurable outcomes I can deliver."


def build_grading_prompt(session: InterviewSession) -> str:
    transcript_lines: list[str] = []
    for message in session.conversation:
        speaker = "Candidate" if message.role == "user" else "Interviewer"
        transcript_lines.append(f"{speaker}: {message.text}")

    transcript = "\n\n".join(transcript_lines).strip() or "No interview transcript available."
    candidate_line = f"Candidate name: {session.candidate_name}\n" if session.candidate_name else ""
    focus_line = session.focus_area if session.focus_area else "No explicit focus areas provided."

    return f"""You are evaluating a mock job interview transcript.

Score the candidate specifically for this interview context, not against a generic interview rubric.

Interview context:
- Role: {session.job_title}
- Industry: {session.industry}
- Difficulty level: {session.difficulty}
- Focus areas: {focus_line}
{candidate_line}
Scoring guidance:
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
"""


def build_employer_summary_prompt(session: InterviewSession) -> str:
    transcript_lines: list[str] = []
    for message in session.conversation:
        speaker = "Candidate" if message.role == "model" else "Employer"
        transcript_lines.append(f"{speaker}: {message.text}")
    transcript = "\n\n".join(transcript_lines).strip() or "No transcript available."
    candidate_line = f"Candidate name: {session.candidate_name}\n" if session.candidate_name else ""
    focus_line = session.focus_area if session.focus_area else "No explicit focus areas provided."
    return f"""You are reviewing a mock interview where the human was the employer and the AI acted as the candidate.

Context:
- Target role: {session.job_title}
- Industry: {session.industry}
- Candidate level: {session.difficulty}
- Focus areas: {focus_line}
{candidate_line}
Return exactly one valid JSON object and no markdown:
{{
  "summary": "<2-3 sentence summary of how this candidate came across for the role>",
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "concerns": ["<concern 1>", "<concern 2>", "<concern 3>"],
  "follow_ups": ["<follow-up question 1>", "<follow-up question 2>", "<follow-up question 3>"],
  "candidate_profile": "<1-2 sentence snapshot of the simulated candidate's fit and style>"
}}

Use the transcript only. Focus on role fit, clarity, realism, depth, and what the employer should probe next.

Transcript:
{transcript}
"""


def parse_json_block(text: str) -> dict[str, Any] | None:
    match = JSON_RE.search(text)
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


def sanitize_employer_summary(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not summary:
        return None
    return {
        "summary": str(summary.get("summary", "")),
        "strengths": [str(x) for x in list(summary.get("strengths", []))[:3]],
        "concerns": [str(x) for x in list(summary.get("concerns", []))[:3]],
        "follow_ups": [str(x) for x in list(summary.get("follow_ups", []))[:3]],
        "candidate_profile": str(summary.get("candidate_profile", "")),
    }


def generate_grading(session: InterviewSession) -> tuple[str, dict[str, Any] | None]:
    prompt = build_grading_prompt(session)
    reply = call_gemini(session.api_key, [InterviewMessage(role="user", text=prompt)])
    reply_text = reply
    grading = None
    if "END_INTERVIEW" in reply:
        before, after = reply.split("END_INTERVIEW", 1)
        reply_text = apply_name_safeguards(before.strip(), session.candidate_name)
        grading = sanitize_grading(parse_json_block(after))
    if not grading:
        grading = sanitize_grading(parse_json_block(reply))
    return reply_text, grading


def generate_employer_summary(session: InterviewSession) -> dict[str, Any] | None:
    prompt = build_employer_summary_prompt(session)
    reply = call_gemini(session.api_key, [InterviewMessage(role="user", text=prompt)])
    return sanitize_employer_summary(parse_json_block(reply))


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
    candidate_name = clean_candidate_name(str(data.get("candidateName", "")).strip())
    mode = normalize_mode(str(data.get("mode", "interviewee")))

    try:
        num_questions = int(data.get("numQuestions", 5))
    except (TypeError, ValueError):
        return err("Number of questions must be a valid integer.")

    if not api_key:
        return err("Please enter your Gemini API key.")
    if not job_title:
        return err("Please enter a job title.")
    if mode == "interviewee" and num_questions not in {3, 5, 8, 10}:
        return err("Number of questions must be one of: 3, 5, 8, 10.")
    if mode == "employer":
        num_questions = max(1, num_questions)

    system_prompt = (
        build_interviewee_system_prompt(job_title, industry, difficulty, num_questions, focus_area, candidate_name)
        if mode == "interviewee"
        else build_employer_system_prompt(job_title, industry, difficulty, focus_area, candidate_name)
    )

    session = InterviewSession(
        session_id=str(uuid.uuid4()),
        api_key=api_key,
        mode=mode,
        job_title=job_title,
        industry=industry,
        difficulty=difficulty,
        num_questions=num_questions,
        focus_area=focus_area,
        candidate_name=candidate_name,
        system_prompt=system_prompt,
    )

    try:
        initial_reply = generate_initial_reply(session)
    except Exception as exc:
        return err(f"Gemini request failed: {exc}", status=502, code="provider_error")

    session.conversation.append(InterviewMessage(role="model", text=initial_reply))
    if mode == "interviewee":
        session.question_count = 1

    with SESSIONS_LOCK:
        SESSIONS[session.session_id] = session

    return ok({
        "ok": True,
        "sessionId": session.session_id,
        "reply": initial_reply,
        "done": False,
        "grading": None,
        "summaryData": None,
        "questionCount": session.question_count,
        "numQuestions": session.num_questions,
        "meta": session.meta,
        "candidateName": session.candidate_name,
        "mode": session.mode,
    })


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

    inferred_name = detect_candidate_name(message)
    if inferred_name and not session.candidate_name:
        session.candidate_name = inferred_name

    session.conversation.append(InterviewMessage(role="user", text=message))

    if session.mode == "employer":
        try:
            reply = generate_employer_turn(session)
        except Exception as exc:
            return err(f"Gemini request failed: {exc}", status=502, code="provider_error")

        session.conversation.append(InterviewMessage(role="model", text=reply))
        return ok({
            "ok": True,
            "reply": reply,
            "done": False,
            "grading": None,
            "summaryData": None,
            "questionCount": 0,
            "numQuestions": session.num_questions,
            "candidateName": session.candidate_name,
            "mode": session.mode,
        })

    session.answers_received += 1
    if session.answers_received >= session.num_questions:
        try:
            reply_text, grading = generate_grading(session)
        except Exception as exc:
            return err(f"Gemini request failed: {exc}", status=502, code="provider_error")

        if not grading:
            return err("Interview completed, but grading JSON could not be parsed.", status=502, code="grading_parse_failed")

        session.complete = True
        session.question_count = session.num_questions
        session.grading = grading
        return ok({
            "ok": True,
            "reply": reply_text,
            "done": True,
            "grading": grading,
            "summaryData": None,
            "questionCount": session.question_count,
            "numQuestions": session.num_questions,
            "candidateName": session.candidate_name,
            "mode": session.mode,
        })

    try:
        reply = generate_question_turn(session, answered_count=session.answers_received)
    except Exception as exc:
        return err(f"Gemini request failed: {exc}", status=502, code="provider_error")

    session.conversation.append(InterviewMessage(role="model", text=reply))
    session.question_count = session.answers_received + 1
    return ok({
        "ok": True,
        "reply": reply,
        "done": False,
        "grading": None,
        "summaryData": None,
        "questionCount": session.question_count,
        "numQuestions": session.num_questions,
        "candidateName": session.candidate_name,
        "mode": session.mode,
    })


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
        return ok({
            "ok": True,
            "done": True,
            "grading": session.grading,
            "summaryData": session.employer_summary,
            "reply": "",
            "candidateName": session.candidate_name,
            "mode": session.mode,
        })

    if session.mode == "employer":
        try:
            summary = generate_employer_summary(session)
        except Exception as exc:
            return err(f"Gemini request failed: {exc}", status=502, code="provider_error")
        if not summary:
            return err("Session completed, but the employer summary could not be parsed.", status=502, code="summary_parse_failed")
        session.complete = True
        session.employer_summary = summary
        return ok({
            "ok": True,
            "done": True,
            "reply": "",
            "grading": None,
            "summaryData": summary,
            "candidateName": session.candidate_name,
            "mode": session.mode,
        })

    try:
        reply_text, grading = generate_grading(session)
    except Exception as exc:
        return err(f"Gemini request failed: {exc}", status=502, code="provider_error")

    if not grading:
        return err("Interview completed, but grading JSON could not be parsed.", status=502, code="grading_parse_failed")

    session.complete = True
    session.question_count = min(session.question_count, session.num_questions)
    session.grading = grading
    return ok({
        "ok": True,
        "done": True,
        "reply": reply_text,
        "grading": grading,
        "summaryData": None,
        "candidateName": session.candidate_name,
        "mode": session.mode,
    })


if __name__ == "__main__":
    print(f"InterviewIQ backend starting on http://{HOST}:{PORT} using model {MODEL_NAME}")
    app.run(host=HOST, port=PORT, debug=True)
