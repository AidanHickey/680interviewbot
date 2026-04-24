from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any




MODEL_NAME = os.getenv("INTERVIEWIQ_MODEL", "gemini-2.5-flash")
QUESTION_RE = re.compile(r"Question\s+(\d+)\s*[:\.)]", re.IGNORECASE)

app = Flask(__name__, static_folder=".")
SESSIONS: dict[str, "InterviewSession"] = {}


@dataclass
class InterviewSession:
    session_id: str
    api_key: str
    job_title: str
    industry: str
    difficulty: str
    num_questions: int
    focus_area: str = ""
    transcript: list[dict[str, str]] = field(default_factory=list)
    question_count: int = 0
    complete: bool = False

    @property
    def config_summary(self) -> str:
        return f"{self.job_title} · {self.industry} · {self.difficulty}"


def ok(data: dict[str, Any], status: int = 200):
    return jsonify(data), status


def err(message: str, status: int = 400, code: str = "bad_request"):
    return jsonify({"ok": False, "error": {"code": code, "message": message}}), status


def get_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def call_gemini(api_key: str, prompt: str) -> str:
    client = get_client(api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("The model returned an empty response.")
    return text


def build_transcript_text(session: InterviewSession) -> str:
    if not session.transcript:
        return "(No conversation yet)"

    lines: list[str] = []
    for turn in session.transcript:
        speaker = "Interviewer" if turn["role"] == "assistant" else "Candidate"
        lines.append(f"{speaker}: {turn['content']}")
    return "\n\n".join(lines)


def build_interview_prompt(session: InterviewSession, first_turn: bool = False) -> str:
    focus_line = (
        f"Focus areas requested by the candidate: {session.focus_area}."
        if session.focus_area.strip()
        else ""
    )

    if first_turn:
        task = (
            f"Start the interview. Give a brief warm professional greeting in 2 to 3 sentences, "
            f"then ask exactly one question labeled 'Question 1:'."
        )
    else:
        next_q = session.question_count + 1
        task = (
            f"Continue the interview. Briefly acknowledge the candidate's latest answer in 1 to 2 sentences, "
            f"then ask exactly one new question labeled 'Question {next_q}:'. Do not grade yet."
        )

    return f"""
You are an expert senior interviewer at a top-tier {session.industry} company conducting a {session.difficulty} interview for the role of {session.job_title}.

Rules:
- Ask exactly {session.num_questions} total interview questions across the full session.
- Keep questions realistic and relevant to the role.
- Vary question types: behavioral, situational, technical, and role-specific.
- Ask exactly one question in this response.
- Do not output JSON.
- Keep your response concise and polished.

{focus_line}

Conversation so far:
{build_transcript_text(session)}

Task:
{task}
""".strip()


GRADING_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "grade": {"type": "string", "enum": ["A", "B", "C", "D"]},
        "grade_label": {"type": "string"},
        "categories": {
            "type": "object",
            "properties": {
                "Communication": {"type": "integer", "minimum": 0, "maximum": 100},
                "Relevance": {"type": "integer", "minimum": 0, "maximum": 100},
                "Depth": {"type": "integer", "minimum": 0, "maximum": 100},
                "Confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                "Problem Solving": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            "required": [
                "Communication",
                "Relevance",
                "Depth",
                "Confidence",
                "Problem Solving",
            ],
        },
        "summary": {"type": "string"},
        "strengths": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
        "improvements": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
        "tips": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
    },
    "required": [
        "overall_score",
        "grade",
        "grade_label",
        "categories",
        "summary",
        "strengths",
        "improvements",
        "tips",
    ],
}


def build_grading_prompt(session: InterviewSession) -> str:
    return f"""
You are grading a completed mock interview for the role of {session.job_title} in {session.industry} at the {session.difficulty} level.

Return valid JSON only. No markdown. No commentary. Follow this schema exactly:
{json.dumps(GRADING_SCHEMA, indent=2)}

Grading guidance:
- overall_score is 0 to 100.
- grade must be A, B, C, or D.
- grade_label should be one of: Exceptional, Good, Adequate, Needs Improvement.
- summary should be 2 to 3 sentences.
- strengths, improvements, and tips must each contain exactly 3 concise items.

Interview transcript:
{build_transcript_text(session)}
""".strip()


def parse_grading_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("Could not find grading JSON in the model response.")
    return json.loads(match.group())


def sanitize_grading(grading: dict[str, Any]) -> dict[str, Any]:
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


def next_question_number(text: str, fallback: int) -> int:
    matches = QUESTION_RE.findall(text)
    if not matches:
        return fallback
    return max(fallback, max(int(m) for m in matches))


@app.get("/")
def root():
    return send_from_directory("/mnt/data", "interviewiq_frontend.html")


@app.get("/health")
def health():
    return ok({"ok": True, "model": MODEL_NAME})


@app.post("/api/start")
def start_session():
    data = request.get_json(silent=True) or {}
    api_key = str(data.get("apiKey", "")).strip()
    job_title = str(data.get("jobTitle", "")).strip()
    industry = str(data.get("industry", "Technology")).strip() or "Technology"
    difficulty = str(data.get("difficulty", "Mid Level")).strip() or "Mid Level"
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

    session = InterviewSession(
        session_id=str(uuid.uuid4()),
        api_key=api_key,
        job_title=job_title,
        industry=industry,
        difficulty=difficulty,
        num_questions=num_questions,
        focus_area=focus_area,
    )

    try:
        reply = call_gemini(session.api_key, build_interview_prompt(session, first_turn=True))
    except Exception as exc:
        return err(f"Gemini request failed: {exc}", status=502, code="provider_error")

    session.transcript.append({"role": "assistant", "content": reply})
    session.question_count = next_question_number(reply, fallback=1)
    SESSIONS[session.session_id] = session

    return ok(
        {
            "ok": True,
            "sessionId": session.session_id,
            "reply": reply,
            "questionCount": session.question_count,
            "numQuestions": session.num_questions,
            "meta": session.config_summary,
        }
    )


@app.post("/api/message")
def submit_message():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("sessionId", "")).strip()
    message = str(data.get("message", "")).strip()

    if not session_id or session_id not in SESSIONS:
        return err("Session not found.", status=404, code="session_not_found")
    if not message:
        return err("Please enter a message.")

    session = SESSIONS[session_id]
    if session.complete:
        return err("This interview is already complete.", code="session_complete")

    session.transcript.append({"role": "user", "content": message})

    if session.question_count >= session.num_questions:
        try:
            grading = sanitize_grading(parse_grading_json(call_gemini(session.api_key, build_grading_prompt(session))))
        except Exception as exc:
            return err(f"Could not generate grading: {exc}", status=502, code="grading_error")

        session.complete = True
        return ok({"ok": True, "done": True, "grading": grading})

    try:
        reply = call_gemini(session.api_key, build_interview_prompt(session, first_turn=False))
    except Exception as exc:
        return err(f"Gemini request failed: {exc}", status=502, code="provider_error")

    session.transcript.append({"role": "assistant", "content": reply})
    session.question_count = next_question_number(reply, fallback=session.question_count + 1)

    return ok(
        {
            "ok": True,
            "done": False,
            "reply": reply,
            "questionCount": session.question_count,
            "numQuestions": session.num_questions,
        }
    )


@app.post("/api/end")
def end_session():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("sessionId", "")).strip()

    if not session_id or session_id not in SESSIONS:
        return err("Session not found.", status=404, code="session_not_found")

    session = SESSIONS[session_id]
    if session.complete:
        return err("This interview is already complete.", code="session_complete")

    try:
        grading = sanitize_grading(parse_grading_json(call_gemini(session.api_key, build_grading_prompt(session))))
    except Exception as exc:
        return err(f"Could not generate grading: {exc}", status=502, code="grading_error")

    session.complete = True
    return ok({"ok": True, "done": True, "grading": grading})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
