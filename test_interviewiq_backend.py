from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


BACKEND_CANDIDATES = [
    "interviewiq_backend_dualmode.py",
    "interviewiq_backend.py",
]


def load_backend_module():
    base_dir = Path(__file__).resolve().parent
    for filename in BACKEND_CANDIDATES:
        path = base_dir / filename
        if path.exists():
            spec = importlib.util.spec_from_file_location("interviewiq_backend_under_test", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(
        f"Could not find any backend file next to the tests: {', '.join(BACKEND_CANDIDATES)}"
    )


@pytest.fixture(scope="module")
def backend():
    return load_backend_module()


@pytest.fixture(autouse=True)
def clear_sessions(backend):
    backend.SESSIONS.clear()
    yield
    backend.SESSIONS.clear()


@pytest.fixture()
def client(backend):
    backend.app.config.update(TESTING=True)
    return backend.app.test_client()


def test_normalize_mode_aliases(backend):
    assert backend.normalize_mode("interviewer") == "employer"
    assert backend.normalize_mode("hiring_manager") == "employer"
    assert backend.normalize_mode("INTERVIEWEE") == "interviewee"
    assert backend.normalize_mode("") == "interviewee"


def test_clean_and_detect_candidate_name(backend):
    assert backend.clean_candidate_name("alex johnson") == "Alex Johnson"
    assert backend.clean_candidate_name("Software Engineer") == ""
    assert backend.detect_candidate_name("Hi, my name is Jordan Lee.") == "Jordan Lee"
    assert backend.detect_candidate_name("I'm Software Engineer") == ""


def test_sanitize_single_question_reply_trims_extra_questions_and_end_marker(backend):
    raw = (
        "Hello [Candidate Name]. Question 1: Tell me about yourself.\n"
        "Question 2: What is your greatest strength?\nEND_INTERVIEW"
    )
    cleaned = backend.sanitize_single_question_reply(raw, expected_question_number=1, candidate_name="Alex")
    assert "Alex" in cleaned
    assert "END_INTERVIEW" not in cleaned
    assert cleaned.count("Question ") == 1
    assert cleaned.startswith("Hello Alex. Question 1:")


def test_start_interviewee_session(client, backend, monkeypatch):
    monkeypatch.setattr(
        backend,
        "generate_initial_reply",
        lambda session: "Welcome Alex. Question 1: Tell me about yourself.",
    )

    response = client.post(
        "/api/start",
        json={
            "apiKey": "test-key",
            "candidateName": "Alex",
            "jobTitle": "Software Engineer",
            "industry": "Technology",
            "difficulty": "Mid Level",
            "numQuestions": 3,
            "focusArea": "system design",
            "mode": "interviewee",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["mode"] == "interviewee"
    assert payload["questionCount"] == 1
    assert payload["candidateName"] == "Alex"
    assert "Question 1:" in payload["reply"]


def test_interviewee_message_progression_and_final_grading(client, backend, monkeypatch):
    monkeypatch.setattr(
        backend,
        "generate_initial_reply",
        lambda session: "Welcome. Question 1: Tell me about yourself.",
    )

    start_response = client.post(
        "/api/start",
        json={
            "apiKey": "test-key",
            "jobTitle": "Data Analyst",
            "industry": "Technology",
            "difficulty": "Entry Level",
            "numQuestions": 3,
            "mode": "interviewee",
        },
    )
    session_id = start_response.get_json()["sessionId"]

    monkeypatch.setattr(
        backend,
        "generate_question_turn",
        lambda session, answered_count: f"Thanks. Question {answered_count + 1}: Follow-up question?",
    )
    monkeypatch.setattr(
        backend,
        "generate_grading",
        lambda session: (
            "Thanks for completing the interview.",
            {
                "overall_score": 88,
                "grade": "B",
                "grade_label": "Good",
                "categories": {
                    "Communication": 85,
                    "Relevance": 90,
                    "Depth": 84,
                    "Confidence": 87,
                    "Problem Solving": 89,
                },
                "summary": "Strong fit for the role.",
                "strengths": ["Clear communication", "Relevant examples", "Good reasoning"],
                "improvements": ["More metrics", "Shorter openings", "Deeper tradeoffs"],
                "tips": ["Practice STAR", "Quantify impact", "Lead with outcome"],
            },
        ),
    )

    msg1 = client.post("/api/message", json={"sessionId": session_id, "message": "Answer 1"})
    data1 = msg1.get_json()
    assert msg1.status_code == 200
    assert data1["done"] is False
    assert data1["questionCount"] == 2
    assert "Question 2:" in data1["reply"]

    msg2 = client.post("/api/message", json={"sessionId": session_id, "message": "Answer 2"})
    data2 = msg2.get_json()
    assert msg2.status_code == 200
    assert data2["done"] is False
    assert data2["questionCount"] == 3
    assert "Question 3:" in data2["reply"]

    msg3 = client.post("/api/message", json={"sessionId": session_id, "message": "Answer 3"})
    data3 = msg3.get_json()
    assert msg3.status_code == 200
    assert data3["done"] is True
    assert data3["grading"]["overall_score"] == 88
    assert data3["questionCount"] == 3


def test_start_employer_session(client, backend, monkeypatch):
    monkeypatch.setattr(
        backend,
        "generate_initial_reply",
        lambda session: "Hi, I'm Alex. Happy to talk through my background for this product manager role.",
    )

    response = client.post(
        "/api/start",
        json={
            "apiKey": "test-key",
            "candidateName": "Alex",
            "jobTitle": "Product Manager",
            "industry": "Technology",
            "difficulty": "Senior",
            "numQuestions": 5,
            "mode": "employer",
        },
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["mode"] == "employer"
    assert payload["questionCount"] == 0
    assert payload["summaryData"] is None
    assert "Alex" in payload["reply"]


def test_employer_mode_message_and_end_summary(client, backend, monkeypatch):
    monkeypatch.setattr(
        backend,
        "generate_initial_reply",
        lambda session: "Hello, I'm Taylor. Please feel free to begin.",
    )
    start_response = client.post(
        "/api/start",
        json={
            "apiKey": "test-key",
            "candidateName": "Taylor",
            "jobTitle": "UX Designer",
            "industry": "Design / UX",
            "difficulty": "Mid Level",
            "mode": "employer",
        },
    )
    session_id = start_response.get_json()["sessionId"]

    monkeypatch.setattr(
        backend,
        "generate_employer_turn",
        lambda session: "I would start by understanding user pain points, then test prototypes quickly.",
    )
    monkeypatch.setattr(
        backend,
        "generate_employer_summary",
        lambda session: {
            "summary": "The candidate sounded thoughtful and practical.",
            "strengths": ["Clear answers", "User focus", "Good prioritization"],
            "concerns": ["Could go deeper on metrics", "Limited leadership detail", "Needs more scale examples"],
            "follow_ups": ["Tell me about a redesign", "How do you measure success?", "How do you handle tradeoffs?"],
            "candidate_profile": "A collaborative mid-level designer with strong product instincts.",
        },
    )

    msg = client.post(
        "/api/message",
        json={"sessionId": session_id, "message": "How do you approach design tradeoffs?"},
    )
    msg_data = msg.get_json()
    assert msg.status_code == 200
    assert msg_data["done"] is False
    assert msg_data["mode"] == "employer"
    assert msg_data["questionCount"] == 0
    assert "user pain points" in msg_data["reply"]

    end_response = client.post("/api/end", json={"sessionId": session_id})
    end_data = end_response.get_json()
    assert end_response.status_code == 200
    assert end_data["done"] is True
    assert end_data["grading"] is None
    assert end_data["summaryData"]["summary"] == "The candidate sounded thoughtful and practical."
    assert len(end_data["summaryData"]["follow_ups"]) == 3


def test_completed_session_rejects_new_messages(client, backend, monkeypatch):
    grading = {
        "overall_score": 75,
        "grade": "B",
        "grade_label": "Good",
        "categories": {
            "Communication": 75,
            "Relevance": 75,
            "Depth": 75,
            "Confidence": 75,
            "Problem Solving": 75,
        },
        "summary": "Solid performance.",
        "strengths": ["One", "Two", "Three"],
        "improvements": ["One", "Two", "Three"],
        "tips": ["One", "Two", "Three"],
    }

    monkeypatch.setattr(
        backend,
        "call_gemini",
        lambda api_key, conversation, extra_instruction=None: (
            "Welcome. Question 1: Why are you interested in this role?"
        ),
    )

    start_response = client.post(
        "/api/start",
        json={
            "apiKey": "test-key",
            "jobTitle": "QA Engineer",
            "industry": "Technology",
            "difficulty": "Mid Level",
            "numQuestions": 3,
        },
    )

    assert start_response.status_code == 200, start_response.get_json()
    session_id = start_response.get_json()["sessionId"]

    backend.SESSIONS[session_id].complete = True
    backend.SESSIONS[session_id].grading = grading

    response = client.post(
        "/api/message",
        json={"sessionId": session_id, "message": "Here is another answer."},
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"]["code"] == "session_complete"


def test_start_requires_api_key_and_job_title(client):
    response = client.post(
        "/api/start",
        json={"apiKey": "", "jobTitle": "", "mode": "interviewee"},
    )
    payload = response.get_json()
    assert response.status_code == 400
    assert payload["ok"] is False
    assert payload["error"]["message"] == "Please enter your Gemini API key."
