from __future__ import annotations

import json
import re
import socket
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import Page, expect


FRONTEND_CANDIDATES = [
    "interviewiq_frontend_dualmode.html",
    "interviewiq_frontend.html",
    "interviewiq_frontend_named.html",
    "interview_bot.html",
]


class QuietHandler(SimpleHTTPRequestHandler):
    """Static file server handler that stays quiet during tests."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


@pytest.fixture(scope="session")
def frontend_path() -> Path:
    """Find the frontend HTML file next to this test file or in the CWD."""
    search_dirs = [Path(__file__).resolve().parent, Path.cwd()]
    for directory in search_dirs:
        for name in FRONTEND_CANDIDATES:
            candidate = directory / name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        "Could not find a frontend HTML file. Looked for: "
        + ", ".join(FRONTEND_CANDIDATES)
    )


@pytest.fixture(scope="session")
def frontend_url(frontend_path: Path) -> str:
    """Serve the frontend over HTTP so fetch() uses same-origin requests."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    handler = partial(QuietHandler, directory=str(frontend_path.parent))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        yield f"http://127.0.0.1:{port}/{frontend_path.name}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.fixture()
def mock_backend(page: Page):
    """Mock the backend endpoints the frontend expects.

    The frontend uses fetch() for /health, /api/start, /api/message, and /api/end.
    These tests intercept those requests so you do not need a real Flask server
    or live Gemini credentials.
    """

    session_state: dict[str, Any] = {
        "mode": "interviewee",
        "message_calls": 0,
        "candidate_name": "Alex Johnson",
        "job_title": "Software Engineer",
        "industry": "Technology",
        "difficulty": "Mid Level",
        "num_questions": 3,
    }

    def fulfill_json(route, payload: dict[str, Any], status: int = 200) -> None:
        route.fulfill(
            status=status,
            content_type="application/json",
            body=json.dumps(payload),
        )

    def parse_request_json(route) -> dict[str, Any]:
        raw = route.request.post_data or "{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def handle_health(route) -> None:
        fulfill_json(route, {"ok": True, "model": "mock-gemini-test"})

    def handle_start(route) -> None:
        data = parse_request_json(route)
        mode = data.get("mode", "interviewee")
        session_state["mode"] = mode
        session_state["message_calls"] = 0
        session_state["candidate_name"] = data.get("candidateName") or "Alex Johnson"
        session_state["job_title"] = data.get("jobTitle") or "Software Engineer"
        session_state["industry"] = data.get("industry") or "Technology"
        session_state["difficulty"] = data.get("difficulty") or "Mid Level"
        session_state["num_questions"] = int(data.get("numQuestions") or 3)

        if mode == "employer":
            fulfill_json(
                route,
                {
                    "ok": True,
                    "sessionId": "sess-employer",
                    "mode": "employer",
                    "candidateName": session_state["candidate_name"],
                    "reply": (
                        f"Hello, I'm {session_state['candidate_name']}. I'm ready to answer "
                        f"questions about the {session_state['job_title']} role."
                    ),
                    "done": False,
                    "meta": (
                        f"{session_state['job_title']} · {session_state['industry']} · "
                        f"{session_state['difficulty']}"
                    ),
                },
            )
            return

        fulfill_json(
            route,
            {
                "ok": True,
                "sessionId": "sess-interviewee",
                "mode": "interviewee",
                "candidateName": session_state["candidate_name"],
                "reply": (
                    f"Welcome, {session_state['candidate_name']}. Question 1: "
                    "Tell me about yourself and why this role interests you."
                ),
                "done": False,
                "questionCount": 1,
                "numQuestions": session_state["num_questions"],
                "meta": (
                    f"{session_state['job_title']} · {session_state['industry']} · "
                    f"{session_state['difficulty']}"
                ),
            },
        )

    def handle_message(route) -> None:
        data = parse_request_json(route)
        message = data.get("message", "")
        session_state["message_calls"] += 1
        call_no = session_state["message_calls"]

        if session_state["mode"] == "employer":
            fulfill_json(
                route,
                {
                    "ok": True,
                    "sessionId": "sess-employer",
                    "mode": "employer",
                    "done": False,
                    "reply": (
                        f"As the candidate, I'd answer your question like this: {message}\n\n"
                        "I would prioritize customer outcomes, cross-functional communication, "
                        "and clear tradeoff reasoning."
                    ),
                    "candidateName": session_state["candidate_name"],
                },
            )
            return

        if call_no < session_state["num_questions"]:
            next_q = call_no + 1
            fulfill_json(
                route,
                {
                    "ok": True,
                    "sessionId": "sess-interviewee",
                    "mode": "interviewee",
                    "done": False,
                    "reply": (
                        "Thanks for that example. "
                        f"Question {next_q}: What is one challenge you solved recently, "
                        "and what was the result?"
                    ),
                    "questionCount": next_q,
                    "numQuestions": session_state["num_questions"],
                    "candidateName": session_state["candidate_name"],
                },
            )
            return

        fulfill_json(
            route,
            {
                "ok": True,
                "sessionId": "sess-interviewee",
                "mode": "interviewee",
                "done": True,
                "reply": "Thanks for completing the interview.",
                "questionCount": session_state["num_questions"],
                "numQuestions": session_state["num_questions"],
                "candidateName": session_state["candidate_name"],
                "grading": {
                    "overall_score": 87,
                    "grade": "B",
                    "grade_label": "Good",
                    "categories": {
                        "Communication": 90,
                        "Relevance": 88,
                        "Depth": 82,
                        "Confidence": 86,
                        "Problem Solving": 84,
                    },
                    "summary": "Strong role alignment with clear examples and solid reasoning.",
                    "strengths": [
                        "Clear communication",
                        "Relevant examples",
                        "Good tradeoff awareness",
                    ],
                    "improvements": [
                        "Add more metrics",
                        "Go deeper on system constraints",
                        "Tighten long answers",
                    ],
                    "tips": [
                        "Practice concise STAR responses",
                        "Use measurable outcomes",
                        "Prepare one architecture example",
                    ],
                },
            },
        )

    def handle_end(route) -> None:
        if session_state["mode"] == "employer":
            fulfill_json(
                route,
                {
                    "ok": True,
                    "sessionId": "sess-employer",
                    "mode": "employer",
                    "done": True,
                    "candidateName": session_state["candidate_name"],
                    "summaryData": {
                        "candidate_profile": "Senior product-oriented candidate with strong stakeholder communication.",
                        "summary": "The mock candidate handled follow-up questions well and explained tradeoffs clearly.",
                        "strengths": [
                            "Strong communication",
                            "Good product reasoning",
                            "Comfort with ambiguity",
                        ],
                        "concerns": [
                            "Could use more hard metrics",
                            "Needs sharper prioritization examples",
                        ],
                        "follow_ups": [
                            "Ask about a roadmap tradeoff",
                            "Ask for a conflict-resolution example",
                            "Ask how they use analytics to make decisions",
                        ],
                    },
                },
            )
            return

        fulfill_json(
            route,
            {
                "ok": True,
                "sessionId": "sess-interviewee",
                "mode": "interviewee",
                "done": True,
                "candidateName": session_state["candidate_name"],
                "grading": {
                    "overall_score": 80,
                    "grade": "B",
                    "grade_label": "Good",
                    "categories": {
                        "Communication": 82,
                        "Relevance": 79,
                        "Depth": 77,
                        "Confidence": 80,
                        "Problem Solving": 82,
                    },
                    "summary": "Solid finish with room to deepen examples.",
                    "strengths": ["Clear structure", "Professional tone", "Relevant examples"],
                    "improvements": ["More detail", "More metrics", "More tradeoff depth"],
                    "tips": ["Prepare examples", "Quantify impact", "Practice concise endings"],
                },
            },
        )

    page.route("**/health", handle_health)
    page.route("**/api/start", handle_start)
    page.route("**/api/message", handle_message)
    page.route("**/api/end", handle_end)

    return session_state


def fill_common_setup(page: Page, job_title: str = "Software Engineer") -> None:
    page.locator("#api-key").fill("test-api-key")
    page.locator("#candidate-name").fill("Alex Johnson")
    page.locator("#job-title").fill(job_title)


def test_frontend_shows_backend_ready(frontend_url: str, page: Page, mock_backend) -> None:
    page.goto(frontend_url)
    expect(page.locator("#backend-status")).to_contain_text("Local backend ready")
    expect(page.locator("#backend-status")).to_contain_text("mock-gemini-test")


def test_mode_toggle_updates_setup_labels(frontend_url: str, page: Page, mock_backend) -> None:
    page.goto(frontend_url)
    page.locator("#mode-employer").click()

    expect(page.locator("#setup-title")).to_have_text("Configure Employer Simulation")
    expect(page.locator("#mode-note")).to_contain_text("you play the employer")
    expect(page.locator("#job-title-label")).to_have_text("Target Role for Candidate")
    expect(page.locator("#questions-group")).to_have_class(re.compile(r"hidden"))
    expect(page.locator("#start-btn")).to_have_text(re.compile(r"Start Candidate Chat"))


def test_interviewee_mode_flow_reaches_graded_results(frontend_url: str, page: Page, mock_backend) -> None:
    page.goto(frontend_url)
    fill_common_setup(page)
    page.locator("#num-questions").select_option("3")
    page.locator("#start-btn").click()

    expect(page.locator("#interview-screen")).to_have_class(re.compile(r"active"))
    expect(page.locator("#chat-area")).to_contain_text("Question 1")
    expect(page.locator("#q-counter")).to_have_text("1 / 3")

    for answer in [
        "I enjoy solving backend problems.",
        "I recently improved an API's latency.",
        "I would design the feature around user impact and maintainability.",
    ]:
        page.locator("#user-input").fill(answer)
        page.locator("#send-btn").click()

    expect(page.locator("#results-screen")).to_have_class(re.compile(r"active"))
    expect(page.locator("#results-title")).to_contain_text("Software Engineer")
    expect(page.locator("#score-display")).to_have_text("87")
    expect(page.locator("#grade-badge")).to_contain_text("B")
    expect(page.locator("#summary-text")).to_contain_text("Strong role alignment")
    expect(page.locator("#weakest-area-name")).to_have_text("Depth")


def test_employer_mode_flow_reaches_summary_results(frontend_url: str, page: Page, mock_backend) -> None:
    page.goto(frontend_url)
    page.locator("#mode-employer").click()
    fill_common_setup(page, job_title="Product Manager")
    page.locator("#start-btn").click()

    expect(page.locator("#interview-screen")).to_have_class(re.compile(r"active"))
    expect(page.locator("#progress-area")).to_have_class(re.compile(r"hidden"))
    expect(page.locator("#end-btn")).to_have_text("End Session")
    expect(page.locator("#user-input")).to_have_attribute("placeholder", "Type your interview question here...")

    page.locator("#user-input").fill("How do you prioritize a roadmap with limited engineering time?")
    page.locator("#send-btn").click()
    expect(page.locator("#chat-area")).to_contain_text("As the candidate, I'd answer your question")

    page.once("dialog", lambda dialog: dialog.accept())
    page.locator("#end-btn").click()

    expect(page.locator("#results-screen")).to_have_class(re.compile(r"active"))
    expect(page.locator("#results-title")).to_have_text("Candidate Simulation Complete")
    expect(page.locator("#summary-text")).to_contain_text("Senior product-oriented candidate")
    expect(page.locator("#strengths-list")).to_contain_text("Strong communication")
    expect(page.locator("#tips-list")).to_contain_text("Ask about a roadmap tradeoff")


def test_start_shows_validation_error_without_required_fields(frontend_url: str, page: Page, mock_backend) -> None:
    page.goto(frontend_url)
    page.locator("#start-btn").click()
    expect(page.locator("#error-msg")).to_contain_text("Please enter your Gemini API key.")
