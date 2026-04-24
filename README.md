# InterviewIQ

InterviewIQ is a browser-based mock interview coach built with a Python backend and a lightweight HTML/CSS/JavaScript frontend. It helps job seekers practice role-specific interviews, answer questions in a conversational format, and receive a structured evaluation at the end.

The project started as a desktop Tkinter prototype and was adapted into a local web app so the frontend and backend can work together more cleanly.

## What the project does

InterviewIQ lets a user:

- enter a Gemini API key locally
- choose a job title, industry, interview level, and number of questions
- optionally add focus areas such as system design or leadership
- complete a mock interview in a chat-style interface
- receive a final score, grade, category breakdown, strengths, improvements, and actionable tips

## Current architecture

The app is split into two local parts:

- **Frontend:** `interviewiq_frontend.html`
  - browser UI
  - setup screen, interview chat, results screen
  - sends requests to the local backend
- **Backend:** `interviewiq_backend.py` or `interviewiq_backend_fixed.py`
  - Flask server
  - manages interview sessions
  - calls the Gemini API using `google-genai`
  - parses grading JSON and returns structured results

## Tech stack

- Python 3.10+
- Flask
- Google Gen AI SDK (`google-genai`)
- HTML, CSS, JavaScript
- Gemini model: `gemini-2.5-flash` by default

## Project files

Recommended main files:

- `interviewiq_backend.py` — backend server
- `interviewiq_frontend.html` — frontend UI
- `requirements_interviewiq.txt` — Python dependencies
- `README.md` — project overview and setup instructions

Optional legacy/prototype files:

- `interview_bot.py` — original Tkinter desktop prototype
- `interview_bot.html` — earlier standalone HTML version
- `README_InterviewIQ.md` — earlier short setup readme

## Features

- role-based and level-based mock interviews
- chat-style interview flow
- configurable number of questions
- optional focus areas
- final grading with:
  - overall score
  - letter grade
  - 5 category scores
  - summary
  - strengths
  - improvements
  - actionable tips
- local backend/frontend integration
- local API key entry

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Install dependencies

```bash
pip install -r requirements_interviewiq.txt
```

If that does not work, try:

```bash
pip install flask google-genai
```

## How to run the app

Make sure the backend Python file and frontend HTML file are in the **same folder**.

Start the backend:

```bash
python interviewiq_backend.py
```

If you are using the fixed version instead:

```bash
python interviewiq_backend_fixed.py
```

Then open this in your browser:

```text
http://127.0.0.1:5000/
```

Do **not** rely on opening the HTML file directly unless you specifically want to test the standalone file behavior. The easiest and most reliable approach is to let Flask serve the frontend.

## How to use

1. Start the backend server.
2. Open `http://127.0.0.1:5000/`.
3. Enter your Gemini API key.
4. Enter a job title.
5. Choose industry, interview level, and number of questions.
6. Optionally add focus areas.
7. Click **Begin Interview**.
8. Answer each question in the chat UI.
9. End the interview or complete all questions.
10. Review the results screen.

## API routes

The backend exposes these routes:

- `GET /` — serves the frontend
- `GET /health` — basic backend health check
- `POST /api/start` — starts a session and gets the first question
- `POST /api/message` — sends a user answer and gets the next reply
- `POST /api/end` — ends the interview and returns grading

## Environment variables

Optional environment variables:

- `INTERVIEWIQ_MODEL` — override the Gemini model name
- `INTERVIEWIQ_HOST` — backend host, default `127.0.0.1`
- `INTERVIEWIQ_PORT` — backend port, default `5000`

Example:

```bash
INTERVIEWIQ_PORT=5050 python interviewiq_backend.py
```

Then open:

```text
http://127.0.0.1:5050/
```

## Troubleshooting

### Backend opens but only shows JSON

You are probably hitting a backend route that returns a status payload instead of the UI, or you are running an older backend file. Make sure you start the updated backend and open:

```text
http://127.0.0.1:5000/
```

### `Failed to fetch`

Most likely causes:

- backend is not running
- frontend and backend are on different origins
- you opened an older standalone HTML file
- the server was not restarted after file changes

Fix:

- stop the server with `Ctrl + C`
- restart the backend
- open the app from `http://127.0.0.1:5000/`

### Gemini validation / contents errors

This happened because the conversation was being sent to the SDK in the wrong format. Use the fixed backend version that converts chat history into Gemini `Content` / `Part` objects before calling `generate_content`.

### Module not found

Install the dependencies:

```bash
pip install flask google-genai
```

### Port already in use

Run on a different port:

```bash
INTERVIEWIQ_PORT=5050 python interviewiq_backend.py
```

## Security notes

- API keys are entered locally by the user.
- Do not commit real API keys to GitHub.
- Do not hard-code secrets into the frontend or backend.
- For a production version, session persistence, encryption, authentication, and stronger logging should be added.

## Current limitations

This version is still an MVP. It does **not** yet include:

- saved session history
- resume previous interviews
- admin question management
- company overview mode
- persistent database storage
- formal structured logging
- authentication / user accounts

## Future improvements

Potential next steps:

- move grading to structured JSON output mode
- add session history and resume support
- add role-specific question banks
- add admin tools for editing questions
- add export to PDF/CSV
- improve accessibility and keyboard support
- add logging and analytics
- separate services into cleaner modules

## Repository purpose

This repository demonstrates:

- a practical interview-prep application
- local frontend/backend integration
- Gemini API usage in Python
- iterative architecture refinement from prototype to web app

## License

Add your preferred license here, for example MIT.
