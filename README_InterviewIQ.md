# InterviewIQ — AI Interview Coach
### Powered by Google Gemini API

---

## Quick Start

### 1. Install dependencies
```bash
pip install google-genai
```

### 2. Run the app
```bash
python interview_bot.py
```

### 3. Get a free Gemini API key
Visit https://aistudio.google.com/apikey — it's free, no credit card needed.

---

## How It Works

1. **Setup Screen** — Enter your API key, job title, industry, difficulty level, and number of questions
2. **Interview Screen** — Chat back and forth with the AI interviewer
3. **Results Screen** — See your score, grade, category breakdown, strengths, and improvement tips

---

## Features

- 🎯 Role-specific questions (behavioral, situational, technical)
- 📊 Scored across 5 categories: Communication, Relevance, Depth, Confidence, Problem Solving
- 💡 Actionable improvement tips
- ⚡ Powered by `gemini-2.5-flash-preview` for fast, intelligent responses
- 🔒 API key stays local — never stored or transmitted except to Google's API

---

## Requirements

- Python 3.9+
- `google-genai` package
- Tkinter (included with standard Python on Windows/macOS; on Linux: `sudo apt install python3-tk`)
