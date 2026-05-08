# Architecture

## Overview

InterviewIQ uses a **layered client-server architecture**.

At a high level:

- The **frontend** is an HTML/CSS/JavaScript single-page interface.
- The **backend** is a local **Python Flask** application.
- The backend manages **session state, prompt construction, Gemini API calls, grading, and response control**.
- The frontend is responsible for **user input, screen flow, chat rendering, progress display, and results presentation**.

This design was chosen to separate user interface concerns from AI/session logic. It makes the project easier to test, maintain, and extend than a single monolithic script.

---

## Architectural Style

The project follows a **layered client-server style** with clear responsibilities:

1. **Presentation Layer**
   - HTML/CSS/JavaScript frontend
   - Handles setup form, chat UI, results screen, and mode switching

2. **Application Layer**
   - Flask routes and session orchestration
   - Controls interview flow, employer mode flow, and grading triggers

3. **AI Integration Layer**
   - Gemini request construction and response handling
   - Encapsulates model communication and prompt behavior

4. **State / Session Layer**
   - In-memory session tracking
   - Stores conversation history, question count, mode, grading, and completion state

This is **not TDD architecture**.  
**TDD (Test-Driven Development)** is a development/testing approach, not a system architecture.  
The architecture is best described as **layered client-server with AI service integration**.

---

## High-Level Component Diagram

```text
Browser UI
(HTML/CSS/JS)
    |
    v
Flask Backend
(routes + session control)
    |
    v
Interview / Employer Logic
(prompt rules + flow control)
    |
    v
Gemini Integration
(google-genai SDK)
    |
    v
Gemini Model
```

---

## Main Components

## 1. Frontend

The frontend provides the user-facing application experience.

### Responsibilities
- Collects setup information:
  - API key
  - candidate name
  - job title
  - industry
  - difficulty
  - number of questions
  - focus areas
  - mode selection
- Sends requests to the backend
- Displays:
  - interview chat
  - progress dots
  - typing state
  - results report
  - weakest-area coaching
- Lets the user:
  - start an interview
  - send answers/questions
  - end a session early
  - download a report card
  - restart the app

### Frontend Characteristics
- Single-page style flow with three screens:
  - Setup
  - Interview
  - Results
- Uses browser `fetch()` calls to communicate with the backend
- Contains minimal business logic
- Defers session flow decisions to the backend

### Why this matters
Keeping business logic out of the frontend reduces duplication and makes the app easier to debug. The UI is mostly a renderer for backend decisions.

---

## 2. Backend

The backend is the central controller of the application.

### Responsibilities
- Serves the frontend HTML
- Provides API routes
- Creates and manages sessions
- Builds prompt instructions
- Sends requests to Gemini
- Controls question flow and completion logic
- Parses grading JSON
- Returns structured results to the frontend

### Current API Endpoints
- `GET /`
  - Serves the frontend page if present
- `GET /health`
  - Returns backend status and model info
- `POST /api/start`
  - Starts a new session
- `POST /api/message`
  - Processes a user message during a session
- `POST /api/end`
  - Ends a session early and generates final evaluation
- `OPTIONS` handlers
  - Support browser preflight / local integration

### Why Flask fits
Flask is lightweight, easy to run locally, and appropriate for a course project or demo app where the backend is mainly API orchestration rather than a large enterprise service.

---

## 3. Session Management

The app uses **in-memory session storage**.

### Session Data Includes
- session ID
- API key
- candidate name
- job title
- industry
- difficulty
- focus area
- mode
- conversation history
- question count
- answers received
- grading data
- completion flag

### Benefits
- Simple to implement
- Fast for local development
- Good for MVP/demo workflows

### Limitation
Because sessions are in memory:
- history is not persisted after restart
- multiple server restarts lose session state
- it is not yet suitable for production-scale use

### Future improvement
Replace or augment in-memory storage with:
- SQLite for local persistence
- PostgreSQL or another DB for multi-user scaling

---

## 4. Gemini Integration Layer

Gemini is integrated through the **`google-genai` SDK**.

### Responsibilities
- Converts internal conversation history into Gemini-compatible content objects
- Sends prompts to the model
- Receives text output
- Passes results back into interview/employer flow logic

### Current Pattern
The backend wraps Gemini access in helper logic rather than scattering direct API calls throughout the route handlers.

### Benefits
- Easier to patch or mock during testing
- Easier to swap providers later
- Cleaner separation between API routes and model-specific behavior

### Improvement direction
A future refactor could isolate this further into a dedicated provider class such as:

```python
class LLMProvider:
    def generate_turn(...)
    def generate_grading(...)
```

That would make provider replacement easier for OpenAI, Ollama, Anthropic, or another model.

---

## 5. Prompt / Control Architecture

A major part of the app architecture is **prompt control**.

The backend does not just “ask Gemini anything.”  
It builds structured instructions to reduce model drift.

### Prompting responsibilities
- define interview context
- control question numbering
- prevent multiple questions in one turn
- prevent premature ending
- tailor questions to role, industry, and level
- generate role-aware grading
- apply candidate name rules
- adapt behavior by mode

### Why this is important
Without backend control prompts, the model may:
- skip questions
- ask multiple questions at once
- end too early
- produce malformed grading
- use placeholder names

So prompt engineering here is part of the **application control layer**, not just content generation.

---

## 6. Dual-Mode Design

The application now supports two major usage modes.

### Interviewee Mode
The AI acts as the interviewer.

Flow:
1. User enters setup information
2. AI asks one question at a time
3. User answers
4. Backend tracks progress
5. Final grading is generated
6. Results page displays score and coaching

### Employer Mode
The AI acts as a candidate.

Flow:
1. User selects employer mode
2. User asks interview questions
3. AI responds as a candidate for the selected role
4. User can continue the conversation
5. Ending the session generates an interviewer-style summary instead of a numeric interview score

### Why this matters architecturally
The dual-mode design extends the same frontend and backend infrastructure without requiring an entirely separate app. The backend changes behavior through:
- mode-dependent prompts
- mode-dependent flow control
- mode-dependent result generation

This is a good example of **feature extension through orchestration logic rather than UI duplication**.

---

## 7. Grading Architecture

Grading is performed after the interview completes.

### Current grading outputs
- overall score
- letter grade
- grade label
- category scores:
  - Communication
  - Relevance
  - Depth
  - Confidence
  - Problem Solving
- summary
- strengths
- improvements
- tips

### Grading design principles
- role-aware
- difficulty-aware
- context-aware
- structured for UI display

### Why the current approach works
The frontend expects structured grading data, and the backend normalizes the result before returning it.

### Current limitation
The grading flow still depends on model-generated JSON.

### Future improvement
Use schema validation or structured generation constraints to reduce parsing failures further.

---

## 8. Error Handling Approach

The project uses a backend-first error handling model.

### Current behavior
- validation errors return structured JSON responses
- provider/model errors return standardized error payloads
- frontend displays user-readable messages
- completed sessions reject extra messages
- invalid start parameters are rejected before model calls

### Why this matters
This keeps the UI stable even when:
- the API key is invalid
- Gemini returns bad data
- the session is already complete
- the frontend submits incomplete setup data

### Recommended next step
Introduce explicit typed error categories:
- `CONFIG_*`
- `PROVIDER_*`
- `PARSING_*`
- `SESSION_*`
- `VALIDATION_*`

---

## 9. Security and Privacy Considerations

This project is designed primarily as a **local development/demo system**, but several architectural decisions already support safer handling.

### Current approach
- API key is entered by the user
- API key is sent only to the local backend
- the page does not persist the key
- the backend does not expose session internals directly to the UI

### Current limitations
- sessions are stored in memory
- API keys are still held in process memory during the session
- there is no authentication layer
- there is no persistent audit trail

### Future improvements
- key redaction in logs
- optional encrypted local persistence
- role-based admin access
- explicit retention rules
- privacy policy / compliance documentation

---

## 10. Testing Strategy

The project includes **Python-based automated tests** for both backend and frontend.

### Backend tests
Focus on:
- session creation
- mode handling
- question flow
- completion protection
- grading behavior
- validation errors

### Frontend tests
Focus on:
- UI flow
- required field validation
- mode switching
- start / message / results interactions

### Architectural value of tests
These tests help verify the orchestration logic that is hardest to validate manually, especially where AI behavior and session state interact.

Again, this means the project **uses automated testing**, but that does **not** make the architecture “TDD.” TDD describes how code is developed, not the shape of the system.

---

## 11. Why This Architecture Was Chosen

This architecture fits the project because it balances:
- simplicity
- local usability
- AI integration
- frontend polish
- testability
- future extensibility

A single-file chatbot would have been faster initially, but much harder to extend.  
Separating UI, backend routing, and AI/session control made it easier to add:

- candidate name support
- role-aware grading
- improved flow control
- report downloads
- weakest-area coaching
- employer mode
- automated tests

---

## 12. Current Limitations

The current architecture is strong for an MVP, but it still has limits:

- session data is not persistent
- model output can still be imperfect
- there is no user authentication
- there is no database-backed question bank
- analytics and admin capabilities are not yet present
- the backend is still relatively monolithic internally

---

## 13. Recommended Future Architecture Improvements

### Short-term
- add SQLite persistence
- add structured JSON schema validation
- split Gemini logic into a provider class
- add structured logging
- modularize backend into services

### Medium-term
- add admin/question-bank management
- add saved session history
- add company-context mode
- add analytics and progress tracking
- improve report export options

### Long-term
- move from in-memory session state to database-backed persistence
- add authentication and role-based access
- support multiple LLM providers
- prepare for containerized or cloud deployment

---

## 14. Summary

InterviewIQ currently uses a **layered client-server architecture**:

- **Frontend:** HTML/CSS/JavaScript UI
- **Backend:** Python Flask API
- **AI layer:** Gemini integration via `google-genai`
- **State layer:** in-memory session management
- **Control layer:** backend prompt/orchestration logic

This architecture is a good fit for a local AI-powered interview platform because it keeps the UI clean, centralizes flow control, and makes future enhancements more manageable.

If the project continues to grow, the next major step should be moving from a monolithic backend file to a more modular service-based structure with persistence and stronger structured validation.
