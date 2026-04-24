"""
InterviewIQ — AI Job Interview Coach
Powered by Google Gemini API (google-genai SDK)

Usage:
    pip install google-genai
    python interview_bot.py
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font
import threading
import json
import re
import os
from google import genai

# ─────────────────────────────────────────────
# THEME CONSTANTS
# ─────────────────────────────────────────────
BG          = "#0d0d14"
SURFACE     = "#14141f"
SURFACE2    = "#1c1c2e"
BORDER      = "#2a2a42"
ACCENT      = "#c8a96e"
ACCENT2     = "#7b68ee"
TEXT        = "#e8e8f0"
TEXT_DIM    = "#7a7a9a"
GREEN       = "#4ade80"
RED         = "#f87171"
YELLOW      = "#fbbf24"
WHITE       = "#ffffff"

FONT_HEADING  = ("Georgia", 20, "bold")
FONT_SUBHEAD  = ("Georgia", 13, "bold")
FONT_BODY     = ("Helvetica", 11)
FONT_BODY_SM  = ("Helvetica", 10)
FONT_MONO     = ("Courier", 10)
FONT_MONO_SM  = ("Courier", 9)
FONT_LABEL    = ("Courier", 9, "bold")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def build_system_prompt(job_title, industry, difficulty, num_questions, focus_area):
    focus = f"Focus areas requested: {focus_area}." if focus_area.strip() else ""
    return f"""You are an expert senior interviewer at a top-tier {industry} company.
You are conducting a {difficulty} job interview for: {job_title}.

Your instructions:
1. Ask exactly {num_questions} thoughtful, realistic interview questions appropriate for the role.
2. After each candidate answer, give a brief (1–2 sentence) acknowledgment or follow-up if needed, then ask the next question.
3. Vary question types: behavioral (STAR method), situational, technical/role-specific.
4. Number each question clearly, e.g. "Question 1:".
5. After the candidate answers the {num_questions}th question, write EXACTLY this line alone: END_INTERVIEW
   Then immediately output a JSON grading block (no markdown, no backticks).

{focus}

JSON grading format (output this right after END_INTERVIEW, valid JSON only):
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
}}"""


def parse_grading(text):
    """Extract JSON grading block from AI response."""
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def grade_color(score):
    if score >= 85: return GREEN
    if score >= 70: return ACCENT
    if score >= 55: return YELLOW
    return RED


# ─────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────
class InterviewApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("InterviewIQ — AI Interview Coach")
        self.geometry("940x720")
        self.minsize(780, 580)
        self.configure(bg=BG)

        # State
        self.api_key        = tk.StringVar()
        self.job_title      = tk.StringVar(value="")
        self.industry       = tk.StringVar(value="Technology")
        self.difficulty     = tk.StringVar(value="Mid Level")
        self.num_questions  = tk.IntVar(value=5)
        self.focus_area     = tk.StringVar(value="")

        self.conversation   = []   # list of {"role": ..., "parts": [...]}
        self.system_prompt  = ""
        self.question_count = 0
        self.total_q        = 5
        self.grading        = None
        self.client         = None

        self._build_ui()
        self.show_frame("setup")

    # ─── FRAME MANAGER ───────────────────────
    def show_frame(self, name):
        for f in (self.setup_frame, self.interview_frame, self.results_frame):
            f.pack_forget()
        if name == "setup":
            self.setup_frame.pack(fill=tk.BOTH, expand=True)
        elif name == "interview":
            self.interview_frame.pack(fill=tk.BOTH, expand=True)
        elif name == "results":
            self.results_frame.pack(fill=tk.BOTH, expand=True)

    # ─── BUILD ALL FRAMES ────────────────────
    def _build_ui(self):
        self._build_setup_frame()
        self._build_interview_frame()
        self._build_results_frame()

    # ═══════════════════════════════════════
    # SETUP FRAME
    # ═══════════════════════════════════════
    def _build_setup_frame(self):
        self.setup_frame = tk.Frame(self, bg=BG)

        # Scrollable inner canvas
        canvas = tk.Canvas(self.setup_frame, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.setup_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(canvas, bg=BG)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def on_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(inner_id, width=canvas.winfo_width())

        inner.bind("<Configure>", on_configure)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(inner_id, width=e.width))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        # ── LOGO AREA ──
        logo_frame = tk.Frame(inner, bg=BG, pady=32)
        logo_frame.pack(fill=tk.X)

        tk.Label(logo_frame, text="🎯", font=("Helvetica", 42), bg=BG).pack()
        tk.Label(logo_frame, text="InterviewIQ", font=("Georgia", 28, "bold"),
                 bg=BG, fg=ACCENT).pack()
        tk.Label(logo_frame, text="AI-POWERED INTERVIEW COACH", font=FONT_LABEL,
                 bg=BG, fg=TEXT_DIM).pack(pady=(2, 0))

        # ── CARD ──
        card = tk.Frame(inner, bg=SURFACE, padx=36, pady=32)
        card.pack(padx=60, pady=(0, 40), fill=tk.X)
        self._add_border(card)

        tk.Label(card, text="Configure Your Interview", font=FONT_SUBHEAD,
                 bg=SURFACE, fg=TEXT).pack(anchor="w")
        tk.Label(card, text="Set up your mock interview. The AI will ask tailored questions and grade your performance.",
                 font=FONT_BODY_SM, bg=SURFACE, fg=TEXT_DIM, wraplength=700, justify="left").pack(anchor="w", pady=(4, 20))

        # API Key
        self._field(card, "GOOGLE GEMINI API KEY", self.api_key, show="•",
                    placeholder="AIza...")

        # Job title
        self._field(card, "JOB TITLE / ROLE", self.job_title,
                    placeholder="e.g. Senior Software Engineer")

        # Industry + Difficulty row
        row = tk.Frame(card, bg=SURFACE)
        row.pack(fill=tk.X, pady=(0, 12))

        left = tk.Frame(row, bg=SURFACE)
        left.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        tk.Label(left, text="INDUSTRY", font=FONT_LABEL, bg=SURFACE, fg=TEXT_DIM).pack(anchor="w")
        ind_cb = ttk.Combobox(left, textvariable=self.industry, state="readonly",
                               values=["Technology","Finance","Healthcare","Marketing",
                                       "Education","Consulting","Sales","Design / UX",
                                       "Operations","Other"])
        self._style_combo(ind_cb)
        ind_cb.pack(fill=tk.X, ipady=5)

        right = tk.Frame(row, bg=SURFACE)
        right.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        tk.Label(right, text="INTERVIEW LEVEL", font=FONT_LABEL, bg=SURFACE, fg=TEXT_DIM).pack(anchor="w")
        diff_cb = ttk.Combobox(right, textvariable=self.difficulty, state="readonly",
                                values=["Entry Level","Mid Level","Senior","Executive"])
        self._style_combo(diff_cb)
        diff_cb.pack(fill=tk.X, ipady=5)

        # Number of questions
        tk.Label(card, text="NUMBER OF QUESTIONS", font=FONT_LABEL,
                 bg=SURFACE, fg=TEXT_DIM).pack(anchor="w", pady=(0, 4))
        q_cb = ttk.Combobox(card, textvariable=self.num_questions, state="readonly",
                              values=[3, 5, 8, 10])
        self._style_combo(q_cb)
        q_cb.pack(fill=tk.X, ipady=5, pady=(0, 12))

        # Focus area
        tk.Label(card, text="FOCUS AREAS (OPTIONAL)", font=FONT_LABEL,
                 bg=SURFACE, fg=TEXT_DIM).pack(anchor="w", pady=(0, 4))
        self.focus_text = tk.Text(card, height=3, bg=SURFACE2, fg=TEXT,
                                   insertbackground=ACCENT, relief="flat",
                                   font=FONT_BODY, wrap="word",
                                   highlightthickness=1, highlightbackground=BORDER,
                                   highlightcolor=ACCENT)
        self.focus_text.pack(fill=tk.X, pady=(0, 20))
        self.focus_text.insert("1.0", "e.g. system design, leadership, conflict resolution...")
        self.focus_text.bind("<FocusIn>", self._clear_placeholder)
        self.focus_text.config(fg=TEXT_DIM)

        # Error label
        self.setup_error = tk.Label(card, text="", font=FONT_BODY_SM,
                                     bg=SURFACE, fg=RED)
        self.setup_error.pack(anchor="w", pady=(0, 4))

        # Start button
        start_btn = tk.Button(card, text="BEGIN INTERVIEW  →",
                               font=FONT_LABEL, bg=ACCENT, fg=BG,
                               activebackground="#d4b07a", activeforeground=BG,
                               relief="flat", cursor="hand2", padx=20, pady=12,
                               command=self.start_interview)
        start_btn.pack(fill=tk.X)

    def _clear_placeholder(self, e):
        if self.focus_text.get("1.0", "end-1c") == "e.g. system design, leadership, conflict resolution...":
            self.focus_text.delete("1.0", tk.END)
            self.focus_text.config(fg=TEXT)

    def _field(self, parent, label, var, show=None, placeholder=""):
        tk.Label(parent, text=label, font=FONT_LABEL,
                 bg=SURFACE, fg=TEXT_DIM).pack(anchor="w", pady=(0, 4))
        e = tk.Entry(parent, textvariable=var, bg=SURFACE2, fg=TEXT,
                     insertbackground=ACCENT, relief="flat", font=FONT_BODY,
                     highlightthickness=1, highlightbackground=BORDER,
                     highlightcolor=ACCENT)
        if show:
            e.config(show=show)
        e.pack(fill=tk.X, ipady=7, pady=(0, 12))

    def _style_combo(self, cb):
        cb.configure(font=FONT_BODY)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox",
                        fieldbackground=SURFACE2, background=SURFACE2,
                        foreground=TEXT, selectbackground=SURFACE2,
                        selectforeground=TEXT, bordercolor=BORDER,
                        arrowcolor=TEXT_DIM)

    def _add_border(self, widget):
        widget.config(highlightthickness=1, highlightbackground=BORDER,
                      highlightcolor=ACCENT)

    # ═══════════════════════════════════════
    # INTERVIEW FRAME
    # ═══════════════════════════════════════
    def _build_interview_frame(self):
        self.interview_frame = tk.Frame(self, bg=BG)

        # ── HEADER ──
        header = tk.Frame(self.interview_frame, bg=SURFACE, pady=10, padx=16)
        header.pack(fill=tk.X)
        header.config(highlightthickness=1, highlightbackground=BORDER)

        left = tk.Frame(header, bg=SURFACE)
        left.pack(side=tk.LEFT)
        tk.Label(left, text="🎯  InterviewIQ", font=("Georgia", 13, "bold"),
                 bg=SURFACE, fg=ACCENT).pack(side=tk.LEFT)
        self.header_meta = tk.Label(left, text="", font=FONT_MONO_SM,
                                     bg=SURFACE, fg=TEXT_DIM)
        self.header_meta.pack(side=tk.LEFT, padx=(14, 0))

        right = tk.Frame(header, bg=SURFACE)
        right.pack(side=tk.RIGHT)
        self.progress_label = tk.Label(right, text="0 / 0", font=FONT_LABEL,
                                        bg=SURFACE, fg=TEXT_DIM)
        self.progress_label.pack(side=tk.LEFT, padx=(0, 12))

        end_btn = tk.Button(right, text="END & GRADE", font=FONT_LABEL,
                             bg=SURFACE2, fg=TEXT_DIM, relief="flat",
                             activebackground=RED, activeforeground=WHITE,
                             cursor="hand2", padx=10, pady=5,
                             command=self.end_interview_early)
        end_btn.pack(side=tk.LEFT)

        # ── CHAT AREA ──
        chat_wrap = tk.Frame(self.interview_frame, bg=BG)
        chat_wrap.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        self.chat_display = scrolledtext.ScrolledText(
            chat_wrap, bg=BG, fg=TEXT, font=FONT_BODY,
            relief="flat", wrap="word", state="disabled",
            padx=12, pady=8, spacing1=4, spacing3=4,
            insertbackground=ACCENT, cursor="arrow"
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)

        # Configure text tags
        self.chat_display.tag_config("ai_name",  foreground=ACCENT2, font=FONT_LABEL)
        self.chat_display.tag_config("ai_text",  foreground=TEXT,    font=FONT_BODY,    lmargin1=8, lmargin2=8)
        self.chat_display.tag_config("user_name",foreground=ACCENT,  font=FONT_LABEL)
        self.chat_display.tag_config("user_text",foreground="#c8d0e0",font=FONT_BODY,   lmargin1=8, lmargin2=8)
        self.chat_display.tag_config("divider",  foreground=BORDER)
        self.chat_display.tag_config("typing",   foreground=TEXT_DIM, font=("Helvetica", 10, "italic"))
        self.chat_display.tag_config("error",    foreground=RED,      font=FONT_BODY_SM)

        # ── INPUT AREA ──
        input_frame = tk.Frame(self.interview_frame, bg=SURFACE, pady=10, padx=12)
        input_frame.pack(fill=tk.X)
        input_frame.config(highlightthickness=1, highlightbackground=BORDER)

        self.user_input = tk.Text(input_frame, height=3, bg=SURFACE2, fg=TEXT,
                                   insertbackground=ACCENT, relief="flat",
                                   font=FONT_BODY, wrap="word",
                                   highlightthickness=1, highlightbackground=BORDER,
                                   highlightcolor=ACCENT)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10), ipady=6)
        self.user_input.bind("<Return>", self.on_enter_key)
        self.user_input.bind("<Shift-Return>", lambda e: None)

        self.send_btn = tk.Button(input_frame, text="SEND\n↵", font=FONT_LABEL,
                                   bg=ACCENT, fg=BG, activebackground="#d4b07a",
                                   activeforeground=BG, relief="flat",
                                   cursor="hand2", width=7, pady=8,
                                   command=self.send_message)
        self.send_btn.pack(side=tk.RIGHT)

        hint = tk.Label(input_frame, text="Enter to send  ·  Shift+Enter for new line",
                        font=FONT_MONO_SM, bg=SURFACE, fg=TEXT_DIM)
        hint.pack(side=tk.BOTTOM, pady=(6, 0))

    # ═══════════════════════════════════════
    # RESULTS FRAME
    # ═══════════════════════════════════════
    def _build_results_frame(self):
        self.results_frame = tk.Frame(self, bg=BG)

        # Scrollable
        canvas = tk.Canvas(self.results_frame, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.results_inner = tk.Frame(canvas, bg=BG)
        inner_id = canvas.create_window((0, 0), window=self.results_inner, anchor="nw")

        def on_cfg(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(inner_id, width=canvas.winfo_width())

        self.results_inner.bind("<Configure>", on_cfg)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(inner_id, width=e.width))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        self.results_canvas = canvas

    def populate_results(self, grading):
        """Populate the results frame with grading data."""
        for widget in self.results_inner.winfo_children():
            widget.destroy()

        pad = dict(padx=50, pady=0)
        score = grading.get("overall_score", 0)
        grade = grading.get("grade", "C")
        grade_label = grading.get("grade_label", "Adequate")

        # ── TITLE ──
        tk.Frame(self.results_inner, bg=BG, height=30).pack()
        tk.Label(self.results_inner, text="Interview Complete",
                 font=("Georgia", 22, "bold"), bg=BG, fg=TEXT).pack(**pad)
        tk.Label(self.results_inner,
                 text=f"{self.job_title.get()}  ·  {self.difficulty.get()}  ·  {self.num_questions.get()} questions",
                 font=FONT_MONO_SM, bg=BG, fg=TEXT_DIM).pack(**pad, pady=(4, 20))

        # ── SCORE CARD ──
        score_card = tk.Frame(self.results_inner, bg=SURFACE, pady=28)
        score_card.pack(fill=tk.X, **pad, pady=(0, 20))
        self._add_border(score_card)

        color = grade_color(score)
        tk.Label(score_card, text=str(score), font=("Georgia", 56, "bold"),
                 bg=SURFACE, fg=color).pack()
        tk.Label(score_card, text="OUT OF 100", font=FONT_LABEL,
                 bg=SURFACE, fg=TEXT_DIM).pack()

        grade_bg = {
            "A": "#0d2b1a", "B": "#2a2010", "C": "#2b2000", "D": "#2b0d0d"
        }.get(grade, SURFACE2)
        g_frame = tk.Frame(score_card, bg=grade_bg, padx=24, pady=6)
        g_frame.pack(pady=(12, 0))
        tk.Label(g_frame, text=f"  {grade} — {grade_label}  ",
                 font=("Courier", 11, "bold"), bg=grade_bg, fg=color).pack()

        # ── CATEGORY SCORES ──
        cat_header = self._section_header(self.results_inner, "📊  CATEGORY BREAKDOWN")
        cat_grid = tk.Frame(self.results_inner, bg=BG)
        cat_grid.pack(fill=tk.X, **pad, pady=(0, 20))

        categories = grading.get("categories", {})
        cat_colors = {
            "Communication": ACCENT2, "Relevance": GREEN,
            "Depth": ACCENT, "Confidence": "#60a5fa", "Problem Solving": "#f472b6"
        }

        cols = 2
        for idx, (name, val) in enumerate(categories.items()):
            col = idx % cols
            row = idx // cols
            c = cat_colors.get(name, ACCENT)

            cell = tk.Frame(cat_grid, bg=SURFACE, padx=16, pady=14)
            cell.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
            self._add_border(cell)
            cat_grid.columnconfigure(col, weight=1)

            top = tk.Frame(cell, bg=SURFACE)
            top.pack(fill=tk.X)
            tk.Label(top, text=name.upper(), font=FONT_LABEL,
                     bg=SURFACE, fg=TEXT_DIM).pack(side=tk.LEFT)
            tk.Label(top, text=str(val), font=("Georgia", 14, "bold"),
                     bg=SURFACE, fg=c).pack(side=tk.RIGHT)

            # Progress bar
            bar_bg = tk.Frame(cell, bg=BORDER, height=4)
            bar_bg.pack(fill=tk.X, pady=(8, 0))
            bar_fill_w = max(int((val / 100) * 300), 4)
            bar_fill = tk.Frame(bar_bg, bg=c, height=4)
            bar_fill.place(x=0, y=0, relwidth=val/100, relheight=1)

        # ── SUMMARY ──
        self._section_card(self.results_inner, "📋  OVERALL SUMMARY",
                           grading.get("summary", ""), pad)

        # ── STRENGTHS ──
        self._list_card(self.results_inner, "✅  STRENGTHS",
                        grading.get("strengths", []), GREEN, pad)

        # ── IMPROVEMENTS ──
        self._list_card(self.results_inner, "🎯  AREAS FOR IMPROVEMENT",
                        grading.get("improvements", []), YELLOW, pad)

        # ── TIPS ──
        self._list_card(self.results_inner, "💡  ACTIONABLE TIPS",
                        grading.get("tips", []), ACCENT2, pad)

        # ── RESTART ──
        tk.Frame(self.results_inner, bg=BG, height=10).pack()
        restart_btn = tk.Button(self.results_inner, text="←  START NEW INTERVIEW",
                                 font=FONT_LABEL, bg=SURFACE, fg=ACCENT,
                                 activebackground=ACCENT, activeforeground=BG,
                                 relief="flat", cursor="hand2", pady=12,
                                 highlightthickness=1, highlightbackground=ACCENT,
                                 command=self.restart)
        restart_btn.pack(fill=tk.X, **pad, pady=(0, 40))

        self.results_canvas.yview_moveto(0)

    def _section_header(self, parent, text):
        lbl = tk.Label(parent, text=text, font=FONT_LABEL,
                        bg=BG, fg=TEXT_DIM)
        lbl.pack(anchor="w", padx=50, pady=(8, 4))
        return lbl

    def _section_card(self, parent, title, text, pad):
        self._section_header(parent, title)
        card = tk.Frame(parent, bg=SURFACE, padx=20, pady=16)
        card.pack(fill=tk.X, **pad, pady=(0, 16))
        self._add_border(card)
        tk.Label(card, text=text, font=FONT_BODY, bg=SURFACE, fg=TEXT,
                 wraplength=700, justify="left").pack(anchor="w")

    def _list_card(self, parent, title, items, bullet_color, pad):
        self._section_header(parent, title)
        card = tk.Frame(parent, bg=SURFACE, padx=20, pady=16)
        card.pack(fill=tk.X, **pad, pady=(0, 16))
        self._add_border(card)
        for item in items:
            row = tk.Frame(card, bg=SURFACE)
            row.pack(fill=tk.X, pady=4)
            tk.Label(row, text="▸", font=FONT_BODY, bg=SURFACE,
                     fg=bullet_color).pack(side=tk.LEFT, anchor="nw", padx=(0, 8))
            tk.Label(row, text=item, font=FONT_BODY, bg=SURFACE, fg=TEXT,
                     wraplength=660, justify="left").pack(side=tk.LEFT, anchor="w")

    # ═══════════════════════════════════════
    # INTERVIEW LOGIC
    # ═══════════════════════════════════════
    def start_interview(self):
        api_key   = self.api_key.get().strip()
        job_title = self.job_title.get().strip()

        if not api_key:
            self.setup_error.config(text="⚠  Please enter your Google Gemini API key.")
            return
        if not job_title:
            self.setup_error.config(text="⚠  Please enter a job title.")
            return
        self.setup_error.config(text="")

        focus = self.focus_text.get("1.0", "end-1c").strip()
        if focus == "e.g. system design, leadership, conflict resolution...":
            focus = ""

        self.total_q = self.num_questions.get()
        self.system_prompt = build_system_prompt(
            job_title, self.industry.get(), self.difficulty.get(),
            self.total_q, focus
        )
        self.conversation = []
        self.question_count = 0
        self.grading = None

        try:
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            self.setup_error.config(text=f"⚠  Client error: {e}")
            return

        # Update header
        self.header_meta.config(
            text=f"  {job_title}  ·  {self.industry.get()}  ·  {self.difficulty.get()}"
        )
        self.update_progress()

        # Clear chat
        self.chat_display.config(state="normal")
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state="disabled")

        self.show_frame("interview")
        self.set_input_enabled(False)

        # Kick off first AI message in background
        init_prompt = (
            f"[INTERVIEW CONTEXT]\n{self.system_prompt}\n\n"
            "Begin with a brief professional greeting (2-3 sentences), "
            "then ask Question 1."
        )
        self.conversation.append({"role": "user", "parts": [init_prompt]})
        threading.Thread(target=self._call_gemini, daemon=True).start()

    def on_enter_key(self, event):
        if not event.state & 0x1:  # Shift not held
            self.send_message()
            return "break"

    def send_message(self):
        text = self.user_input.get("1.0", "end-1c").strip()
        if not text:
            return
        self.user_input.delete("1.0", tk.END)
        self._append_chat("user_name", "YOU\n")
        self._append_chat("user_text", text + "\n\n")
        self.conversation.append({"role": "user", "parts": [text]})
        self.set_input_enabled(False)
        threading.Thread(target=self._call_gemini, daemon=True).start()

    def end_interview_early(self):
        if not messagebox.askyesno("End Interview",
                                    "End the interview now and get your grade?"):
            return
        self.set_input_enabled(False)
        end_msg = ("Please end the interview now. Provide the full grading assessment "
                   "exactly as specified — write END_INTERVIEW on its own line, "
                   "then the JSON block.")
        self.conversation.append({"role": "user", "parts": [end_msg]})
        threading.Thread(target=self._call_gemini, daemon=True).start()

    def _call_gemini(self):
        self.after(0, self._show_typing)
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-preview-04-17",
                contents=self.conversation,
            )
            reply = response.text or ""
            self.conversation.append({"role": "model", "parts": [reply]})
            self.after(0, lambda: self._handle_reply(reply))
        except Exception as e:
            self.after(0, lambda: self._handle_error(str(e)))

    def _handle_reply(self, reply):
        self._remove_typing()

        if "END_INTERVIEW" in reply:
            parts = reply.split("END_INTERVIEW", 1)
            before = parts[0].strip()
            after  = parts[1].strip() if len(parts) > 1 else ""

            if before:
                self._append_chat("ai_name", "INTERVIEWER\n")
                self._append_chat("ai_text", before + "\n\n")

            grading = parse_grading(after)
            if grading:
                self.grading = grading
                self.after(800, lambda: (
                    self.populate_results(grading),
                    self.show_frame("results")
                ))
            else:
                self._append_chat("error",
                    "⚠  Interview complete, but grading data could not be parsed.\n\n")
        else:
            # Count questions
            if re.search(r'Question\s+\d+\s*[:\.\)]', reply, re.IGNORECASE):
                self.question_count = min(self.question_count + 1, self.total_q)
                self.update_progress()

            self._append_chat("ai_name", "INTERVIEWER\n")
            self._append_chat("ai_text", reply + "\n\n")
            self.set_input_enabled(True)
            self.user_input.focus_set()

    def _handle_error(self, msg):
        self._remove_typing()
        self._append_chat("error", f"⚠  Error: {msg}\n\n")
        self.set_input_enabled(True)

    def _show_typing(self):
        self._append_chat("typing", "Interviewer is typing…\n", tag="typing_indicator")

    def _remove_typing(self):
        self.chat_display.config(state="normal")
        try:
            start = self.chat_display.tag_ranges("typing_indicator")
            if start:
                self.chat_display.delete(start[0], start[1])
        except Exception:
            pass
        self.chat_display.config(state="disabled")

    def _append_chat(self, tag, text, tag_override=None):
        self.chat_display.config(state="normal")
        t = tag_override or tag
        self.chat_display.insert(tk.END, text, (t, tag))
        self.chat_display.see(tk.END)
        self.chat_display.config(state="disabled")

    def update_progress(self):
        self.progress_label.config(
            text=f"{self.question_count} / {self.total_q}"
        )

    def set_input_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        self.user_input.config(state=state)
        self.send_btn.config(state=state)

    def restart(self):
        self.show_frame("setup")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = InterviewApp()
    app.mainloop()
