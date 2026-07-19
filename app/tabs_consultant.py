"""AI Consultant tab: a bounded, deterministic Q&A interface over analysis already computed
elsewhere in the app. See ``src.consultant``'s module docstring for the "no LLM" design
rationale -- this tab explains analysis, it never runs any.
"""

from __future__ import annotations

import streamlit as st
from tabs_month import FUEL_DISPLAY_LABELS, MODE_DISPLAY_LABELS

from src.consultant import ConsultantAnswer, ConsultantContext
from src.consultant_router import QUESTIONS, route_question

_CONFIDENCE_ICON = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}


def _render_answer(answer: ConsultantAnswer) -> None:
    st.write(f"**{answer.question}**")
    st.write(answer.answer)
    if answer.confidence:
        icon = _CONFIDENCE_ICON[answer.confidence]
        st.caption(f"Confidence: {icon} {answer.confidence}")
    if answer.evidence:
        with st.expander("Evidence"):
            for e in answer.evidence:
                st.write(f"- {e}")
    if answer.related_tab:
        st.caption(f"See more detail on the **{answer.related_tab}** tab.")


def render_consultant(ctx: ConsultantContext) -> None:
    st.caption(
        "Ask a question about your energy analysis in plain English, or pick one from the list "
        "below. This doesn't run any new analysis -- it explains what's already been computed "
        "elsewhere in the app, and always cites the evidence behind its answer. No LLM: it "
        "recognizes a bounded set of question types via keyword matching, not open-ended "
        "language understanding, so if it can't confidently match your question it shows the "
        "list instead of guessing."
    )

    if ctx.fuel_label:
        st.caption(f"Answering for: **{ctx.fuel_label}** (change via the sidebar's Fuel selector)")
    if ctx.selected_comparison_month is not None:
        st.caption(
            f"Month questions answer for: **{ctx.selected_comparison_month.strftime('%B %Y')}** · "
            f"**{FUEL_DISPLAY_LABELS.get(ctx.comparison_fuel, ctx.comparison_fuel)}** · "
            f"**{MODE_DISPLAY_LABELS.get(ctx.comparison_mode, ctx.comparison_mode)}** "
            "(change these on 'How did this month compare?')"
        )

    question_text = st.text_input("Ask a question", placeholder="e.g. Why did my bill go up?")

    if question_text:
        matched = route_question(question_text, ctx)
        if matched:
            _render_answer(matched)
            return
        st.info("I couldn't confidently match that to one of the questions I can answer -- try one below:")

    st.write("**Questions I can answer:**")
    for question, _handler in QUESTIONS:
        if st.button(question, key=f"consultant_q_{question}"):
            st.session_state["consultant_selected_question"] = question

    # Store the *question* and recompute the answer every rerun (handlers are sub-millisecond,
    # measured) -- storing the computed answer froze it against the context it was built from,
    # so switching fuel or toggling weather kept showing the old fuel's answer (audit finding F2).
    selected_question = st.session_state.get("consultant_selected_question")
    if selected_question and not question_text:
        handler = dict(QUESTIONS).get(selected_question)
        if handler:
            st.divider()
            _render_answer(handler(ctx))
