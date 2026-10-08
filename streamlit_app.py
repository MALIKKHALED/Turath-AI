"""Turath AI chat UI (Streamlit). Thin display layer over the /query API.

Run:  uvicorn app.main:app        (terminal 1, port 8000)
      streamlit run streamlit_app.py  (terminal 2, port 8501)
"""

import os

import httpx
import streamlit as st
import base64
from pathlib import Path


BACKEND = os.environ.get("TURATH_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="تراث AI", page_icon="📜", layout="wide")

def _bg_css() -> str:
    p = Path("assets/bg.png")
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    return ("<style>.stApp { background-image: url('data:image/png;base64," + b64
            + "'); background-size: cover; background-attachment: fixed; }</style>")


_bg = _bg_css()
if _bg:
    st.markdown(_bg, unsafe_allow_html=True)

st.markdown(
    "<div dir='rtl' style='text-align:center'>"
    "<h1>📜 تراث AI</h1>"
    "<p>المساعد الذكي للتاريخ الإسلامي والسيرة النبوية</p>"
    "<p>─── ❖ ───</p></div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("<div dir='rtl'>", unsafe_allow_html=True)
    st.header("حول")
    st.write("إجابات موثقة من كتب السيرة والتاريخ، مع ذكر المصادر.")
    try:
        ok = httpx.get(f"{BACKEND}/health", timeout=5.0).is_success
    except Exception:
        ok = False
    st.markdown(" الخادم يعمل" if ok else " الخادم متوقف — شغّل uvicorn أولا",
                unsafe_allow_html=True)
    k = st.slider("عدد المقاطع المسترجعة", 1, 10, 5)
    if st.button("مسح المحادثة"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

left, mid, right = st.columns([1, 2.2, 1])
with mid:
    st.session_state.setdefault("messages", [])
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if question := st.chat_input("اسأل عن السيرة والتاريخ الإسلامي…"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("أبحث في الكتب…"):
                try:
                    resp = httpx.post(f"{BACKEND}/query",
                                      json={"question": question, "k": k},
                                      timeout=180.0)
                    resp.raise_for_status()
                    answer = resp.json()["answer"]
                except httpx.HTTPError:
                    answer = ("⚠️ تعذر الوصول إلى الخادم. "
                              "تأكد من تشغيل `uvicorn app.main:app` أولا.")
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})