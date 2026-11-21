"""Streamlit Community Cloud entry point, launched from the repository root."""

import sys
from pathlib import Path

import streamlit as st

repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from demo.session import Visit, encode_photo  # noqa: E402

st.set_page_config(page_title="iDetectFace · Webcam check-in", layout="centered")
st.title("iDetectFace")
st.write("Enroll with one photo, then use a second photo to check yourself in.")
st.caption(
    "The server processes your photos. The demo keeps your face embedding and UTC check-ins in session memory. Use only your own photos."
)


def reset():
    st.session_state.clear()


def clear_result():
    st.session_state.pop("result", None)


if not st.checkbox("I agree to process my photos for this demo.", key="consent"):
    st.session_state.pop("visit", None)
    st.session_state.pop("result", None)
    st.stop()

visit = st.session_state.get("visit")
enrolling = visit is None
stage = "enroll" if enrolling else "check"
st.subheader("Enroll yourself" if enrolling else "Check in")
source = st.radio(
    "Photo source",
    ["Camera", "Upload"],
    horizontal=True,
    key=f"source_{stage}",
    on_change=clear_result,
)
if source == "Camera":
    photo = st.camera_input("Take a photo", key=f"camera_{stage}", on_change=clear_result)
else:
    photo = st.file_uploader(
        "Choose a JPEG or PNG",
        type=["jpg", "jpeg", "png"],
        key=f"upload_{stage}",
        on_change=clear_result,
    )

if st.button("Enroll" if enrolling else "Check this photo", type="primary", disabled=photo is None):
    st.session_state.pop("result", None)
    try:
        with st.spinner("Processing photo…"):
            vector, digest = encode_photo(photo.getvalue())
        if enrolling:
            st.session_state.visit = Visit(vector, digest)
            st.rerun()
        st.session_state.result = visit.check_in(vector, digest)
    except ValueError as error:
        st.warning(str(error))

if visit is not None:
    result = st.session_state.get("result")
    if result:
        status, distance = result
        message = f"{status} · face distance {distance:.3f}"
        (st.warning if status == "Unknown" else st.success)(message)
        st.caption("A distance below 0.500 accepts a match.")
    if visit.entries:
        st.subheader("Your attendance")
        st.table(
            [
                {"Participant": "visitor", "Date (UTC)": day, "First check-in (UTC)": at}
                for day, at in visit.entries.items()
            ]
        )
        st.download_button("Download CSV", visit.csv_text(), "demo-attendance.csv", "text/csv")
    st.button("Clear my session", on_click=reset)
