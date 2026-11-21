"""Streamlit entry point for the guided two-photo check-in."""

import sys
from pathlib import Path

import streamlit as st

repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from demo.interface_text import TEXT  # noqa: E402
from demo.session import PhotoError, Visit, encode_photo  # noqa: E402


def clear_visit():
    for key in list(st.session_state):
        if key != "ui_language":
            del st.session_state[key]


def new_capture():
    st.session_state.pop("result", None)
    st.session_state.pop("photo_error", None)
    st.session_state.capture_number = st.session_state.get("capture_number", 0) + 1


def show_match_details(copy, result=None):
    with st.expander(copy["explain"]):
        st.write(copy["explanation"])
        if result is not None:
            st.write(f"{copy['score']} · {result[1]:.4f}")


def photo_step(copy, visit):
    first = visit is None
    if not first:
        st.success(copy["ready"])
    st.subheader(copy["first" if first else "second"])
    st.write(copy["first_hint" if first else "second_hint"])
    if reason := st.session_state.get("photo_error"):
        st.warning(copy[reason])
    methods = ["camera", "upload"]
    method = st.radio(
        copy["method"],
        methods,
        index=methods.index(st.session_state.get("preferred_method", "camera")),
        format_func=copy.__getitem__,
        horizontal=True,
        key="photo_method",
    )
    st.session_state.preferred_method = method
    widget_key = f"{method}_{st.session_state.get('capture_number', 0)}"
    if method == "camera":
        st.caption(copy["camera_hint"])
        photo = st.camera_input(copy["camera_label"], key=widget_key)
    else:
        photo = st.file_uploader(copy["upload_label"], type=["jpg", "jpeg", "png"], key=widget_key)
    if photo is None:
        return

    # Rotate the widget after every submission so reruns and language changes
    # cannot process a previous capture a second time.
    new_capture()
    try:
        with st.spinner(copy["working" if first else "comparing"]):
            vector, digest = encode_photo(photo.getvalue())
        if first:
            st.session_state.visit_bd = Visit(vector, digest)
        else:
            st.session_state.result = visit.check_in(vector, digest)
    except PhotoError as error:
        st.session_state.photo_error = error.reason
    st.rerun()


def result_step(copy, visit, result):
    status, _ = result
    st.subheader(copy["third"])
    message = {
        "Unknown": "unmatched",
        "Checked in": "recorded",
        "Already checked in today": "duplicate",
    }
    (st.warning if status == "Unknown" else st.success)(copy[status])
    st.write(copy[message[status]])
    if visit.entries:
        st.subheader(copy["record"])
        st.table([{copy["date"]: day, copy["time"]: at} for day, at in visit.entries.items()])
        st.download_button(
            copy["download"], visit.csv_text(), "demo-attendance.csv", "text/csv", on_click="ignore"
        )
    st.button(
        copy["again"], type="primary" if status == "Unknown" else "secondary", on_click=new_capture
    )
    show_match_details(copy, result)


st.set_page_config(page_title="iDetectFace · Photo check-in", layout="centered")
language = st.segmented_control(
    "Language / ভাষা", list(TEXT), default="English", key="ui_language", required=True
)
copy = TEXT[language]
st.title("iDetectFace")
st.write(copy["intro"])

if not st.session_state.get("consented"):
    st.subheader(copy["prepare"])
    st.write(copy["overview"])
    st.caption(copy["privacy"])
    if st.button(copy["start"], type="primary"):
        st.session_state.consented = True
        st.rerun()
    show_match_details(copy)
else:
    if "visit" in st.session_state:
        st.session_state.visit_bd = Visit.from_utc_session(st.session_state.pop("visit"))
    visit = st.session_state.get("visit_bd")
    result = st.session_state.get("result")
    if visit is not None and result is not None:
        result_step(copy, visit, result)
    else:
        photo_step(copy, visit)
    st.button(copy["clear"], on_click=clear_visit)
