import io
import os
import pickle
import numpy as np
import pandas as pd
from shiny import App, reactive, render, ui

# ─────────────────────────────────────────────────────────────────────────────
# Config  (PARENT version)
# ─────────────────────────────────────────────────────────────────────────────
CSS_PATH = "styles.css"

# Input score ranges. NOTE: the 1-STS binarization threshold below assumes HBI
# items are scored 0–3 (present if score > 1). If your HBI uses 0–4, revisit
# HBI_MAX and BINARIZE_THRESHOLD together.
HBI_MAX  = 3
PCSI_MAX = 1

# 1-STS converter. Instead of trained models, we use sentence-embedding
# similarity: each OUTPUT item is mapped to the single most cosine-similar INPUT
# item, and its prediction is that input item's entered score, binarized.
EMBED_PATH = "embed_dict.p"

# Map the app's inventory names to the keys inside embed_dict.p.
# PARENT version -> *_p keys. Verify against your embed_dict.p; the startup check
# below prints the available keys if these don't match.
EMBED_KEYS = {
    "HBI":    "HBI_p",
    "M-PCSI": "PCSI_p",
}

# HBI (0–3) is "present" when score > 1; M-PCSI is already 0/1 (pass-through).
BINARIZE_THRESHOLD = 1


# ─────────────────────────────────────────────────────────────────────────────
# Load embeddings once
# ─────────────────────────────────────────────────────────────────────────────
def _load_embeddings():
    if not os.path.exists(EMBED_PATH):
        raise FileNotFoundError(f"Missing embeddings file: {EMBED_PATH}")
    with open(EMBED_PATH, "rb") as f:
        return pickle.load(f)          # {inventory_key: ndarray (n_items, dim)}

EMBED = _load_embeddings()


def _emb(inv):
    """Embedding matrix (n_items, dim) for an inventory, in text_dict item order."""
    key = EMBED_KEYS[inv]
    if key not in EMBED:
        raise KeyError(
            f"embed_dict.p has no key '{key}' for inventory '{inv}'. "
            f"Available keys: {list(EMBED.keys())}. Edit EMBED_KEYS to match."
        )
    arr = np.asarray(EMBED[key], dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"Embedding for '{key}' must be 2-D (n_items, dim); got shape {arr.shape}."
        )
    return arr


def _most_similar_idx(source_emb, target_vec):
    """Index of the source item whose embedding is most cosine-similar to target."""
    sims = (source_emb @ target_vec) / (
        np.linalg.norm(source_emb, axis=1) * np.linalg.norm(target_vec) + 1e-12)
    return int(np.argmax(sims))


# Cache the output-item -> input-item mapping per direction (depends only on the
# fixed embeddings, not on user scores).
_MATCH_CACHE = {}

def _matches(inv_input, inv_output):
    key = (inv_input, inv_output)
    if key not in _MATCH_CACHE:
        se, te = _emb(inv_input), _emb(inv_output)
        _MATCH_CACHE[key] = np.array(
            [_most_similar_idx(se, te[j]) for j in range(te.shape[0])], dtype=int)
    return _MATCH_CACHE[key]


def _binarize_score(raw, inv_input):
    """Binarize one entered score by the INPUT inventory's scale."""
    raw = int(raw)
    if inv_input == "M-PCSI":                  # already 0/1
        return int(raw > 0)
    return int(raw > BINARIZE_THRESHOLD)       # HBI 0-3 -> present if > 1


# ─────────────────────────────────────────────────────────────────────────────
# Conversion: 1-STS. Each output item copies its most-similar input item's
# (binarized) score. No training / no predict().
# ─────────────────────────────────────────────────────────────────────────────
def convert(inv_input, inv_output, scores):
    """Returns ({output_item_text: 0/1}, number_of_symptoms_present)."""
    scores = np.asarray(list(map(int, scores)), dtype=float)   # (n_input,)
    idx = _matches(inv_input, inv_output)                      # (n_output,) input-item idx
    preds = np.array([_binarize_score(scores[k], inv_input) for k in idx], dtype=int)

    out_rows = text_dict[inv_output]
    outdict = {out_rows[j]: int(preds[j]) for j in range(len(preds))}
    return outdict, int(preds.sum())


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
app_ui = ui.page_fluid(
    {"class": "container"},
    ui.include_css(CSS_PATH),
    ui.tags.h3(
        "Proof of Concept for Test Purposes Only: Symptom Inventories "
        "Calculator (parent version)",
        class_="app-heading",
    ),
    ui.tags.div(
        {"class": "file"},
        ui.tags.div(
            ui.input_select("input_name", "Inventory Input", ["HBI", "M-PCSI"]),
            class_="select_dropdown",
        ),
        ui.tags.div(
            ui.input_select("output_name", "Inventory Output", ["M-PCSI", "HBI"]),
            class_="select_dropdown",
        ),
        ui.input_action_button("convert", "Convert table"),
        ui.download_button("download_conversion", "Download Conversion"),
        ui.download_button("download_readme", "Download Readme"),
    ),
    ui.tags.div(
        {"class": "tables"},
        ui.tags.div({"class": "table_parent"}, ui.output_ui("input_table")),
        ui.tags.div({"class": "table_parent"}, ui.output_ui("output_table")),
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────────────────────────────────────
def server(input, output, session):
    num_rows = reactive.Value(0)

    def _max_for(inv):
        return HBI_MAX if inv == "HBI" else PCSI_MAX

    def _read_scores():
        return [input[f"number_{i + 1}"]() for i in range(num_rows.get())]

    def _validate(scores, inv_input, inv_output):
        """Return True if OK, else show a notification and return False."""
        mx = _max_for(inv_input)
        rng_msg = (f"Please enter scores between 0 and {mx}"
                   if inv_input == "HBI" else "Please input 0 or 1 in the fields")
        for s in scores:
            if s is None:
                ui.notification_show("Please fill out scores for all symptoms",
                                     duration=5, close_button=True, type="error")
                return False
            if s < 0 or s > mx:
                ui.notification_show(rng_msg, duration=5, close_button=True, type="error")
                return False
        if inv_input == inv_output:
            ui.notification_show("Please select a different output measure",
                                 duration=5, close_button=True, type="error")
            return False
        return True

    # ── Input table ──────────────────────────────────────────────────────────
    @output
    @render.ui
    @reactive.event(input.input_name)
    def input_table():
        inv_input = input.input_name()
        if not inv_input:
            return
        rows = text_dict[inv_input]
        mx = _max_for(inv_input)

        table_rows = []
        for i, row in enumerate(rows):
            table_rows.append(ui.tags.tr(
                ui.tags.td(row),
                ui.tags.td(
                    ui.input_numeric(f"number_{i + 1}", label="", value="", min=0, max=mx),
                    class_="row_numbers",
                ),
                class_="table_row",
            ))
        num_rows.set(len(rows))

        heading = (f"Enter integer values between 0 and {mx}"
                   if inv_input == "HBI" else "Enter 0 or 1")
        return ui.tags.table(
            ui.tags.legend(
                ui.tags.span(f"{inv_input} table", class_="table_title"),
                ui.tags.br(class_="br"),
                ui.tags.span(heading, class_="table_instr"),
                class_="table_heading",
            ),
            ui.tags.tbody(*table_rows),
        )

    # ── Output table ─────────────────────────────────────────────────────────
    @output
    @render.ui
    @reactive.event(input.convert)
    def output_table():
        inv_input, inv_output = input.input_name(), input.output_name()
        scores = _read_scores()
        if not _validate(scores, inv_input, inv_output):
            return

        outdict, total = convert(inv_input, inv_output, scores)
        denom = len(text_dict[inv_output])

        body = [
            ui.tags.tr(ui.tags.tr(ui.tags.td(title), ui.tags.td(score),
                                  class_="output_table_row"), class_="table_row")
            for title, score in outdict.items()
        ]
        body.append(ui.tags.tr(
            ui.tags.td(f"Symptoms present / {denom}"),
            ui.tags.td(total),
            class_="total_score_row",
        ))
        return ui.tags.table(
            ui.tags.legend(
                ui.tags.span(f"{inv_output} table", class_="table_title"),
                ui.tags.br(), class_="table_heading",
            ),
            ui.tags.div(
                ui.tags.span("Output symptoms", class_="table_instr"),
                ui.tags.span("Present (1) / Absent (0)", class_="table_instr"),
                class_="output_captions",
            ),
            ui.tags.br(class_="br"),
            ui.tags.tbody(*body),
        )

    # ── Downloads ────────────────────────────────────────────────────────────
    @session.download(filename="README.txt")
    def download_readme():
        return os.path.join(os.path.dirname(__file__), "README.txt")

    @session.download(filename="converted_table.csv")
    def download_conversion():
        inv_input, inv_output = input.input_name(), input.output_name()
        scores = _read_scores()
        if not _validate(scores, inv_input, inv_output):
            return

        outdict, total = convert(inv_input, inv_output, scores)
        denom = len(text_dict[inv_output])

        df = pd.DataFrame()
        for title, score in outdict.items():
            df.loc["scores", title] = score
        df.loc["scores", f"Symptoms present / {denom}"] = total
        return io.BytesIO(df.to_csv().encode("utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# Item text — PARENT versions.
# Order MUST match the embedding row order: text_dict[k][i] == embed row i
# (HBI -> embed_dict['HBI_p'] rows; M-PCSI -> embed_dict['PCSI_p'] rows).
# ─────────────────────────────────────────────────────────────────────────────
text_dict = {
    "HBI": [
        "Had trouble sustaining attention.",
        "Was easily distracted.",
        "Had difficulty concentrating.",
        "Had problems remembering what he/she was told.",
        "Had difficulty following directions.",
        "Tended to daydream.",
        "Got confused.",
        "Was forgetful.",
        "Had difficulty completing tasks.",
        "Had poor problem-solving skills.",
        "Had problems learning.",
        "Had headaches.",
        "Felt dizzy.",
        "Had a feeling that the room is spinning.",
        "Felt faint.",
        "Had blurred vision.",
        "Had double vision.",
        "Experienced nausea.",
        "Got tired a lot.",
        "Got tired easily.",
    ],
    "M-PCSI": [
        "Has your child been tired a lot in the last week?",
        "Has your child had headaches in the last week?",
        "Has your child had any trouble remembering things in the past week?",
        "Has bright light hurt your child's eyes in the last week?",
        "Has your child's head been dizzy in the past week?",
        "Has your child been cranky or irritable in the last week?",
        "Has your child felt nervous or scared in the last week?",
        "Has your child had trouble paying attention in the last week?",
        "Has your child been sad or depressed in the last week?",
        "Has it been hard for your child to think in the last week?",
        "Has your child had trouble seeing in the last week?",
        "Has loud noise hurt your child's ears in the last week?",
        "Has your child had trouble sleeping in the last week?",
        "Has your child been less interested in doing things in the last week?",
        "Has your child's personality seemed different in the last week?",
    ],
}

app = App(app_ui, server)