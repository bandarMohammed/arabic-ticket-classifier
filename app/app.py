"""Gradio demo: Arabic-aware support ticket classifier (SDAIA Applied NLP capstone).

Pipeline:  text -> multilingual-e5-small embedding (frozen) -> logistic regression (M3)
           -> intent + confidence -> route (auto / human review) + category + priority

Needs `intent_clf_M3.joblib` (saved by notebooks/capstone_final.ipynb, section 10) next to this file.
Run locally:  pip install -r requirements.txt && python app.py
"""

import re
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
bundle = joblib.load(HERE / "intent_clf_M3.joblib")
CLF, TAU = bundle["clf"], float(bundle["tau"])
ENCODER = SentenceTransformer(bundle.get("encoder", "intfloat/multilingual-e5-small"), device="cpu")
PREFIX = bundle.get("prefix", "query: ")

# Business rules copied from the course (labs/prepare_data.py): category per intent, priority per intent + urgency.
CATEGORY = {
    "cancel_order": "ORDER", "change_order": "ORDER", "place_order": "ORDER", "track_order": "ORDER",
    "change_shipping_address": "SHIPPING", "set_up_shipping_address": "SHIPPING",
    "check_cancellation_fee": "CANCEL",
    "check_invoice": "INVOICE", "get_invoice": "INVOICE",
    "check_payment_methods": "PAYMENT", "payment_issue": "PAYMENT",
    "check_refund_policy": "REFUND", "get_refund": "REFUND", "track_refund": "REFUND",
    "complaint": "FEEDBACK", "review": "FEEDBACK",
    "contact_customer_service": "CONTACT", "contact_human_agent": "CONTACT",
    "create_account": "ACCOUNT", "delete_account": "ACCOUNT", "edit_account": "ACCOUNT",
    "recover_password": "ACCOUNT", "registration_problems": "ACCOUNT", "switch_account": "ACCOUNT",
    "delivery_options": "DELIVERY", "delivery_period": "DELIVERY",
    "newsletter_subscription": "SUBSCRIPTION",
}
PRIORITIES = ["low", "medium", "high", "urgent"]
BASE_PRIORITY = {
    "payment_issue": "high", "complaint": "high",
    "get_refund": "medium", "track_refund": "medium", "cancel_order": "medium", "change_order": "medium",
    "change_shipping_address": "medium", "track_order": "medium", "recover_password": "medium",
    "registration_problems": "medium", "delete_account": "medium", "contact_human_agent": "medium",
}
URGENCY_EN = re.compile(r"\b(urgent|urgently|asap|immediately|right now|now!|at once|emergency|furious|"
                        r"unacceptable|ridiculous|worst|scam|fraud|lawyer|sue)\b|!!", re.IGNORECASE)
URGENCY_AR = re.compile(r"(عاجل|فورا|فوراً|حالا|حالاً|بسرعة|الحين|!!|شي ما يصير|سيئة)")


def priority_for(intent: str, text: str) -> str:
    level = PRIORITIES.index(BASE_PRIORITY.get(intent, "low"))
    if URGENCY_EN.search(text) or URGENCY_AR.search(text):
        level = min(level + 1, len(PRIORITIES) - 1)
    return PRIORITIES[level]


def classify(text: str, tau: float):
    text = (text or "").strip()
    if not text:
        return {}, "—", "—", "—", "—"
    x = ENCODER.encode([PREFIX + text], normalize_embeddings=True)
    probs = CLF.predict_proba(x)[0]
    order = np.argsort(-probs)
    top = {str(CLF.classes_[i]): float(probs[i]) for i in order[:3]}
    intent, conf = str(CLF.classes_[order[0]]), float(probs[order[0]])
    lang = "Arabic · عربي" if re.search(r"[؀-ۿ]", text) else "English"
    if conf < tau:
        route = f"🧑‍💼 Human review — confidence {conf:.0%} < τ {tau:.0%}"
    else:
        route = f"🤖 Auto-route — confidence {conf:.0%} ≥ τ {tau:.0%}"
    return top, route, CATEGORY.get(intent, "—"), priority_for(intent, text), lang


EXAMPLES = [
    ["متى يوصل طلبي؟ ابي رقم التتبع"],
    ["تجربة سيئة مرة، المنتج وصل مكسور وما احد رد علي"],
    ["حولني على أحد من الموظفين"],
    ["احذفوا حسابي لو سمحتوا ما عاد ابيه"],
    ["فشلت عملية الدفع عدة مرات، ما المشكلة؟"],
    ["I was charged twice for order A-55821, fix it now!"],
    ["اقدر ادفع بأبل باي؟"],  # a known failure (coverage: brand name unseen in training)
]

with gr.Blocks(title="Arabic-Aware Ticket Classifier") as demo:
    gr.Markdown(
        "# Arabic-Aware Support Ticket Classifier · مصنّف تذاكر الدعم\n"
        "Type a customer message in **English, Arabic (MSA) or Gulf dialect**. The model picks one of "
        "**27 intents**; tickets below the confidence threshold τ go to a human.\n\n"
        "`multilingual-e5-small` (frozen) → logistic regression trained on English + synthetic Arabic · "
        "Arabic macro-F1 **0.26 → 0.73** vs the course baseline."
    )
    with gr.Row():
        with gr.Column(scale=3):
            inp = gr.Textbox(label="Customer message · رسالة العميل", lines=3, placeholder="وين طلبي؟")
            tau = gr.Slider(0.0, 0.9, value=TAU, step=0.05, label="Confidence threshold τ (model default from validation)")
            btn = gr.Button("Classify · صنّف", variant="primary")
            gr.Examples(EXAMPLES, inputs=inp)
        with gr.Column(scale=2):
            out_top = gr.Label(num_top_classes=3, label="Intent (top 3)")
            out_route = gr.Textbox(label="Route")
            with gr.Row():
                out_cat = gr.Textbox(label="Category")
                out_pri = gr.Textbox(label="Priority")
                out_lang = gr.Textbox(label="Language")
    outs = [out_top, out_route, out_cat, out_pri, out_lang]
    btn.click(classify, [inp, tau], outs)
    inp.submit(classify, [inp, tau], outs)

if __name__ == "__main__":
    demo.launch()
