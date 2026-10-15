"""Export the trained M3 classifier to web/model.json for the browser demo.

Usage (e.g. in Colab):  python export_model.py /content/drive/MyDrive/NLP_capstone_final/intent_clf_M3.joblib
"""
import json
import sys

import joblib
import numpy as np

bundle = joblib.load(sys.argv[1] if len(sys.argv) > 1 else "intent_clf_M3.joblib")
clf = bundle["clf"]

# The browser computes softmax(x @ coef.T + intercept); check that this matches sklearn's predict_proba.
x = np.random.default_rng(0).normal(size=(3, clf.coef_.shape[1]))
z = x @ clf.coef_.T + clf.intercept_
p = np.exp(z - z.max(axis=1, keepdims=True))
p /= p.sum(axis=1, keepdims=True)
assert np.allclose(p, clf.predict_proba(x), atol=1e-6), "classifier is not multinomial softmax"

model = {
    "classes": [str(c) for c in clf.classes_],
    "coef": [[round(float(v), 6) for v in row] for row in clf.coef_],
    "intercept": [round(float(v), 6) for v in clf.intercept_],
    "tau": float(bundle["tau"]),
    "encoder": "Xenova/multilingual-e5-small",
}
with open("model.json", "w") as f:
    json.dump(model, f, separators=(",", ":"))
print(f"✅ model.json written: {len(model['classes'])} intents x {len(model['coef'][0])} dims, tau={model['tau']}")
