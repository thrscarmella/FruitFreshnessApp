from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io
import os

app = Flask(__name__, template_folder='../templates')

# ── Absolute base path (works on Vercel's serverless environment) ──────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Load your trained freshness model ─────────────────────────────────────────
model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, 'model', 'fruit_freshness_model.keras')
)

# ── Load class names ───────────────────────────────────────────────────────────
with open(os.path.join(BASE_DIR, 'class_names.json'), 'r') as f:
    class_names = json.load(f)

# ── Gatekeeper model (MobileNetV2) ─────────────────────────────────────────────
# Cache dir set explicitly so Vercel's /tmp is used for downloaded weights
os.environ['KERAS_HOME'] = '/tmp/.keras'
os.environ['TFHUB_CACHE_DIR'] = '/tmp/tfhub'

gate_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=True
)

SUPPORTED_FRUITS = ['apple', 'banana', 'orange']

# ── Gatekeeper label groups ────────────────────────────────────────────────────
APPLE_LABELS  = {'Granny_Smith', 'fig', 'pomegranate', 'hip', 'buckeye'}
BANANA_LABELS = {'banana', 'slug'}
ORANGE_LABELS = {'orange', 'lemon', 'lime'}

ALL_ALLOWED = APPLE_LABELS | BANANA_LABELS | ORANGE_LABELS

BLOCKED_LABELS = {
    'watermelon', 'pineapple', 'strawberry', 'grape', 'mango',
    'jackfruit', 'durian', 'pomelo', 'grapefruit', 'guava',
    'papaya', 'pear', 'peach', 'plum', 'cherry', 'coconut',
    'melon', 'cantaloupe', 'honeydew'
}

# ── Thresholds (unchanged) ─────────────────────────────────────────────────────
GATE_MIN_CONF      = 0.15
FRESHNESS_MIN_CONF = 60.0
FRESHNESS_LOW_CONF = 40.0


def check_gatekeeper(img_pil):
    """
    Two-step gate:
    1. If top-10 contains a BLOCKED label with >20% → reject
    2. If top-10 contains an ALLOWED label with >15% → pass
    Otherwise → reject
    """
    img = img_pil.resize((224, 224))
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.array(img, dtype=np.float32)
    )
    img_array = np.expand_dims(img_array, axis=0)

    preds = gate_model.predict(img_array, verbose=0)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=10)[0]

    print("Gatekeeper top-10:", [(lbl, f"{sc:.3f}") for _, lbl, sc in decoded])

    # Step 1: Block look-alike fruits first
    for (_, label, score) in decoded:
        if label in BLOCKED_LABELS and score >= 0.20:
            return False, label, float(score)

    # Step 2: Pass if any allowed label found
    for (_, label, score) in decoded:
        if label in ALL_ALLOWED and score >= GATE_MIN_CONF:
            return True, label, float(score)

    return False, decoded[0][1], float(decoded[0][2])


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    # ── Gate check ────────────────────────────────────────────────────────────
    passed, gate_label, gate_score = check_gatekeeper(img)

    if not passed:
        return jsonify({
            'fruit': 'Unknown',
            'status': 'invalid',
            'confidence': f"{gate_score * 100:.2f}%",
            'message': 'This does not appear to be an apple, banana, or orange!',
            'warning': False
        })

    # ── Freshness model ───────────────────────────────────────────────────────
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index]) * 100

    if confidence < FRESHNESS_LOW_CONF:
        return jsonify({
            'fruit': 'Unknown',
            'status': 'invalid',
            'confidence': f"{confidence:.2f}%",
            'message': 'This does not appear to be an apple, banana, or orange!',
            'warning': False
        })

    status = 'fresh' if 'fresh' in predicted_class.lower() else 'rotten'
    fruit  = (predicted_class
              .replace('fresh', '')
              .replace('rotten', '')
              .strip()
              .capitalize())

    # ── Low confidence → show result with warning ─────────────────────────────
    if confidence < FRESHNESS_MIN_CONF:
        return jsonify({
            'fruit': fruit,
            'status': status,
            'confidence': f"{confidence:.2f}%",
            'message': '',
            'warning': True,
            'warning_message': f'Low confidence ({confidence:.1f}%) — result may not be accurate. Try a clearer, well-lit image.'
        })

    return jsonify({
        'fruit': fruit,
        'status': status,
        'confidence': f"{confidence:.2f}%",
        'message': '',
        'warning': False
    })