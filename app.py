
import os
import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

from model import MelaScanCNN


# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = os.path.join(
    "models",
    "mela_scan_model.pth"
)

CLASS_NAMES = [
    "Healthy",
    "Eczema"
]

IMAGE_SIZE = 160

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("=" * 60)
print("MELA-SCAN")
print("=" * 60)

print(
    f"Device: {DEVICE}"
)

print(
    f"Loading model: {MODEL_PATH}"
)


model = MelaScanCNN(
    num_classes=2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model.to(DEVICE)

model.eval()


print(
    "Model loaded successfully."
)

print("=" * 60)


# ============================================================
# PREDICTION
# ============================================================

def predict(image):

    if image is None:

        return {
            "Healthy": 0.0,
            "Eczema": 0.0
        }


    image = image.convert(
        "RGB"
    )


    image_tensor = transform(
        image
    )


    image_tensor = image_tensor.unsqueeze(
        0
    )


    image_tensor = image_tensor.to(
        DEVICE
    )


    with torch.no_grad():

        outputs = model(
            image_tensor
        )


        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]


    healthy_probability = float(
        probabilities[0].cpu()
    )


    eczema_probability = float(
        probabilities[1].cpu()
    )


    return {
        "Healthy": healthy_probability,
        "Eczema": eczema_probability
    }


# ============================================================
# ANALYSIS
# ============================================================

def analyze_image(image):

    if image is None:

        return (
            "## No image uploaded\n\n"
            "Please upload a skin image to begin.",
            {
                "Healthy": 0,
                "Eczema": 0
            }
        )


    probabilities = predict(
        image
    )


    healthy = probabilities[
        "Healthy"
    ]

    eczema = probabilities[
        "Eczema"
    ]


    if eczema > healthy:

        prediction = "Eczema"

        confidence = eczema

    else:

        prediction = "Healthy"

        confidence = healthy


    confidence_percent = (
        confidence * 100
    )


    result = f"""
## Analysis Complete

### Prediction

# {prediction}

**Model confidence: {confidence_percent:.2f}%**

---

### Class probabilities

| Class | Probability |
|---|---:|
| Healthy | {healthy * 100:.2f}% |
| Eczema | {eczema * 100:.2f}% |

---

### How to interpret this

The model predicts which of the two classes in
its training dataset most closely matches the
uploaded image.

A higher probability means the model produced
a stronger prediction for that class.

---

> ⚠️ **Important medical disclaimer**
>
> MELA-SCAN is an educational and competition
> prototype. It has not been clinically validated
> and is not a medical device.
>
> This prediction must **not** be used to diagnose
> a medical condition or make medical decisions.
>
> If you are concerned about a skin condition,
> consult a qualified healthcare professional.
"""

    return (
        result,
        probabilities
    )


# ============================================================
# EXAMPLE INFORMATION
# ============================================================

description = """
# 🩺 MELA-SCAN

### AI-Assisted Skin Image Classification

MELA-SCAN is a computer-vision prototype that uses
a **10.42-million-parameter convolutional neural
network** to classify skin images into two dataset
classes:

**Healthy** or **Eczema**

---

### How MELA-SCAN works

**Upload image**
↓
**Image preprocessing**
↓
**10.42M-parameter CNN**
↓
**Class probabilities**
↓
**Prediction**

---

### Model

- Architecture: Custom CNN
- Parameters: 10,420,738
- Input resolution: 160 × 160 pixels
- Classes: Healthy / Eczema
- Training images: 3,316

### Test-set performance

- Accuracy: **99.00%**
- Precision: **99.55%**
- Sensitivity: **98.22%**
- Specificity: **99.63%**
- F1 score: **98.88%**
- ROC-AUC: **99.93%**

These metrics describe performance on the project's
held-out test set and should not be interpreted as
clinical performance.
"""


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title="MELA-SCAN"
) as demo:

    gr.Markdown(
        description
    )


    gr.Markdown(
        "---"
    )


    with gr.Row():

        # ----------------------------------------------------
        # LEFT SIDE
        # ----------------------------------------------------

        with gr.Column():

            gr.Markdown(
                "### 1. Upload an image"
            )


            image_input = gr.Image(
                type="pil",
                label="Skin Image",
                height=400
            )


            analyze_button = gr.Button(
                "🔬 Analyze Image",
                variant="primary",
                size="lg"
            )


            clear_button = gr.ClearButton(
                components=[
                    image_input
                ],
                value="Clear"
            )


        # ----------------------------------------------------
        # RIGHT SIDE
        # ----------------------------------------------------

        with gr.Column():

            gr.Markdown(
                "### 2. MELA-SCAN analysis"
            )


            result_output = gr.Markdown(
                """
## Ready

Upload an image and click
**Analyze Image**.
"""
            )


            probability_output = gr.Label(
                num_top_classes=2,
                label="Model probabilities"
            )


    # --------------------------------------------------------
    # BUTTON ACTION
    # --------------------------------------------------------

    analyze_button.click(
        fn=analyze_image,

        inputs=image_input,

        outputs=[
            result_output,
            probability_output
        ]
    )


    gr.Markdown(
        "---"
    )


    gr.Markdown(
        """
## ⚠️ Responsible-use notice

MELA-SCAN is a student-built AI prototype created
for educational and competition purposes.

It is **not clinically validated**, is **not a
medical device**, and should not be used for
diagnosis, treatment, or medical decision-making.

The model classifies images according to the
Healthy and Eczema labels present in its dataset.
Real-world performance may differ substantially
from the reported test-set performance.
"""
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))