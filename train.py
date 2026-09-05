from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "dataset"

OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"

BEST_MODEL_PATH = MODEL_DIR / "mela_scan_model.pth"

IMAGE_SIZE = 160

BATCH_SIZE = 32

NUM_EPOCHS = 20

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0001

RANDOM_SEED = 42

NUM_WORKERS = 0

PATIENCE = 5

CLASS_NAMES = [
    "Healthy",
    "Eczema",
]

CLASS_TO_INDEX = {
    "Healthy": 0,
    "Eczema": 1,
}

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    """
    Set random seeds so that the experiment is reproducible.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET DISCOVERY
# ============================================================

def collect_image_paths():
    """
    Find every supported image inside:

        dataset/Healthy/
        dataset/Eczema/

    Returns a pandas DataFrame containing:

        filepath
        class_name
        label
    """

    records = []

    for class_name in CLASS_NAMES:

        class_directory = DATASET_DIR / class_name

        if not class_directory.exists():

            raise FileNotFoundError(
                f"Could not find dataset class folder:\n"
                f"{class_directory}\n\n"
                f"Expected structure:\n"
                f"MELA-SCAN/\n"
                f"├── dataset/\n"
                f"│   ├── Healthy/\n"
                f"│   └── Eczema/\n"
            )

        image_paths = [
            path
            for path in class_directory.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        ]

        for image_path in image_paths:

            records.append(
                {
                    "filepath": str(image_path.resolve()),
                    "class_name": class_name,
                    "label": CLASS_TO_INDEX[class_name],
                }
            )

    if len(records) == 0:

        raise RuntimeError(
            "No images were found in the dataset."
        )

    dataframe = pd.DataFrame(records)

    dataframe = dataframe.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    return dataframe


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_images(dataframe):
    """
    Verify that every image can actually be opened.
    """

    print("\nValidating images...")

    valid_records = []

    invalid_count = 0

    for index, row in dataframe.iterrows():

        image_path = Path(row["filepath"])

        try:

            with Image.open(image_path) as image:

                image.verify()

            valid_records.append(row)

        except Exception:

            invalid_count += 1

            print(
                f"WARNING: Invalid image skipped:\n"
                f"{image_path}"
            )

    dataframe = pd.DataFrame(valid_records)

    dataframe = dataframe.reset_index(drop=True)

    print(f"Valid images:   {len(dataframe)}")
    print(f"Invalid images: {invalid_count}")

    if len(dataframe) == 0:

        raise RuntimeError(
            "No valid images remain after validation."
        )

    return dataframe


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def create_splits(dataframe):
    """
    Create a stratified:

        70% training
        15% validation
        15% test

    split.

    Stratification keeps the Healthy/Eczema proportions
    approximately equal between the three datasets.
    """

    train_dataframe, temporary_dataframe = train_test_split(
        dataframe,
        test_size=0.30,
        stratify=dataframe["label"],
        random_state=RANDOM_SEED,
    )

    validation_dataframe, test_dataframe = train_test_split(
        temporary_dataframe,
        test_size=0.50,
        stratify=temporary_dataframe["label"],
        random_state=RANDOM_SEED,
    )

    train_dataframe = train_dataframe.reset_index(drop=True)

    validation_dataframe = validation_dataframe.reset_index(drop=True)

    test_dataframe = test_dataframe.reset_index(drop=True)

    return (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    )


# ============================================================
# DATASET CLASS
# ============================================================

class SkinDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.dataframe = dataframe.reset_index(drop=True)

        self.transform = transform

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_path = row["filepath"]

        label = int(row["label"])

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:

            image = self.transform(image)

        return image, label


# ============================================================
# DATA TRANSFORMS
# ============================================================

def create_transforms():

    training_transform = transforms.Compose(
        [

            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.RandomVerticalFlip(
                p=0.2
            ),

            transforms.RandomRotation(
                degrees=15
            ),

            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.03,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    evaluation_transform = transforms.Compose(
        [

            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    return (
        training_transform,
        evaluation_transform,
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(dataframe):

    class_counts = (
        dataframe["label"]
        .value_counts()
        .sort_index()
    )

    total = len(dataframe)

    number_of_classes = len(CLASS_NAMES)

    weights = []

    for class_index in range(number_of_classes):

        count = class_counts.get(
            class_index,
            0
        )

        if count == 0:

            weights.append(0.0)

        else:

            weight = total / (
                number_of_classes * count
            )

            weights.append(weight)

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    true_labels,
    predicted_labels,
    probabilities,
):
    """
    Calculate binary classification metrics.

    Eczema (class 1) is treated as the positive class.
    """

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        pos_label=1,
        zero_division=0,
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        pos_label=1,
        zero_division=0,
    )

    confusion = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1],
    )

    true_negative = confusion[0, 0]

    false_positive = confusion[0, 1]

    false_negative = confusion[1, 0]

    true_positive = confusion[1, 1]

    specificity_denominator = (
        true_negative + false_positive
    )

    if specificity_denominator > 0:

        specificity = (
            true_negative
            / specificity_denominator
        )

    else:

        specificity = 0.0

    try:

        roc_auc = roc_auc_score(
            true_labels,
            probabilities,
        )

    except ValueError:

        roc_auc = 0.0

    metrics = {

        "accuracy": float(accuracy),

        "precision": float(precision),

        "recall_sensitivity": float(recall),

        "specificity": float(specificity),

        "f1_score": float(f1),

        "roc_auc": float(roc_auc),

        "true_negative": int(true_negative),

        "false_positive": int(false_positive),

        "false_negative": int(false_negative),

        "true_positive": int(true_positive),

    }

    return metrics, confusion


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    dataloader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    true_labels = []

    predicted_labels = []

    probabilities = []

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item()
                * images.size(0)
            )

            probabilities_batch = torch.softmax(
                outputs,
                dim=1
            )[:, 1]

            predictions_batch = (
                outputs.argmax(dim=1)
            )

            true_labels.extend(
                labels.cpu().numpy()
            )

            predicted_labels.extend(
                predictions_batch.cpu().numpy()
            )

            probabilities.extend(
                probabilities_batch.cpu().numpy()
            )

    average_loss = (
        total_loss
        / len(dataloader.dataset)
    )

    metrics, confusion = calculate_metrics(
        true_labels,
        predicted_labels,
        probabilities,
    )

    return (
        average_loss,
        metrics,
        confusion,
        np.array(true_labels),
        np.array(probabilities),
    )


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    confusion,
    output_path,
):

    figure = plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        confusion,
        interpolation="nearest",
    )

    plt.title(
        "MELA-SCAN Confusion Matrix"
    )

    plt.colorbar()

    tick_marks = np.arange(
        len(CLASS_NAMES)
    )

    plt.xticks(
        tick_marks,
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        tick_marks,
        CLASS_NAMES,
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "True Class"
    )

    threshold = confusion.max() / 2

    for row in range(
        confusion.shape[0]
    ):

        for column in range(
            confusion.shape[1]
        ):

            plt.text(
                column,
                row,
                str(confusion[row, column]),
                horizontalalignment="center",
                verticalalignment="center",
                color=(
                    "white"
                    if confusion[row, column]
                    > threshold
                    else "black"
                ),
                fontsize=14,
            )

    plt.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# PLOT TRAINING CURVES
# ============================================================

def save_training_curves(
    history,
    output_path,
):

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    figure = plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        epochs,
        history["train_loss"],
        label="Training Loss",
    )

    plt.plot(
        epochs,
        history["validation_loss"],
        label="Validation Loss",
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title(
        "MELA-SCAN Training and Validation Loss"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# PLOT ROC CURVE
# ============================================================

def save_roc_curve(
    true_labels,
    probabilities,
    output_path,
):

    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            true_labels,
            probabilities,
        )
    )

    auc_score = roc_auc_score(
        true_labels,
        probabilities,
    )

    figure = plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {auc_score:.4f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "MELA-SCAN ROC Curve"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    set_seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)

    print(
        "MELA-SCAN TRAINING"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(
            f"\nDevice: GPU"
        )

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    else:

        device = torch.device("cpu")

        print(
            "\nDevice: CPU"
        )

        print(
            "No CUDA-compatible GPU was detected."
        )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print(
        "\nSearching for images..."
    )

    dataframe = collect_image_paths()

    print(
        f"Images discovered: {len(dataframe)}"
    )

    dataframe = validate_images(
        dataframe
    )

    print(
        "\nFinal dataset:"
    )

    print(
        dataframe["class_name"]
        .value_counts()
    )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    ) = create_splits(dataframe)

    print(
        "\nDataset split:"
    )

    print(
        f"Training:   {len(train_dataframe)}"
    )

    print(
        f"Validation: {len(validation_dataframe)}"
    )

    print(
        f"Test:       {len(test_dataframe)}"
    )

    print(
        "\nTraining class distribution:"
    )

    print(
        train_dataframe["class_name"]
        .value_counts()
    )

    print(
        "\nValidation class distribution:"
    )

    print(
        validation_dataframe["class_name"]
        .value_counts()
    )

    print(
        "\nTest class distribution:"
    )

    print(
        test_dataframe["class_name"]
        .value_counts()
    )

    # --------------------------------------------------------
    # SAVE SPLIT SUMMARY
    # --------------------------------------------------------

    split_records = []

    for split_name, split_dataframe in [

        ("train", train_dataframe),

        ("validation", validation_dataframe),

        ("test", test_dataframe),

    ]:

        for _, row in split_dataframe.iterrows():

            split_records.append(
                {
                    "filepath": row["filepath"],
                    "class_name": row["class_name"],
                    "label": row["label"],
                    "split": split_name,
                }
            )

    split_dataframe_output = pd.DataFrame(
        split_records
    )

    split_dataframe_output.to_csv(
        OUTPUT_DIR / "split_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # TRANSFORMS
    # --------------------------------------------------------

    (
        training_transform,
        evaluation_transform,
    ) = create_transforms()

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = SkinDataset(
        train_dataframe,
        transform=training_transform,
    )

    validation_dataset = SkinDataset(
        validation_dataframe,
        transform=evaluation_transform,
    )

    test_dataset = SkinDataset(
        test_dataframe,
        transform=evaluation_transform,
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    from model import (
        create_model,
        count_parameters,
    )

    model = create_model(
        num_classes=2
    )

    model = model.to(device)

    parameter_count = count_parameters(
        model
    )

    print(
        f"\nTrainable parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        train_dataframe
    )

    class_weights = class_weights.to(
        device
    )

    print(
        "\nClass weights:"
    )

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{class_name}: "
            f"{class_weights[index].item():.4f}"
        )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # LEARNING RATE SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    # --------------------------------------------------------
    # TRAINING HISTORY
    # --------------------------------------------------------

    history = {

        "train_loss": [],

        "validation_loss": [],

        "validation_accuracy": [],

        "validation_recall": [],

        "validation_f1": [],

        "validation_roc_auc": [],

    }

    best_validation_auc = -1.0

    best_epoch = 0

    epochs_without_improvement = 0

    training_start_time = time.time()

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STARTING TRAINING"
    )

    print(
        "=" * 70
    )

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        epoch_start_time = time.time()

        model.train()

        running_loss = 0.0

        correct_predictions = 0

        total_samples = 0

        for images, labels in train_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_samples += (
                labels.size(0)
            )

        training_loss = (
            running_loss
            / len(train_loader.dataset)
        )

        training_accuracy = (
            correct_predictions
            / total_samples
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        (
            validation_loss,
            validation_metrics,
            _,
            _,
            _,
        ) = evaluate_model(
            model,
            validation_loader,
            criterion,
            device,
        )

        validation_accuracy = (
            validation_metrics["accuracy"]
        )

        validation_recall = (
            validation_metrics[
                "recall_sensitivity"
            ]
        )

        validation_f1 = (
            validation_metrics[
                "f1_score"
            ]
        )

        validation_auc = (
            validation_metrics[
                "roc_auc"
            ]
        )

        scheduler.step(
            validation_auc
        )

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        history["train_loss"].append(
            training_loss
        )

        history["validation_loss"].append(
            validation_loss
        )

        history["validation_accuracy"].append(
            validation_accuracy
        )

        history["validation_recall"].append(
            validation_recall
        )

        history["validation_f1"].append(
            validation_f1
        )

        history["validation_roc_auc"].append(
            validation_auc
        )

        epoch_time = (
            time.time()
            - epoch_start_time
        )

        current_lr = optimizer.param_groups[0][
            "lr"
        ]

        print(
            f"\nEpoch "
            f"{epoch}/{NUM_EPOCHS}"
        )

        print(
            f"Training loss: "
            f"{training_loss:.4f}"
        )

        print(
            f"Training accuracy: "
            f"{training_accuracy:.4f}"
        )

        print(
            f"Validation loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation accuracy: "
            f"{validation_accuracy:.4f}"
        )

        print(
            f"Validation recall: "
            f"{validation_recall:.4f}"
        )

        print(
            f"Validation F1: "
            f"{validation_f1:.4f}"
        )

        print(
            f"Validation ROC-AUC: "
            f"{validation_auc:.4f}"
        )

        print(
            f"Learning rate: "
            f"{current_lr:.6f}"
        )

        print(
            f"Epoch time: "
            f"{epoch_time:.1f} seconds"
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if validation_auc > best_validation_auc:

            best_validation_auc = (
                validation_auc
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            checkpoint = {

                "model_state_dict":
                    model.state_dict(),

                "class_names":
                    CLASS_NAMES,

                "image_size":
                    IMAGE_SIZE,

                "parameter_count":
                    parameter_count,

                "best_validation_auc":
                    best_validation_auc,

                "epoch":
                    epoch,

            }

            torch.save(
                checkpoint,
                BEST_MODEL_PATH,
            )

            print(
                "✓ New best model saved."
            )

        else:

            epochs_without_improvement += 1

            print(
                f"No improvement for "
                f"{epochs_without_improvement} "
                f"epoch(s)."
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    # --------------------------------------------------------
    # TRAINING COMPLETE
    # --------------------------------------------------------

    total_training_time = (
        time.time()
        - training_start_time
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation ROC-AUC: "
        f"{best_validation_auc:.4f}"
    )

    print(
        f"Training time: "
        f"{total_training_time / 60:.2f} minutes"
    )

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    print(
        "\nLoading best model..."
    )

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------
    # FINAL TEST EVALUATION
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL TEST EVALUATION"
    )

    print(
        "=" * 70
    )

    (
        test_loss,
        test_metrics,
        test_confusion,
        test_true_labels,
        test_probabilities,
    ) = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
    )

    print(
        f"\nTest loss: "
        f"{test_loss:.4f}"
    )

    print(
        f"Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall / Sensitivity: "
        f"{test_metrics['recall_sensitivity']:.4f}"
    )

    print(
        f"Specificity: "
        f"{test_metrics['specificity']:.4f}"
    )

    print(
        f"F1 score: "
        f"{test_metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        test_confusion
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    final_metrics = {

        "project": "MELA-SCAN",

        "task": "Healthy vs Eczema",

        "dataset": (
            "Skin Diseases Identification"
        ),

        "device": str(device),

        "parameter_count": int(
            parameter_count
        ),

        "image_size": IMAGE_SIZE,

        "batch_size": BATCH_SIZE,

        "epochs_requested": NUM_EPOCHS,

        "best_epoch": best_epoch,

        "training_images": len(
            train_dataframe
        ),

        "validation_images": len(
            validation_dataframe
        ),

        "test_images": len(
            test_dataframe
        ),

        "training_time_minutes": (
            total_training_time / 60
        ),

        "test_loss": float(
            test_loss
        ),

        **test_metrics,

    }

    with open(
        OUTPUT_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            final_metrics,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # SAVE GRAPHS
    # --------------------------------------------------------

    save_confusion_matrix(
        test_confusion,
        OUTPUT_DIR
        / "confusion_matrix.png",
    )

    save_training_curves(
        history,
        OUTPUT_DIR
        / "training_curves.png",
    )

    save_roc_curve(
        test_true_labels,
        test_probabilities,
        OUTPUT_DIR
        / "roc_curve.png",
    )

    # --------------------------------------------------------
    # SAVE TRAINING HISTORY
    # --------------------------------------------------------

    with open(
        OUTPUT_DIR / "training_history.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # FINAL MESSAGE
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MELA-SCAN TRAINING FINISHED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        "\nModel saved to:"
    )

    print(
        BEST_MODEL_PATH
    )

    print(
        "\nOutputs saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nThe next step will be creating the Gradio application."
    )


if __name__ == "__main__":

    main()