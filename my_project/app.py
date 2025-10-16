import gradio as gr
import pytorch_lightning as pl
import torch
import pandas as pd

from my_project.config import TEST_CSV_PATH, TRAIN_CSV_PATH
from my_project.dataset import FashionMNISTCSVDataModule
from my_project.model import Net
from my_project.plots import (
    evaluate_and_plot,
    FASHION_CLASSES,
    plot_class_distribution,
    get_sample_images_for_gallery,
)

# --- Pre-load data for exploration to speed up the UI ---
train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)


def update_data_exploration(dataset_choice, class_filter):
    """
    Updates the components in the Data Exploration tab based on user selection.
    """
    df = train_df if dataset_choice == "Train" else test_df

    # 1. Update the class distribution plot
    dist_plot_fig = plot_class_distribution(df, FASHION_CLASSES)

    # 2. Update the gallery
    df_to_sample = df
    if class_filter != "All":
        class_index = FASHION_CLASSES.index(class_filter)
        df_to_sample = df[df["label"] == class_index]

    gallery_images = get_sample_images_for_gallery(
        df_to_sample, FASHION_CLASSES, n_samples=20
    )

    # 3. Update statistics text
    stats_md = f"""
    ### Dataset Statistics
    - **Selected Set:** {dataset_choice}
    - **Total Samples:** {len(df)}
    - **Number of Classes:** {len(FASHION_CLASSES)}
    - **Image Size:** 28x28 pixels (grayscale)
    """

    return stats_md, dist_plot_fig, gallery_images



def train_and_evaluate(
    batch_size: int,
    max_epochs: int,
    lr: float,
    num_filters: int,
    hidden_size: int,
    progress=gr.Progress(track_tqdm=True),
):
    """
    A function to train and evaluate the model with given hyperparameters.
    This will be connected to the Gradio interface.
    """
    progress(0, desc="Initializing DataModule...")
    datamodule = FashionMNISTCSVDataModule(
        train_csv=TRAIN_CSV_PATH,
        test_csv=TEST_CSV_PATH,
        batch_size=int(batch_size),
    )

    progress(0.1, desc="Initializing Model...")
    model = Net(num_filters=int(num_filters), hidden_size=int(hidden_size), lr=lr)

    # Use a simple callback to update progress
    class GradioProgressCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            progress(
                trainer.current_epoch / trainer.max_epochs,
                desc=f"Epoch {trainer.current_epoch+1}/{trainer.max_epochs}",
            )

    trainer = pl.Trainer(
        max_epochs=int(max_epochs),
        accelerator="gpu",  # Explicitly use GPU
        devices="auto",
        logger=False,  # Disable logging for the demo
        callbacks=[GradioProgressCallback()],
        enable_checkpointing=False,
    )

    progress(0.2, desc="Starting Training...")
    trainer.fit(model, datamodule=datamodule)

    progress(0.9, desc="Evaluating on Test Set...")
    # Manually call setup for the 'test' stage to ensure test_ds is initialized.
    datamodule.setup(stage="test")

    # The evaluate_and_plot function already returns the paths to the plots
    artifacts = evaluate_and_plot(model, datamodule, out_dir="reports/figures/gradio")

    return (
        f"{artifacts['test_accuracy']:.4f}",
        artifacts["confusion_matrix"],
        artifacts["per_class_accuracy"],
        artifacts["misclassified_grid"],
        artifacts["calibration_curve"],
    )


# We define the Gradio interface using Blocks for more control.
with gr.Blocks() as demo:
    gr.Markdown("# Fashion-MNIST Interactive Demo")

    with gr.Tab("Data Exploration"):
        gr.Markdown("## Exploración Interactiva del Dataset Fashion-MNIST")
        gr.Markdown(
            "Usa los controles para cambiar entre los datos de entrenamiento (Train) y prueba (Test), o para filtrar las imágenes por clase."
        )

        with gr.Row():
            dataset_selector = gr.Radio(
                ["Train", "Test"], value="Train", label="Seleccionar Dataset"
            )
            class_selector = gr.Dropdown(
                ["All"] + FASHION_CLASSES, value="All", label="Filtrar por Clase"
            )
            refresh_button = gr.Button("Refrescar Imágenes")

        with gr.Row():
            stats_md_box = gr.Markdown(
                value=f"""
                ### Dataset Statistics
                - **Selected Set:** Train
                - **Total Samples:** {len(train_df)}
                - **Number of Classes:** {len(FASHION_CLASSES)}
                - **Image Size:** 28x28 pixels (grayscale)
                """
            )
            dist_plot = gr.Plot()

        gr.Markdown("### Muestras de Imágenes")
        gallery = gr.Gallery(
            label="Muestras aleatorias del dataset",
            columns=10,
            object_fit="contain",
            height="auto",
        )

        # --- Event Listeners for Interactivity ---
        # Combine selectors to a single update function for efficiency
        controls = [dataset_selector, class_selector]
        outputs = [stats_md_box, dist_plot, gallery]

        for control in controls:
            control.change(fn=update_data_exploration, inputs=controls, outputs=outputs)
        refresh_button.click(
            fn=update_data_exploration, inputs=controls, outputs=outputs
        )

    with gr.Tab("Train & Evaluate"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Hyperparameters")
                batch_size_slider = gr.Slider(
                    32, 512, value=128, step=32, label="Batch Size"
                )
                epochs_slider = gr.Slider(1, 20, value=5, step=1, label="Max Epochs")
                lr_slider = gr.Slider(
                    1e-5, 1e-2, value=1e-3, label="Learning Rate", step=1e-5
                )
                num_filters_slider = gr.Slider(
                    8, 64, value=32, step=8, label="Conv Filters"
                )
                hidden_size_slider = gr.Slider(
                    32, 256, value=64, step=32, label="Hidden Layer Size"
                )
                train_button = gr.Button("Start Training", variant="primary")

            with gr.Column(scale=3):
                gr.Markdown("## Evaluation Results")
                test_acc_box = gr.Textbox(label="Test Accuracy")
                with gr.Row():
                    cm_plot = gr.Image(label="Confusion Matrix")
                    pca_plot = gr.Image(label="Per-Class Accuracy")
                misclassified_plot = gr.Image(label="Misclassified Samples")
                calibration_plot = gr.Image(label="Calibration Curve")

    train_button.click(
        fn=train_and_evaluate,
        inputs=[
            batch_size_slider,
            epochs_slider,
            lr_slider,
            num_filters_slider,
            hidden_size_slider,
        ],
        outputs=[
            test_acc_box,
            cm_plot,
            pca_plot,
            misclassified_plot,
            calibration_plot,
        ],
    )
    # Initial load for the data exploration tab
    demo.load(fn=update_data_exploration, inputs=[gr.State("Train"), gr.State("All")], outputs=[stats_md_box, dist_plot, gallery])

if __name__ == "__main__":
    demo.launch()