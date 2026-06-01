import modal

app = modal.App("pbcell-training-example")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "pandas",
        "datasets",
    )
    .add_local_dir(
        "./src",
        remote_path="/root/src",
    )
)

volume = modal.Volume.from_name(
    "training-results",
    create_if_missing=True,
)


@app.function(
    gpu="A100",
    timeout=14400,
    image=image,
    volumes={"/mnt/data": volume},
)
def train():

    import os

    command = (
        "accelerate launch "
        "/root/src/pbcell/train/train.py "
        "--config configs/example.yaml"
    )

    os.system(command)


@app.local_entrypoint()
def main():

    train.remote()