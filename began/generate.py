import torch
from began.models.skip import SkipDecoder
from torchvision.utils import make_grid, save_image
import cv2
import torchvision.transforms as transforms
import os


base_model_path = "models/device_cuda_manual_seed_84_save_step_2000_img_size_32_workers_dl_10_batch_size_64_lr_0.0001_beta1_0.5_lr_step_5000_lr_gamma_0.95_gamma_0.5_lambda_k_0.001_skip_True_n_filters_64_max_iter_20000"


def generate_photo(
    model_folder=base_model_path,
    model=SkipDecoder,
    kwargs=dict(n_filters=64),
):
    models_path = [
        x
        for x in sorted(
            os.listdir(model_folder),
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )
        if x.endswith(".pth")
    ]

    models = [torch.load(os.path.join(model_folder, x)) for x in models_path]

    noise = torch.randn(64, 64)

    gen = model(**kwargs)

    examples = []
    for model in models:
        gen.load_state_dict(model["generator_state_dict"])
        gen.eval()

        samples = gen(noise)
        new_samples = [
            transforms.ToTensor()(
                cv2.cvtColor(
                    (x.detach().permute(1, 2, 0).numpy() * 255),
                    cv2.COLOR_BGR2RGB,
                ).astype(int)
                / 255
            )
            for x in samples
        ]
        examples.extend(new_samples[:8])

    nrow = len(examples) // len(models)

    grid = make_grid(examples, nrow=nrow, padding=2, normalize=False)

    save_image(grid, "gen.jpg")


if __name__ == "__main__":
    generate_photo()
