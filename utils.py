from pathlib import Path
import torch
from anomalib.data.utils import download_and_extract, DownloadInfo
from anomalib.data.image import Visa
from torch_model import PDNModel

WEIGHTS_DOWNLOAD_INFO = DownloadInfo(
    name="efficientad_pretrained_weights.zip",
    url="https://github.com/openvinotoolkit/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip",
    hashsum="c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a",
)


def load_pdn(path: str | Path = "./pre_trained/") -> torch.nn.Module:
    """Prepare the pretrained feature extractor"""
    pretrained_models_dir = Path(path)
    if not (pretrained_models_dir / "efficientad_pretrained_weights").is_dir():
        download_and_extract(pretrained_models_dir, WEIGHTS_DOWNLOAD_INFO)
    model = PDNModel(384)

    teacher_path = (
        pretrained_models_dir
        / "efficientad_pretrained_weights"
        / "pretrained_teacher_small.pth"
    )
    model.load_state_dict(torch.load(teacher_path, map_location=torch.device("cuda")))
    return model


def load_visa_dataset(path: str | Path = "./datasets/VisA"):
    datamodule = Visa(path)
    datamodule.prepare_data()
