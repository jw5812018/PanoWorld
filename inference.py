import importlib
import os
import warnings

import torch
from torch.utils.data import DataLoader

from metric_utils import export_results
from setup import init_config
from utils import export_ply_forviewer, prepare_viewer


AMP_DTYPE_MAPPING = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "tf32": torch.float32,
}


def load_symbol(dotted_path):
    module_name, symbol_name = dotted_path.rsplit(".", 1)
    return importlib.import_module(module_name).__dict__[symbol_name]


def build_dataloader(config):
    dataset_cls = load_symbol(config.inference.get("dataset_name", "dataset.Dataset"))
    dataset = dataset_cls(config)

    num_workers = config.inference.num_workers
    dataloader_kwargs = {
        "batch_size": config.inference.batch_size_per_gpu,
        "shuffle": False,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "pin_memory": False,
    }
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = config.inference.prefetch_factor

    return DataLoader(dataset, **dataloader_kwargs)


def build_model(config, device):
    model_cls = load_symbol(config.model.class_name)
    model = model_cls(config).to(device)
    msg = model.load_ckpt(config.inference.ckpt_path)
    print(msg)
    model.eval()
    return model


def export_viewer_assets(result, out_dir, sh_degree):
    batch_size = result.input["input_images"].size(0)
    for batch_idx in range(batch_size):
        scene_name = result.input["input_target_scene_name"][batch_idx]
        inputs_view_name = result.input["input_view_names"][batch_idx]
        viewerdir = os.path.join(out_dir, f"{scene_name}/{inputs_view_name}/output_ply")
        point_cloud_dir = os.path.join(viewerdir, "point_cloud/iteration_0")
        os.makedirs(point_cloud_dir, exist_ok=True)

        export_ply_forviewer(
            result.gs_params,
            result.input["input_masks"][batch_idx],
            batch_idx,
            os.path.join(point_cloud_dir, "point_cloud.ply"),
        )
        prepare_viewer(result, viewerdir, sh_degree)


def run_inference(config):
    os.environ["OMP_NUM_THREADS"] = str(config.inference.get("num_threads", 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cuda.matmul.allow_tf32 = config.inference.use_tf32
    torch.backends.cudnn.allow_tf32 = config.inference.use_tf32

    dataloader = build_dataloader(config)
    model = build_model(config, device)

    print(f"Running inference; save results to: {config.inference.out_dir}")
    warnings.filterwarnings("ignore", category=FutureWarning)

    evaluation_folder_list = []
    autocast_enabled = config.inference.use_amp and device.type == "cuda"
    autocast_dtype = AMP_DTYPE_MAPPING[config.inference.amp_dtype]

    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        enabled=autocast_enabled,
        dtype=autocast_dtype,
    ):
        sample_target_images = config.data.get("sample_target_images", 6)
        for uid, batch in enumerate(dataloader, start=1):
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            print(uid - 1)

            input_data_dict = {key: value for key, value in batch.items() if "input" in key}
            target_data_dict = {key: value for key, value in batch.items() if "target" in key}
            result = model(input_data_dict, target_data_dict)

            export_results(
                result,
                config.inference.out_dir,
                sample_target_images=sample_target_images,
                uid=uid,
            )

            for batch_idx in range(input_data_dict["input_images"].size(0)):
                scene_name = result.input["input_target_scene_name"][batch_idx]
                inputs_view_name = result.input["input_view_names"][batch_idx]
                evaluation_folder_list.append(
                    os.path.join(config.inference.out_dir, f"{scene_name}/{inputs_view_name}")
                )

            export_viewer_assets(result, config.inference.out_dir, config.model.gaussians.sh_degree)

    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    run_inference(init_config())


if __name__ == "__main__":
    main()
