import torch

from .models import model_classes


def export_model(models_dir, model, device):
    checkpoint = torch.load(f"{models_dir}/{model}/best.pth")
    model_type = checkpoint["model_type"]
    tree = checkpoint["tree"]

    model_id = checkpoint["model_id"]
    model_class = model_classes[model_id]
    model = model_class(tree)
    model.load_state_dict(checkpoint["model_state_dict"])

    batch_size = 8

    if model_type == "image":  # torch.Size([8, 3, 224, 224])
        input_shape = (3, 224, 224)
        dummy_input = torch.randn(batch_size, *input_shape).to(device)
    elif model_type == "text":  # torch.Size([8, 256])
        input_shape = 256
        dummy_input = torch.randn(batch_size, input_shape).to(device)
    else:
        raise ValueError(f"Unknown model type {model_type}")

    model.to(device)
    model.eval()
    dummy_input = dummy_input.long()

    # print('dummy_input', dummy_input.shape)
    # Testing against text model gives following error. May be
    # worth trying model.half or float.
    # RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when
    # calling `cublasCreate(handle)`
    torch.onnx.export(
        model,
        dummy_input,
        f"{model}.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
    )
