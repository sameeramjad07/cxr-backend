import torch
import torchvision.models as models
import time

def load_model(model_path="app/models/model_epoch_16.pth"):
    # Use InceptionV3 architecture
    model = models.inception_v3(weights=None, aux_logits=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 14)  # 14 classes for NIH dataset
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Remove aux_logits if not needed in state_dict
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("AuxLogits")}
    model.load_state_dict(state_dict, strict=False)  # Allow partial loading
    model.eval()
    return model

# Global model instance
model = load_model()

def predict(img_tensor):
    start_time = time.time()
    with torch.no_grad():
        outputs = model(img_tensor)
        if isinstance(outputs, tuple):  # Handle InceptionV3's aux_logits
            outputs = outputs[0]  # Use main logits
        predictions = torch.sigmoid(outputs).squeeze().cpu().numpy()
    inference_time = time.time() - start_time
    return predictions, inference_time