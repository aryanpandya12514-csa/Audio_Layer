import torch
import torch.nn as nn
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

class WavLMFakeDetector(nn.Module):
    def __init__(self, model_name="microsoft/wavlm-base"):
        super(WavLMFakeDetector, self).__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.wavlm = WavLMModel.from_pretrained(model_name)
        
        # Freeze foundation parameters
        for param in self.wavlm.parameters():
            param.requires_grad = False
            
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, audio_arrays):
        # FIX: Detect device from the data, not the unmaterialized model
        device = audio_arrays.device 
        
        # Safely move data to CPU for the pre-processor
        audio_list = [audio_arrays[i].detach().cpu().numpy() for i in range(audio_arrays.shape[0])]
            
        inputs = self.feature_extractor(
            audio_list, 
            sampling_rate=16000, 
            padding=True, 
            return_tensors="pt"
        )
        
        input_values = inputs.input_values.to(device)
        outputs = self.wavlm(input_values=input_values)
        
        # Geometric Pooling (Mean across time)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        
        return self.classifier(pooled_output)