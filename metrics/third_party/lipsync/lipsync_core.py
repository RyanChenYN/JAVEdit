import os
from .syncnet_model import SyncNet
import torch
from omegaconf import OmegaConf
import math
from .audio import melspectrogram
from decord import AudioReader

class LipSync():
    def __init__(self, 
                 device_id,
                 config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs.yaml")
        config_path = os.path.abspath(config_path)
        self.device = torch.device(f'cuda:{device_id}')
        config = OmegaConf.load(config_path)
        self.syncnet = SyncNet(OmegaConf.to_container(config.model)).to(self.device)
        ckpt_path = config.ckpt.inference_ckpt_path
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.abspath(os.path.join(os.path.dirname(config_path), ckpt_path))
        print(f"Load checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        self.syncnet.load_state_dict(checkpoint["state_dict"])
        self.syncnet.to(dtype=torch.float16)
        self.syncnet.requires_grad_(False)
        self.syncnet.eval()

        self.resolution = config.data.resolution
        self.num_frames = config.data.num_frames

        self.mel_window_length = math.ceil(self.num_frames / 5 * 16)

        self.audio_sample_rate = config.data.audio_sample_rate
        self.video_fps = config.data.video_fps
        self.audio_samples_length = int(
            config.data.audio_sample_rate // config.data.video_fps * config.data.num_frames
        )

    def crop_audio_window(self, original_mel, start_index):
        start_idx = int(80.0 * (start_index / float(self.video_fps)))
        end_idx = start_idx + self.mel_window_length
        return original_mel[:, start_idx:end_idx].unsqueeze(0)
    
    def get_mel(self, wav_data):
        return melspectrogram(wav_data)
    
    def read_audio(self, video_path: str):
        ar = AudioReader(video_path, sample_rate=self.audio_sample_rate)
        original_mel = melspectrogram(ar[:].asnumpy().squeeze(0))
        return torch.from_numpy(original_mel)

if __name__ == '__main__':
    from video_analysis import VideoAnalysis
    lipsync = LipSync(0)
    video_analysis = VideoAnalysis(0)
    segs, wav_data = video_analysis.audio_segment('data/3_20s.mp4')
    print(wav_data.shape)
    mel_data = melspectrogram(wav_data)
    print(mel_data.shape)
