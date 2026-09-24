"""Speaker-lip matching (audio-visual sync) model: which on-screen face produced a separated audio stream
(docs/SYNC_SPEC.md)."""
from .data import SyncWindowDataset, audio_window_batch, audio_window_features, sync_collate
from .losses import frame_scores, pair_scores, sync_loss
from .model import SyncModel, build_sync_model

__all__ = [
    "SyncWindowDataset", "sync_collate", "audio_window_features", "audio_window_batch",
    "SyncModel", "build_sync_model", "pair_scores", "frame_scores", "sync_loss",
]
