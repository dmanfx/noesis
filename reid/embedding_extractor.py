import logging
from typing import List, Optional

import numpy as np


class EmbeddingExtractor:
    """Thin wrapper around torchreid's FeatureExtractor for OSNet embeddings.

    Accepts BGR numpy crops and returns L2-normalized embeddings as np.ndarray (N x D).
    Falls back to simple color histogram features if torchreid is unavailable.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda:0", image_size=(256, 128), model_name: str = "osnet_x1_0"):
        self.logger = logging.getLogger("EmbeddingExtractor")
        self.device = device
        self.image_size = image_size
        self.model_name = model_name
        # Disable classical feature fallback so failures are visible
        self.allow_fallback = False
        self._use_torchreid = False
        self._extractor = None

        # Try path A: torchreid FeatureExtractor (KaiyangZhou API)
        # Normalize device if CUDA is unavailable
        try:
            import torch
            if ('cuda' in str(device).lower()) and (not torch.cuda.is_available()):
                self.logger.warning(f"CUDA not available; falling back to CPU for ReID (requested {device})")
                device = 'cpu'
                self.device = device
        except Exception:
            pass

        try:
            # Try to detect torchreid presence and version for clearer logs
            try:
                import torchreid  # type: ignore
                tr_ver = getattr(torchreid, '__version__', 'unknown')
                self.logger.info(f"torchreid detected (version: {tr_ver})")
            except Exception:
                torchreid = None  # type: ignore

            # Prefer non-raising detection of FeatureExtractor to avoid noisy ModuleNotFound warnings
            import importlib.util as _ils
            fe_specs = [
                'torchreid.utils',
                'torchreid.utils.feature_extractor',
            ]
            fe_available = any(_ils.find_spec(name) is not None for name in fe_specs)

            if fe_available:
                # Import FeatureExtractor from whichever layout is available
                FeatureExtractor = None  # type: ignore
                try:
                    from torchreid.utils import FeatureExtractor  # type: ignore
                except Exception:
                    try:
                        from torchreid.utils.feature_extractor import FeatureExtractor  # type: ignore
                    except Exception:
                        FeatureExtractor = None  # type: ignore

                if FeatureExtractor is not None:
                    kwargs = {
                        "model_name": model_name,
                        "device": device,
                        "image_size": image_size,
                        "pixel_norm": True,
                    }
                    if model_path:
                        kwargs["model_path"] = model_path
                    self._extractor = FeatureExtractor(**kwargs)
                    self._use_torchreid = True
                    self.logger.info("Initialized OSNet FeatureExtractor (torchreid.utils)")
                else:
                    # Spec reported available but import failed; fall back gracefully
                    raise ImportError("FeatureExtractor symbol not found in torchreid.utils")
            else:
                # FeatureExtractor module not present in this torchreid; fall through
                raise ImportError("FeatureExtractor module not present in this torchreid build")
        except Exception as e:
            # FeatureExtractor path unavailable; try build_model path quietly
            self.logger.info(f"Using build_model path for ReID (FeatureExtractor unavailable: {e})")
            try:
                import torch
                from torchreid import models  # type: ignore
                self._tr_model = models.build_model(model_name, num_classes=1000, loss='softmax', pretrained=not bool(model_path))
                if model_path:
                    ckpt = torch.load(model_path, map_location='cpu')
                    # Handle common torchreid checkpoint formats
                    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                        state = ckpt['state_dict']
                    else:
                        state = ckpt
                    # Strip any DistributedDataParallel prefixes
                    state = {k.replace('module.', ''): v for k, v in state.items()}
                    # Drop classifier weights/bias to avoid num_classes mismatches (e.g., MSMT17 has 4101)
                    state = {k: v for k, v in state.items() if not k.startswith('classifier.')}
                    missing_unexp = self._tr_model.load_state_dict(state, strict=False)
                    self.logger.info(
                        f"Loaded ReID weights with classifier removed; missing={len(getattr(missing_unexp, 'missing_keys', []))}, "
                        f"unexpected={len(getattr(missing_unexp, 'unexpected_keys', []))}"
                    )
                # Move to device with fallback to CPU if needed
                try:
                    self._tr_model.eval().to(device)
                    self._tr_device = device
                except Exception as move_err:
                    self.logger.warning(f"Failed to move ReID model to {device}: {move_err}; using CPU")
                    self._tr_model.eval().to('cpu')
                    self._tr_device = 'cpu'
                self._use_torchreid = True
                self._extractor = None  # Using direct model path
                self.logger.info("Initialized OSNet via torchreid.models.build_model (headless classifier)")
            except Exception as e2:
                self._use_torchreid = False
                self.logger.error(f"torchreid unavailable, embeddings disabled: {e2}")

    def _l2_normalize(self, feats: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        return feats / norms

    def _extract_color_hist(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        feats = []
        for img in crops_bgr:
            if img is None or img.size == 0:
                feats.append(np.zeros(96, dtype=np.float32))
                continue
            try:
                import cv2
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
                s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
                v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
                vec = np.concatenate([h.flatten(), s.flatten(), v.flatten()]).astype(np.float32)
                vec = vec / (np.sum(vec) + 1e-6)
                feats.append(vec)
            except Exception:
                feats.append(np.zeros(96, dtype=np.float32))
        return self._l2_normalize(np.stack(feats, axis=0))

    def extract(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        """Return L2-normalized embeddings for a list of BGR crops.

        Args:
            crops_bgr: list of HxWxC uint8 images in BGR order
        Returns:
            np.ndarray of shape (N, D)
        """
        if not crops_bgr:
            return np.zeros((0, 512), dtype=np.float32)

        if self._use_torchreid and self._extractor is not None:
            try:
                # Convert BGR → RGB for torchreid
                import cv2
                rgb_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in crops_bgr]
                feats = self._extractor(rgb_imgs)  # returns torch.Tensor on CPU
                feats = feats.detach().cpu().numpy().astype(np.float32)
                return self._l2_normalize(feats)
            except Exception as e:
                self.logger.error(f"torchreid extraction failed (FeatureExtractor path): {e}")

        # Path B: direct model forward
        if hasattr(self, '_tr_model'):
            try:
                import torch
                import cv2
                H, W = int(self.image_size[0]), int(self.image_size[1])
                batch = []
                for img in crops_bgr:
                    if img is None or img.size == 0:
                        continue
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
                    ten = torch.from_numpy(resized).permute(2,0,1).float() / 255.0
                    # ImageNet normalization
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                    ten = (ten - mean) / std
                    batch.append(ten)
                if not batch:
                    return np.zeros((0, 512), dtype=np.float32)
                device = getattr(self, '_tr_device', self.device if self.device else 'cpu')
                inp = torch.stack(batch, dim=0).to(device)
                with torch.no_grad():
                    feats = self._tr_model(inp)  # eval() returns embeddings
                feats = feats.detach().cpu().numpy().astype(np.float32)
                return self._l2_normalize(feats)
            except Exception as e:
                self.logger.error(f"torchreid model forward failed: {e}")

        # Fallback disabled: return empty embeddings to surface failures clearly
        if self.allow_fallback:
            return self._extract_color_hist(crops_bgr)
        self.logger.warning("EmbeddingExtractor fallback disabled; returning empty embeddings")
        return np.zeros((0, 512), dtype=np.float32)
