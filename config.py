"""
Hardware detection and model tier configuration.
Phase 2 update: adds vision_model per tier.
Phase 3 update: adds NUM_PREDICT token caps per tier and phase.
               Fixes vision model tags for cpu/low tiers.
"""
import subprocess
from typing import Optional
import requests

OLLAMA_BASE_URL = "http://localhost:11434"

TIERS = {
    'cpu': {
        'tier_label': 'CPU Only',
        'model': 'phi3:mini',
        'vision_model': 'llava:7b',
        'context_limit': 4096,
    },
    'low': {
        'tier_label': 'Low GPU (6-8 GB VRAM)',
        'model': 'llama3.1:8b-instruct-q4_K_M',
        'vision_model': 'llava:7b',
        'context_limit': 8192,
    },
    'mid': {
        'tier_label': 'Mid GPU (16-24 GB VRAM)',
        'model': 'llama3.3:70b-instruct-q4_K_M',
        'vision_model': 'llama3.2-vision:11b',
        'context_limit': 16384,
    },
    'high': {
        'tier_label': 'High GPU (48+ GB VRAM)',
        'model': 'llama3.3:70b-instruct-fp16',
        'vision_model': 'llama3.2-vision:11b',
        'context_limit': 32768,
    },
}

NUM_PREDICT = {
    'cpu': {
        'phase1':   768,
        'phase2':  1536,
        'critique': 768,
        'revise':  1536,
        'delta':   1024,
        'followup': 1024,
        'vision':   384,
    },
    'low': {
        'phase1':  1024,
        'phase2':  2048,
        'critique': 1024,
        'revise':  2048,
        'delta':   1536,
        'followup': 1536,
        'vision':   512,
    },
    'mid': {
        'phase1':  1536,
        'phase2':  3072,
        'critique': 1536,
        'revise':  3072,
        'delta':   2048,
        'followup': 2048,
        'vision':   768,
    },
    'high': {
        'phase1':  2048,
        'phase2':  4096,
        'critique': 2048,
        'revise':  4096,
        'delta':   3072,
        'followup': 3072,
        'vision':  1024,
    },
}


def get_num_predict(tier: str, phase: str) -> int:
    """
    Return the num_predict cap for a given hardware tier and pipeline phase.

    Args:
        tier:  'cpu' | 'low' | 'mid' | 'high'
        phase: 'phase1' | 'phase2' | 'critique' | 'revise' |
               'delta' | 'followup' | 'vision'

    Returns:
        Token cap (int). Defaults to mid-tier phase2 value if not found.
    """
    return NUM_PREDICT.get(tier, NUM_PREDICT['mid']).get(phase, 3072)


def check_ollama() -> bool:
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def get_available_vram_gb() -> Optional[float]:
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            free_mb = int(result.stdout.strip().split('\n')[0])
            return free_mb / 1024
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


def detect_tier() -> str:
    vram = get_available_vram_gb()
    if vram is None: return 'cpu'
    if vram >= 40:   return 'high'
    elif vram >= 14: return 'mid'
    elif vram >= 5:  return 'low'
    else:            return 'cpu'


def get_model_config(
    tier: str = 'auto',
    model_override: Optional[str] = None,
    vision_model_override: Optional[str] = None,
) -> dict:
    """
    Return model configuration for the specified tier.
    Phase 3: tier is stored in returned config for use by reasoning scripts.
    """
    if tier == 'auto':
        tier = detect_tier()
    config = TIERS.get(tier, TIERS['cpu']).copy()
    config['tier'] = tier
    if model_override:
        config['model'] = model_override
        config['tier_label'] += ' (override)'
    if vision_model_override:
        config['vision_model'] = vision_model_override
    return config
