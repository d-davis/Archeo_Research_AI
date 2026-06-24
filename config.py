"""
Hardware detection and model tier configuration.
Phase 2 update: adds vision_model per tier.
"""
import subprocess
from typing import Optional
import requests

OLLAMA_BASE_URL = "http://localhost:11434"

TIERS = {
    'cpu': {
        'tier_label':    'CPU Only',
        'model':         'phi3:mini',
        'vision_model':  'llava:7b-v1.6-q4_K_M',
        'context_limit': 4096,
    },
    'low': {
        'tier_label':    'Low GPU (6-8 GB VRAM)',
        'model':         'llama3.1:8b-instruct-q4_K_M',
        'vision_model':  'llava:7b-v1.6-q4_K_M',
        'context_limit': 8192,
    },
    'mid': {
        'tier_label':    'Mid GPU (16-24 GB VRAM)',
        'model':         'llama3.3:70b-instruct-q4_K_M',
        'vision_model':  'llama3.2-vision:11b-instruct-q4_K_M',
        'context_limit': 16384,
    },
    'high': {
        'tier_label':    'High GPU (48+ GB VRAM)',
        'model':         'llama3.3:70b-instruct-fp16',
        'vision_model':  'llama3.2-vision:11b-instruct-q4_K_M',
        'context_limit': 32768,
    },
}


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
    if vram is None:  return 'cpu'
    if   vram >= 40:  return 'high'
    elif vram >= 14:  return 'mid'
    elif vram >=  5:  return 'low'
    else:             return 'cpu'


def get_model_config(
    tier: str = 'auto',
    model_override: Optional[str] = None,
    vision_model_override: Optional[str] = None,
) -> dict:
    """
    Return model configuration for the specified tier.
    Phase 2: accepts optional vision_model_override.
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
