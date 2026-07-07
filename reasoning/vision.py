"""
Vision model reasoning for imagery inputs.

Sends a base64-encoded image thumbnail to a local Ollama vision model
(LLaVA or Llama 3.2 Vision) and returns an archaeologically-framed
text description.

Called from main.py after imagery preprocessing. The returned description
is added to the imagery summary dict under 'vision_description' before
the summary enters Phase 1 reasoning.

Two-model architecture note:
  The vision model provides the image DESCRIPTION.
  The text model (Phase 1, Phase 2) provides the INTERPRETATION.
  These are intentionally separate concerns.
"""
import ollama
from config import get_num_predict

VISION_SYSTEM_PROMPT = """You are a visual pattern analyst. Your job is to describe
what you observe in an image as objectively and precisely as possible.
DO NOT attempt archaeological interpretation. DO NOT reference site types,
cultures, periods, or historical context. That interpretation is done separately.

Describe only what is directly visible:
- Geometric and linear features: lines, curves, edges, boundaries, shapes
- Tonal and colour patterns: light/dark contrasts, gradients, colour zones,
  anomalies that differ from surrounding areas
- Texture: uniformity, roughness, grain, repeating patterns, texture boundaries
- Spatial distribution: where features occur within the image (use directional
  references: north/south/east/west quadrants, centre, periphery)
- Anomalies: anything that appears discontinuous, unusual, or distinct from
  the surrounding visual field
- Image quality: if resolution, contrast, or compression limits what can be
  observed, state this explicitly

Be precise and spatial. Describe location, extent, orientation, and contrast
of every notable feature. Do not speculate about cause or meaning.
Target length: 200-300 words. Plain prose, no headings or bullet points."""

def run_vision_description(
    image_b64: str,
    user_prompt: str,
    model: str,
    filename: str = '',
    tier: str = 'mid',
) -> str:
    """
    Submit a base64 image to a local Ollama vision model.

    Args:
        image_b64:   Base64-encoded JPEG thumbnail string
        user_prompt: The researcher's analytical question
        model:       Ollama model name (must support vision, e.g. llava, llama3.2-vision)
        filename:    Original filename for context

    Returns:
        Text description string, or error message on failure.
    """
    context_block = (
        f'IMAGE FILE: {filename}\n'
        f'RESEARCHER QUESTION (for context only — do not interpret): {user_prompt}\n\n'
        'Describe the visual patterns, features, and anomalies you observe in this image. '
        'Be objective and spatial. Do not interpret — only describe.'
    )

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': VISION_SYSTEM_PROMPT},
                {
                    'role':    'user',
                    'content': context_block,
                    'images':  [image_b64],
                },
            ],
            options={
                'temperature': 0.1,
                'repeat_penalty': 1.15,
                'num_predict': get_num_predict(tier, 'vision'),
            },
        )
        return response['message']['content']

    except Exception as e:
        return f'Vision model error ({model}): {e}'
