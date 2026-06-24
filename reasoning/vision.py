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

VISION_SYSTEM_PROMPT = """You are an archaeological imagery analyst.
You will be shown an image alongside a researcher's question.

Describe what you observe with archaeological relevance:
- Landscape features, terrain patterns, vegetation anomalies
- Visible structures, earthworks, mounds, or surface disturbances
- Color patterns, tonal contrasts, or texture variations
- Anything suggesting human activity, site formation, or land modification

Spatial precision: describe locations within the image (e.g., 'northeast quadrant', 'center').
Epistemic clarity: distinguish observation from interpretation.
Resolution honesty: if image quality limits analysis, state this explicitly.

Target length: 200-300 words. Do not add headings or bullet points."""


def run_vision_description(
    image_b64: str,
    user_prompt: str,
    model: str,
    filename: str = '',
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
    user_message = (
        f'RESEARCHER QUESTION: {user_prompt}\n\n'
        f'IMAGE FILE: {filename}\n\n'
        'Describe what you observe in this image with archaeological relevance.'
    )

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': VISION_SYSTEM_PROMPT},
                {
                    'role':    'user',
                    'content': user_message,
                    'images':  [image_b64],
                },
            ],
            options={'temperature': 0.1},
        )
        return response['message']['content']

    except Exception as e:
        return f'Vision model error ({model}): {e}'