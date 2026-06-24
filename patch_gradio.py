import re

path = r"C:\Users\DylanDavis\anaconda3\envs\archaeo_ai\lib\site-packages\gradio_client\utils.py"

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Add isinstance guard at the start of get_type
old = 'def get_type(schema):\n'
new = 'def get_type(schema):\n    if not isinstance(schema, dict): return "Any"\n'

if new not in content:
    content = content.replace(old, new, 1)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Patched successfully.")
else:
    print("Already patched.")