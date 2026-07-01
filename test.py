from preprocessors.imagery import preprocess_imagery
from pathlib import Path
r = preprocess_imagery(Path('data/Figure1.png'))
print('vision_description' in r)
print(r.get('vision_description', 'NOT PRESENT')[:300])
