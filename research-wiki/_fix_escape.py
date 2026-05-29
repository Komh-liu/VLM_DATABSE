path = '/home/liu/Desktop/VLM/research-wiki/generate-graph-data.py'
with open(path, 'rb') as f:
    content = f.read()

# The current bytes in the file should be:
old = b"  tex: {inlineMath: [['$', '$'], ['\\\\\\(', '\\\\\\)']]},"
# Wait, in bytes literal, \\\\ is 4 backslashes and \\( is \\(
# Actually let me use a simpler approach - just search for inlineMath line

import re
text = content.decode('utf-8')

# Find the inlineMath line
for line in text.split('\n'):
    if 'inlineMath' in line:
        print(repr(line))
