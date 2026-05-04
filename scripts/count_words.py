import sys, re, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from docx import Document

paths = [
    r'C:\Users\User\OneDrive\Desktop\FYP REPORT AI\FYP AI REPORT AI.docx',
    r'C:\Users\User\OneDrive\Desktop\FYP-Fraud-Detection\dissertation\FYP REPORT (AI CLAUDE CODE).docx',
]
doc = None
used_path = ''
for p in paths:
    try:
        doc = Document(p)
        used_path = p
        break
    except Exception:
        continue
if doc is None:
    print('Could not open any document path')
    sys.exit(1)
print(f'Opened: {used_path}\n')

# Map heading paragraphs to chapter numbers
# Based on the dump: Heading 1 = chapter level, Heading 2/3 = sub-sections
# The actual content chapters start after the TOC and front matter

chapter_map = {
    'Literature Review': 'Ch 2: Literature Review',
    'Product Review (if applicable)': 'Ch 3-TEMPLATE',
    'Requirements Analysis and Design': 'Ch 3: Requirements & Design',
    'Implementation': 'Ch 4: Implementation',
    'Results and Evaluation': 'Ch 5: Results & Evaluation',
    'Legal, Social, Ethical and Professional Issues': 'Ch 6: Legal/Ethical',
    'Conclusion': 'Ch 7: Conclusion',
}

current_section = 'FRONT_MATTER'
sections = {}
sections[current_section] = []

# Track which sections to count
countable = set()

for i, para in enumerate(doc.paragraphs):
    text = para.text.strip()
    if not text:
        continue
    style = para.style.name if para.style else ''

    # Skip TOC entries
    if style.startswith('toc'):
        continue

    # Detect pre-contents headings
    if style == 'Headings (pre contents)':
        if 'Abstract' in text:
            current_section = 'Abstract'
        elif 'Acknowledgement' in text:
            current_section = 'Acknowledgements'
        elif 'Contents' in text:
            current_section = 'TOC'
        elif 'Declaration' in text:
            current_section = 'Declaration'
        sections.setdefault(current_section, [])
        continue

    # Detect appendix headings
    if style == 'Heading1 for appendices':
        current_section = 'APPENDIX'
        sections.setdefault(current_section, [])
        continue

    # Detect chapter headings (Heading 1)
    if style == 'Heading 1':
        mapped = chapter_map.get(text, None)
        if mapped:
            current_section = mapped
            if 'TEMPLATE' not in mapped:
                countable.add(mapped)
        else:
            current_section = f'OTHER: {text}'
        sections.setdefault(current_section, [])
        continue

    # Ch 1 special case: detected by Heading 3 "Questions/points..." before it
    # Actually Ch 1 starts around paragraph 126 with "Abstract" as Normal
    # Let me check: paragraph 126 is "Abstract" in Normal style, then content follows
    # The intro chapter doesn't have a Heading 1. Let me find it by looking for
    # "Background" or "Aims and Objectives" in Heading 2
    if style == 'Heading 2' and text == 'Aims and Objectives' and current_section == 'FRONT_MATTER':
        current_section = 'Ch 1: Introduction'
        countable.add(current_section)
        sections.setdefault(current_section, [])
        continue
    if style == 'Heading 3' and 'Questions/points to consider' in text:
        current_section = 'Ch 1: Introduction'
        countable.add(current_section)
        sections.setdefault(current_section, [])
        continue

    # Skip sub-headings (Heading 2, Heading 3) from word count
    if style in ('Heading 2', 'Heading 3'):
        continue

    # Skip "References" as Normal style
    if text in ('References', 'REFERENCES') and style == 'Normal':
        current_section = 'References'
        sections.setdefault(current_section, [])
        continue

    # Skip code snippets
    if re.match(r'^\s*(def |import |print\(|df\[|df\.|#\s)', text):
        continue
    if any(kw in text for kw in ['lambda x:', '.rolling(', '.transform(', '.groupby(', '.sort_values(', '.set_index(', '.reset_index(']):
        continue
    # Lines that are mostly code chars
    if len(text) > 15:
        code_chars = sum(1 for c in text if c in '(){}[]=_.:;')
        if code_chars / len(text) > 0.2:
            continue

    # Skip figure/table captions
    if re.match(r'^(Table|Figure)\s+\d+', text):
        continue

    sections.setdefault(current_section, [])
    sections[current_section].append(text)

# Print results
print('=' * 65)
print(f'{"Section":<35} {"Words":>8}')
print('=' * 65)

total_chapters = 0
for section in ['FRONT_MATTER', 'Abstract', 'Declaration', 'Acknowledgements', 'TOC',
                'Ch 1: Introduction', 'Ch 2: Literature Review',
                'Ch 3-TEMPLATE', 'Ch 3: Requirements & Design',
                'Ch 4: Implementation', 'Ch 5: Results & Evaluation',
                'Ch 6: Legal/Ethical', 'Ch 7: Conclusion',
                'References', 'APPENDIX']:
    if section not in sections:
        continue
    words = sections[section]
    all_text = ' '.join(words)
    wc = len(all_text.split())
    is_countable = section in countable
    marker = '  <<' if is_countable else ''
    print(f'{section:<35} {wc:>8}{marker}')
    if is_countable:
        total_chapters += wc

# Check for any unmapped sections
for section in sections:
    if section.startswith('OTHER'):
        words = sections[section]
        wc = len(' '.join(words).split())
        print(f'{section:<35} {wc:>8}')

print('=' * 65)
print(f'{"TOTAL (Chapters 1-7)":<35} {total_chapters:>8}')
print(f'{"Max allowed":<35} {13200:>8}')
print(f'{"Remaining":<35} {13200 - total_chapters:>8}')
