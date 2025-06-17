import requests
import re
import json

url = "https://www.gutenberg.org/cache/epub/1526/pg1526.txt"
response = requests.get(url)
text = response.text

start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK THE MERCHANT OF VENICE ***"
end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK THE MERCHANT OF VENICE ***"
text = text.split(start_marker)[-1].split(end_marker)[0]

lines = text.splitlines()
dialogues = []
current = ""

def clean_line(line):
    # Szereplőnevek eltávolítása pl. ANTONIO.
    line = re.sub(r'^[A-Z][A-Z\s\-]+\.?\s*$', '', line)
    # Szereplőnévvel kezdődő sorokból eltávolítani a nevet pl. BASSANIO. Hello → Hello
    line = re.sub(r'^[A-Z][A-Z\s\-]+\.\s+', '', line)
    # Színpadi utasítások eltávolítása
    line = re.sub(r'\[.*?\]', '', line)
    return line.strip()

for line in lines:
    line = line.strip()

    if not line:
        if current:
            dialogues.append({"text": current.strip()})
            current = ""
        continue
    if re.match(r'^(ACT|SCENE)\b', line):
        continue
    if line.startswith("Enter") or line.startswith("Exit") or line.startswith("Exeunt"):
        continue
    if re.match(r'^[A-Z .\-]+$', line):  
        continue
    if re.match(r'^\[.*\]$', line):  
        continue

    cleaned = clean_line(line)
    if cleaned:
        current += " " + cleaned

if current:
    dialogues.append({"text": current.strip()})

with open("merchant_of_venice_clean.json", "w", encoding="utf-8") as f:
    json.dump(dialogues, f, ensure_ascii=False, indent=2)

print(f"{len(dialogues)} szövegrészlet mentve a merchant_of_venice_clean.json fájlba.")
