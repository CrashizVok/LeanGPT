import requests
import re
import json

url = "https://www.gutenberg.org/cache/epub/1526/pg1526.txt"
response = requests.get(url)
text = response.text

start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK TWELFTH NIGHT; OR, WHAT YOU WILL ***"
end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK TWELFTH NIGHT; OR, WHAT YOU WILL ***"
text = text.split(start_marker)[-1].split(end_marker)[0]

lines = text.splitlines()
dialogues = []
current = ""

for line in lines:
    line = line.strip()

    if not line:
        if current:
            dialogues.append({"text": current.strip()})
            current = ""
        continue

    if re.match(r'^(ACT|SCENE)\b', line):
        continue
    if line.startswith(("Enter", "Exit", "Exeunt")):
        continue
    if re.search(r'\[.*?\]', line):
        continue
    if re.match(r'^[A-Z .\-]+\.$', line):
        continue

    # Név eltávolítása a sor elejéről (pl. "VIOLA.")
    line = re.sub(r'^[A-Z .\-]+\.\s*', '', line)

    current += " " + line

if current:
    dialogues.append({"text": current.strip()})

with open("twelfth_night.json", "w", encoding="utf-8") as f:
    json.dump(dialogues, f, ensure_ascii=False, indent=2)

print(f"{len(dialogues)} szövegrészlet mentve a twelfth_night.json fájlba.")
