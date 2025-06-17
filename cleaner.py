import requests
import re
import json

url = "https://www.gutenberg.org/cache/epub/1524/pg1524.txt"
response = requests.get(url)
text = response.text

start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK HAMLET, PRINCE OF DENMARK ***"
end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK HAMLET, PRINCE OF DENMARK ***"
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
    if line.startswith("Enter") or line.startswith("Exit") or line.startswith("Exeunt"):
        continue
    if re.match(r'^[A-Z .\-]+$', line):  
        continue
    if re.match(r'^\[.*\]$', line):  
        continue

    current += " " + line

if current:
    dialogues.append({"text": current.strip()})

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(dialogues, f, ensure_ascii=False, indent=2)

print(f"{len(dialogues)} szövegrészlet mentve a data.json fájlba.")
