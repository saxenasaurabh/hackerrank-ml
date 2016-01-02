import json
count = 0
categoryToSection = {}
with open('training.json') as f:
    for line in f:
        if count != 0:
            parsedLine = json.loads(line)
            categoryToSection[parsedLine['category']] = parsedLine['section']
            if len(categoryToSection) == 16:
                break
        count += 1
for category in categoryToSection:
    print (category, categoryToSection[category])
