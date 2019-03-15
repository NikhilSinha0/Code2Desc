import csv
import re

def main():
    rows = []
    regex = re.compile('<code>(.+?)</code>', re.M)
    with open('raw_data.csv', 'r') as readfile:
        lines = csv.reader(readfile)
        while True:
            try: 
                row = next(lines)
                row = next(lines)
                if len(row) != 2:
                    continue
                question = parse_question(row[0], regex)
                answers = parse_answer(row[1], regex)
                for answer in answers:
                    rows.append([question, answer])
                break
            except UnicodeDecodeError:
                pass
            except StopIteration:
                break
    with open('data.csv', 'w+') as writefile:
        writer = csv.writer(writefile, delimiter=',')
        for row in rows:
            writer.writerow(row)

def parse_question(text, regex):
    code = regex.findall(text)
    for item in code:
        text = text.replace(item, '')
    parts = re.split('<|>', text)
    parsed_parts = []
    for x in range(len(parts)):
        if x%2 == 0:
            parsed_parts.append(parts[x])
    return ''.join(parsed_parts)

def parse_answer(text, regex):
    res = regex.findall(text)
    return res

if(__name__=='__main__'):
    main()

    