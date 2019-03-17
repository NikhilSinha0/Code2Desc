import csv
# import re
from bs4 import BeautifulSoup

def main():
    # rows = []
    # regex = re.compile('<code>(.+?)</code>', re.M)
    with open('raw_data.csv', 'r') as readfile:
        lines = csv.reader(readfile)
        next(lines)
        with open('data.csv', 'w+') as writefile:
            writer = csv.writer(writefile)
            while True:
                try:
                    row = next(lines)
                    if len(row) != 2:
                        continue
                    question = parse_question(row[0])
                    answers = parse_answer(row[1])
                    for answer in answers:
                        writer.writerow([question, answer])

                    # question = parse_question(row[0], regex)
                    # print('Question:' + question)
                    # answers = parse_answer(row[1], regex)

                    # break
                except UnicodeDecodeError:
                    pass
                except StopIteration:
                    break

# def parse_question(text, regex):
#     code = regex.findall(text)
#     for item in code:
#         text = text.replace(item, '')
#     parts = re.split('<|>', text)
#     parsed_parts = []
#     for x in range(len(parts)):
#         if x%2 == 0:
#             parsed_parts.append(parts[x])
#     return ''.join(parsed_parts)

def parse_question(text):
    soup = BeautifulSoup(text, 'html.parser')
    for block in soup.find_all('code'):
        block.decompose()
    return soup.get_text()

# def parse_answer(text, regex):
#     res = regex.findall(text)
#     return res

def parse_answer(text):
    soup = BeautifulSoup(text, 'html.parser')
    blocks = []
    for block in soup.find_all('code'):
        blocks.append(block.get_text())
    return blocks

if __name__=='__main__':
    main()

