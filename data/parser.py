import csv
import nltk
import string
import re
from bs4 import BeautifulSoup

stop_words = set(nltk.corpus.stopwords.words('english'))

def main():
    # rows = []
    # regex = re.compile('<code>(.+?)</code>', re.M)
    with open('raw_data_titles.csv', 'r') as readfile:
        lines = csv.reader(readfile)
        next(lines)
        with open('data.csv', 'w+') as writefile:
            writer = csv.writer(writefile)
            writer.writerow(['Question', 'Answer'])
            while True:
                try:
                    row = next(lines)
                    if len(row) != 2:
                        continue
                    question = parse_question(row[0])
                    answer = parse_answer(row[1])
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
    words = nltk.word_tokenize(text)
    filtered_words = [word.lower() for word in words if word not in stop_words and word.lower!="how"]
    return ' '.join(filtered_words)

# def parse_answer(text, regex):
#     res = regex.findall(text)
#     return res

def parse_answer(text):
    soup = BeautifulSoup(text, 'html.parser')
    blocks = []
    for block in soup.find_all('code'):
        lines = block.get_text().split('\n')
        for line in lines:
            words = nltk.word_tokenize(' '.join(re.split('['+''.join(string.punctuation)+']', line)))
            if len(words)==0 or words[0]=='#': #remove comments
                continue
            words=[word.lower() for word in words if word not in string.punctuation and not word.isnumeric() and not len(word)==1]
            blocks.append(' '.join(words))
    return ' '.join(blocks)

if __name__=='__main__':
    main()

