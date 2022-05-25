import re
import json
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup as bs
from urllib.parse import urljoin

def crawling(file_path, url):
    # request로 문서 전체 가져오고 beautifulsoup으로 파싱
    soup = bs(requests.get(url).text, 'html.parser')

    data = {}
    data['statistics'] = [] # 범용화 필요 -> url에서 book 다음에 나오는 이름으로 지정 [변경 팔요]

    base = url.split('/')
    base = '/'.join(base[:-1])+'/'
    for link in tqdm(soup.find_all('a')): # 모든 하이퍼 링크 찾기
        number, title = link.find('span', class_='os-number'), link.find('span', class_='os-text')
        if title and number:
             # chapter 번호가 없는 링크 존재 (Key Terms 만 중요해보임)
            if number:
                number = number.get_text()
                title = title.get_text()
            # url 당 content 가져오기
            url = urljoin(base, link.get('href'))  # base 링크와 상대경로 링크를 결합
            soup = bs(requests.get(url).content, 'html.parser', from_encoding='utf-8')
            content = soup.find('div', id="main-content") # main content 가져오기

            # ('div.main-content > div.chapter-content-module > div.unnumbered (여기에 mathml 코드 있음... 왜그렇지?) > div.MathJax_Display > span.MathJax')
            mathjax = content.select('div.chapter-content-module > div.unnumbered')
            if mathjax:
                mathml_all = []
                for mathml in mathjax:
                    mathml = str(mathml).split('>')[1:-2]
                    mathml = ">".join(mathml) + '>'
                    mathml_all.append(mathml)
                data['statistics'].append({"chapter" : number,
                                           "title" : title,
                                           "mathml" : mathml_all,
                                           "content" : content.get_text()})
    with open(file_path, 'w') as file:
        json.dump(data, file)

if __name__=="__main__":
    file_path = './data/crawling_data.json'
    url = 'https://openstax.org/books/introductory-business-statistics/pages/1-introduction'
    crawling(file_path, url)