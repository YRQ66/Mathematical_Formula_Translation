import argparse
import requests
from bs4 import BeautifulSoup

import os
import time
from os.path import join
from tqdm import tqdm

url_li = set()

def get_url(url, count):
    # origin_url에 해당하는 하위 url 크롤링
    if count == 0:
        return 
    # 크롤링 불가능한 사이트는 skip
    try:
        req = requests.get(url) 
        soup = BeautifulSoup(req.content, "html.parser")
    except:
        return

    for line in soup.select("body"):
        can_url = line.find_all('a')
        for can_u in can_url:
            new_url = can_u.get('href')
            if new_url!=None and new_url.startswith('/wiki/'):
                new_url = 'https://ko.wikipedia.org'+new_url
                url_li.add(new_url)
                get_url(new_url, count - 1)

def save_url(origin_url, count, save_dir):

    pth = join(save_dir, 'url.lst')

    if not os.path.isfile(pth):
        open(pth, 'w')

    with open(pth, 'r') as f:
        pre_url_li = [url.strip('\n') for url in f.readlines()]
        pre_url_li = set(pre_url_li)

    print("Start crawling URL...")
    url_li.add(origin_url)
    get_url(origin_url, count = count)
    print('Crawling URL Done! Toatl {}'.format(len(url_li)))

    new_url = url_li - pre_url_li

    with open(pth, "a") as f:
        f.write('\n'.join(new_url))

    return new_url

def get_latex(url, latex_li=[]):
    # url안에 alt 태그 달린 내용 중 latex 코드만 필터링
    try:
        page = requests.get(url)
    except:
        print('Error URL {}'.format(url))
        return

    soup = BeautifulSoup(page.content, "html.parser", from_encoding="iso-8859-1")

    for line in soup.select("body"):
        text = line.find_all('img')
        for t in text:
            latex = t.get('alt')
            if latex != None and latex.startswith('{\displaystyle'):
                latex = latex.replace("\\displaystyle ","")
                latex = latex.replace("\\overline ","")
                latex_li.append(latex)
    return latex_li

def save_latex(origin_url, count, save_dir = './'):
    # url_li 안의 모든 url의 latex 코드 크롤링 (crawl_save_latex > get_latex)

    # Crawling URL
    new_url = save_url(origin_url, count, save_dir) # depth 

    # Crwaling LaTeX
    print('Crawling LaTeX data....')
    latex = set() # 중복 제거
    for url in tqdm(new_url):
        code = get_latex(url)
        if code:
            latex.update(code) # latex crawling
    print('Crawling LaTeX Done! Total {}'.format(len(latex)))
        # save
    str_latex = '\n'.join(latex)
    pth = join(save_dir, 'formulas.txt')
    with open(pth, "a") as f:
        f.write(str_latex)
    print('Save latex Data!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--url")
    args = parser.parse_args()

    origin_url = args.url
    save_latex(origin_url, 2)