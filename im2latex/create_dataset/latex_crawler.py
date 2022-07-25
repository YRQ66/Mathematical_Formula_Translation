import requests
from bs4 import BeautifulSoup

from os.path import join
from tqdm import tqdm

def get_url(url):
    # origin url에 포함된 모든 링크 크롤링
    req = requests.get(url)
    soup = BeautifulSoup(req.content, "html.parser")
    url_li = []
    for line in soup.select("body"):
        content_name = line.find('h1').text # 각 데이터베이스마다 카테고리를 명시하기 위해 (아직 코드에 반영 안됨, ex. 통계학)
        can_url = line.find_all('a')
        for can_u in can_url:
            url = can_u.get('href')
            if url!=None and url.startswith('https://'):
                url_li.append(url)
    return url_li
    
def get_latex(url, latex_li = []):
    # url안에 alt 태그 달린 내용 중 latex 코드만 필터링
    req = requests.get(url)
    soup = BeautifulSoup(req.content, "html.parser")

    for line in soup.select("body"):
        text = soup.find_all('img')
        for t in text:
            latex = t.get('alt')
            if latex != None and latex.startswith('{\displaystyle'):
                latex = latex.replace("\\displaystyle ","")
                latex = latex.replace("\\overline ","")
                latex_li.append(latex)
    return latex_li

def save_latex(origin_url, save_dir = './formula'):
    # crawl latex
    print('Crwaling LaTeX data....')
    latex = set() # 중복 제거
    url_li = [origin_url]
    url_li.extend(get_url(origin_url)) # url crawling
    for url in tqdm(url_li):
        latex.update(get_latex(url)) # latex crawling
    print('Crawling Done!')
    str_latex = '\n'.join(latex)
    # save
    data_name = "{}.lst".format(origin_url.split('/')[-1])
    full_dir = join(save_dir, data_name)
    with open(full_dir, "w") as f:
        f.write(str_latex)
    

if __name__=='__main__':
    url = "https://ko.wikipedia.org/wiki/%ED%86%B5%EA%B3%84%ED%95%99"
    save_latex(url)