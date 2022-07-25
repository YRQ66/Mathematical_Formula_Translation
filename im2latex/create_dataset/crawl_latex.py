import requests
from selenium import webdriver
from time import sleep
from bs4 import BeautifulSoup

def get_url(url):
    req = requests.get(url)
    soup = BeautifulSoup(req.content, "html.parser")
    url_li = []
    for line in soup.select("body"):
        can_url = line.find_all('a')
        for can_u in can_url:
            url = can_u.get('href')
            if url!=None and url.startswith('https://'):
                url_li.append(url)
    return url_li
    
def get_latex(url, latex_li = []):
    req = requests.get(url)
    soup = BeautifulSoup(req.content, "html.parser")

    for line in soup.select("body"):
        text = soup.find_all('img')
        for t in text:
            latex = t.get('alt')
            if latex != None and latex.startswith('{\displaystyle'):
                latex_li.append(latex)
    return latex_li

if __name__ == '__main__':
    latex = []
    url_li = ["https://ko.wikipedia.org/wiki/%ED%86%B5%EA%B3%84%ED%95%99"]
    url_li.extend(get_url(url_li[0]))
    for url in url_li:
        latex.extend(get_latex(url))
    print(len(latex), latex[0])