import os
from os.path import join
import hashlib
import subprocess
from threading import Timer
from tqdm import tqdm

from latex_crawler import save_latex

DEVNULL = open(os.devnull, "w")

def run(cmd, timeout_sec):
    """Run cmd in the shell with timeout"""
    proc = subprocess.Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL)
    kill_proc = lambda p: p.kill()
    timer = Timer(timeout_sec, kill_proc, [proc])
    try:
        timer.start()
        stdout,stderr = proc.communicate()
    finally:
        timer.cancel()

def formula_to_image(formula, data_path = './'):
    formula = formula.strip("%")
    name = hashlib.sha1(formula.encode('utf-8')).hexdigest()[:15]

    # write formula into a .tex file
    tex_output = join(data_path, 'pdf')
    with open(tex_output + "/{}.tex".format(name), "w") as f:
        f.write(
        r"""\documentclass[preview]{standalone}
        \begin{document}
            $$ %s $$
        \end{document}""" % (formula))
    
    # call pdflatex to create pdf
    run("pdflatex -interaction=nonstopmode -output-directory={} {}".format(
        tex_output, tex_output+"/{}.tex".format(name)), 10)
    # call(['pdflatex', '-interaction=nonstopmode','-halt-on-error'])

    img_output = join(data_path, 'images')
    # call magick to convert the pdf into a png file
    run("magick convert -density {} -quality {} {} {}".format(200, 100,
        tex_output+"/{}.pdf".format(name), img_output+"/{}.png".format(name)),
        10)

def make_folder(path):
    folder_li = ['formulas', 'pdf', 'images']
    for folder in folder_li:
        if not os.path.isdir(join(path, folder)):
            os.mkdir(folder)


if __name__=='__main__':
    data_path = './'
    make_folder(path = data_path)
    # crawling 
    url = "https://ko.wikipedia.org/wiki/%ED%86%B5%EA%B3%84%ED%95%99" # 위키백과 통계학 페이지
    save_latex(url)

    formula_path = join(data_path,'formulas')
    formula_li = os.listdir(formula_path)
    # 일단 통계학.lst 로 테스트 -> for문으로 바꾸기 (+렌더링한 txt 파일은 다시 하지 않도록 조건문 넣기)
    print('Start rendering {} files ....'.format(len(formula_li)))
    for i in range(len(formula_li)):
        if formula_li[i].endswith('.txt'):
            print('Rendering {}'.format(formula_li[i]))
            with open(join(formula_path, formula_li[i]), 'r') as f:
                formulas = [formula.strip('\n') for formula in f.readlines()]
                [formula_to_image(formula, data_path = data_path) for formula in tqdm(formulas)]