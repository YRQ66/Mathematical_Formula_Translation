import os
from os.path import join
import hashlib
import subprocess
from threading import Timer
from tqdm import tqdm

DEVNULL = open(os.devnull, "w")

matches = []

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

def formula_to_image(idx, formula, data_path = './'):
    formula = formula.strip("%")
    name = hashlib.sha1(formula.encode('utf-8')).hexdigest()[:15]

    matches.append(str(idx)+','+formula+','+name)

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
    if not os.path.isdir(join(path, 'matches.csv')):
        open("matches.csv", "w")

if __name__=='__main__':
    data_path = './data'
    make_folder(path = data_path)

    formula_path = join(data_path,'formulas.txt')
    rendered_path = join(data_path,'rendered_formulas.txt')
    # formula_li = os.listdir(formula_path)
    # +렌더링한 txt 파일은 다시 하지 않도록 (formula - rendered_formula)
    with open(formula_path, 'r') as f:
        formulas = [formula.strip('\n') for formula in f.readlines()]
        formulas = set(formulas)

    with open(rendered_path, 'r') as f:
        renders = [rendered.strip('\n') for rendered in f.readlines()]
        renders = set(renders)

    remain_formula = formulas - renders
        
    print('Start rendering {} formulas ....'.format(len(remain_formula)))
    [formula_to_image(idx, formula) for idx, formula in enumerate(tqdm(formulas)) if formula not in renders]

    with open(rendered_path, "a") as f:
        f.write('\n'.join(remain_formula))

    # formula - image name 매칭 데이터프레임
    with open("matches.csv","a") as f:
        f.write("\n".join(matches))