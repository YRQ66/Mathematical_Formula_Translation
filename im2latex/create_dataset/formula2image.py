import os
from os.path import join
import hashlib
import subprocess
from threading import Timer
from tqdm import tqdm
import pandas as pd
from subprocess import call

formula_li, image_li = [], []

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

def data_cleansing(formula_dir):
    with open(formula_dir, "r") as f:
        formulas = [i.strip('\n') for i in f.readlines()]

    formula = [f[1:-1] for f in formulas] # {} 문자열 제거

    # dataframe 만들기

    df = pd.DataFrame()

    df['idx'] = [i for i in range(len(formula))]
    df['formula'] = [i for i in formula]
    df['len'] = df.apply(lambda row: len(row['formula']), axis=1)

    # 5 < len(formula) < 500

    df_sorted = df.sort_values(by="len")
    df_filtered = df_sorted[df_sorted['len'] > 5 ]
    df_filtered = df_filtered[df_filtered['len'] < 500 ]
    df = df_filtered.reset_index(drop=True)

    return df

def formula_to_image(formula, IMAGE_DIR, PDF_DIR):
    name = hashlib.sha1(formula.encode('utf-8')).hexdigest()[:15]

    # write formula into a .tex file
    tex_output = PDF_DIR
    with open(tex_output + "/{}.tex".format(name), "w") as f:
        f.write(
        r"""\documentclass[preview]{standalone}
        \begin{document}
            $$ %s $$
        \end{document}""" % (formula))

    # call pdflatex to create pdf
    run("pdflatex -interaction=nonstopmode -output-directory={} {}".format(
        tex_output, tex_output+"/{}.tex".format(name)), 10)

    img_output = IMAGE_DIR
    # call magick to convert the pdf into a png file
    run("magick convert -density {} -quality {} {} {}".format(200, 100,
        tex_output+"/{}.pdf".format(name), img_output+"/{}.png".format(name)),
        10)

    formula_li.append(formula)
    image_li.append(name)

if __name__=='__main__':

    formula_path = './data/formulas.txt'
    formula_df = data_cleansing(formula_path)
    formulas = list(formula_df['formula'])

    IMAGE_DIR = './data/images'
    PDF_DIR = './data/pdf'
    try:
        os.mkdir(IMAGE_DIR)
    except OSError as e:
        pass #except because throws OSError if dir exists

    try:
        os.mkdir(PDF_DIR)
    except OSError as e:
        pass #except because throws OSError if dir exists
        
    print('Start rendering {} formulas ....'.format(len(set(formulas))))

    [formula_to_image(formula, IMAGE_DIR, PDF_DIR) for formula in tqdm(set(formulas))]

    # os.rmdir('./data/pdf')

    df = pd.DataFrame({'formula':formula_li, 'image':image_li})
    df.to_csv("../match.csv", mode='a', header=False)