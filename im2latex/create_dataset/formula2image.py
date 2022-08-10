import os
import glob
from os.path import join
import hashlib
import subprocess
from threading import Timer
from tqdm import tqdm
import pandas as pd
from subprocess import call

formula_li, image_li = [], []
error = [0]

DEVNULL = open(os.devnull, "w")

# 원래 코드는 A4 크기만큼의 pdf를 그대로 image 로 변환하였음 
# 아래 코드는 formula에 해당하는 크기만 image로 변환 (속도 빠름)
BASIC_SKELETON = r"""\documentclass[preview]{standalone}
        \begin{document}
            $$ %s $$
        \end{document}"""

rendering_setup = [BASIC_SKELETON, 
                    "convert -density 200 -quality 100 %s.pdf %s.png",
                    lambda filename: os.path.isfile(filename + ".png")]

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

def formula_to_image(formula):
    name = hashlib.sha1(formula.encode('utf-8')).hexdigest()[:15]
    # 이미지 렌더링에 필요한 3개의 단계를 거침

    # 1) 이미 렌더링 한 formula 는 다시 렌더링 하지 않음
    if rendering_setup[2](name):
        print("Skipping, already done: {}.png".format(name))
        return None

    # 2) Create latex source
    latex = rendering_setup[0] % formula
    # Write latex source
    with open(name+".tex", "w") as f:
        f.write(latex)

    # Call pdflatex to turn .tex into .pdf
    code = call(["pdflatex", '-interaction=nonstopmode', '-halt-on-error', name+".tex"],
                stdout=DEVNULL, stderr=DEVNULL)

    # code = call("pdflatex -interaction=nonstopmode {}".format("/{}.tex".format(name)), 10)
    if code != 0: # 렌더링 실패시 생성 파일 모두 삭제
        os.system("rm -rf "+name+"*")
        error[0]+=1
        print('Error execute {}'.format(error[0]))
        return None

    # 3) Turn .pdf to .png
    # Handles variable number of places to insert path.
    # i.e. "%s.tex" vs "%s.pdf %s.png"
    full_path_strings = rendering_setup[1].count("%")*(name,)
    code = call((rendering_setup[1] % full_path_strings).split(" "),
                stdout=DEVNULL, stderr=DEVNULL)

    #Remove files : png를 제외한 쓸데없는 파일 모두 삭제
    try:
        remove_temp_files(name)
    except Exception as e:
        # try-except in case one of the previous scripts removes these files
        # already
        return None

    # # Detect of convert created multiple images -> multi-page PDF
    # resulted_images = glob.glob(name+"-*") 
    
    # if code != 0:
    #     # Error during rendering, remove files and return None
    #     os.system("rm -rf "+name+"*")
    #     return None
    # elif len(resulted_images) > 1:
    #     # We have multiple images for same formula
    #     # Discard result and remove files
    #     for filename in resulted_images:
    #         os.system("rm -rf "+filename+"*")
    #     return None

    # matches.append(str(idx)+','+formula+','+name)
    formula_li.append(formula)
    image_li.append(name)

def remove_temp_files(name):
    """ Removes .aux, .log, .pdf and .tex files for name """
    os.remove(name+".aux")
    os.remove(name+".log")
    os.remove(name+".pdf")
    os.remove(name+".tex")

if __name__=='__main__':

    formula_path = './data/test.txt'
    formula_df = data_cleansing(formula_path)
    formulas = list(formula_df['formula'])

    IMAGE_DIR = './data/image'
    try:
        os.mkdir(IMAGE_DIR)
    except OSError as e:
        pass #except because throws OSError if dir exists

    # Change to image dir because textogif doesn't seem to work otherwise...
    oldcwd = os.getcwd()
    # Check we are not in image dir yet (avoid exceptions)
    if not IMAGE_DIR in os.getcwd():
        os.chdir(IMAGE_DIR)

    # with open('../bookmark.txt', "r") as f:
    #     start = int(f.readlines()[-1])
        
    print('Start rendering {} formulas ....'.format(len(formulas)))

    [formula_to_image(formula) for formula in tqdm(formulas)]

    # with open('../bookmark.txt', "a") as f:
    #     f.write('\n{}'.format(len(formulas)))

    df = pd.DataFrame({'formula':formula_li, 'image':image_li})
    df.to_csv("match.csv", mode='a', header=False)