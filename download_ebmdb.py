from bs4 import BeautifulSoup as BS4

import numpy as np
import os
import requests
import wget

repo_base = os.path.join(os.getcwd(), '../')
if not os.path.exists(os.path.join(repo_base, 'data')):
    os.makedirs(os.path.join(repo_base, 'data'))
if not os.path.exists(os.path.join(repo_base, 'data/bvh')):
    os.makedirs(os.path.join(repo_base, 'data/bvh'))
if not os.path.exists(os.path.join(repo_base, 'data/mvnx')):
    os.makedirs(os.path.join(repo_base, 'data/mvnx'))
if not os.path.exists(os.path.join(repo_base, 'data/tags')):
    os.makedirs(os.path.join(repo_base, 'data/tags'))

url = 'http://ebmdb.tuebingen.mpg.de/'
page = requests.get(url)
soup = BS4(page.content, 'html.parser')

header_row = soup.find_all('tr')[1]
tag_names = header_row.contents[1].text + '\n'
for c in range(5, len(header_row.contents), 2):
    tag_names += header_row.contents[c].text + '\n'
with open(os.path.join(repo_base, 'data/tags/tag_names.txt'), 'w') as tag_names_file:
    tag_names_file.write(tag_names)

data_rows = soup.find_all('tr')[3::2]
num_files = len(data_rows)
for l in range(num_files):
    id = str(int(data_rows[l].contents[1].text)).zfill(6)
    text = id + '\n'
    bvh_path = data_rows[l].contents[3].contents[1]['href'][2:]
    if not os.path.exists(os.path.join(repo_base, bvh_path)):
        wget.download(os.path.join(url, bvh_path), os.path.join(repo_base, 'data/bvh/' + id + '.bvh'))
    mvnx_path = data_rows[l].contents[3].contents[3]['href'][2:]
    if not os.path.exists(os.path.join(repo_base, mvnx_path)):
        wget.download(os.path.join(url, mvnx_path), os.path.join(repo_base, 'data/mvnx/' + id + '.mvnx'))
    for c in range(5, len(data_rows[l].contents) - 2, 2):
        text += data_rows[l].contents[c].text + '\n'
        with open(os.path.join(repo_base,  'data/tags/' + id + '.txt'), 'w') as tags_file:
            tags_file.write(text)
    print('\rDownloading files:\t{:.2f}%'.format(l * 100./num_files), end='')
print('\rDownloading files: done.')
