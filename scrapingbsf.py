import bs4
from bs4 import BeautifulSoup
import pandas as pd
import requests

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"}
link="https://in.indeed.com/jobs?q=Data+Scientist&l=Bangalore%2C+Karnataka&start=10"
r=requests.get(link,headers=headers)
print(r.status_code)
soup=BeautifulSoup(r.text,"html.parser")
print(soup)
href_list =[]
base_link="https://in.indeed.com"
# find all element of tag
