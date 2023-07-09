# %%
import re
import requests
from bs4 import BeautifulSoup
import csv
import os
import numpy as np
from tkinter import scrolledtext
import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
import nbconvert
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from PIL import Image, ImageTk
import webbrowser
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd

# %%
import tkinter as tk
from PIL import Image, ImageTk
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import warnings
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import nltk
#nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from tkinter import filedialog
import os

import re
import pandas as pd
import bs4
import requests
import spacy
import re
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

from spacy.matcher import Matcher 
from spacy.tokens import Span 

import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

pd.set_option('display.max_colwidth', 200)
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
import csv
import os
import numpy as np
import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
import nbconvert
import nbformat

from nbconvert.preprocessors import ExecutePreprocessor
from PIL import Image, ImageTk
import webbrowser

# %%
def extract_sections_from_pdf(pdf_path):
    # Open the PDF file in read-binary mode
    pdf_file = open(pdf_path, 'rb')

    # Create a PDF resource manager and a text converter
    resource_manager = PDFResourceManager()
    output_string = StringIO()
    codec = 'utf-8'
    laparams = LAParams()

    converter = TextConverter(resource_manager, output_string, codec=codec, laparams=laparams)

    # Create a PDF page interpreter
    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    # Extract the text from the PDF file
    text = ''
    for page in PDFPage.get_pages(pdf_file):
        page_interpreter.process_page(page)

    text = output_string.getvalue()

    # Close the PDF file and the text converter
    pdf_file.close()
    converter.close()

    # Clean and preprocess the extracted text
    text = text.replace('\n', ' ').replace('\r', '')

    # Visual Cue Analysis
    title_match = re.search(r'\n?(.+)\n', text)
    title = title_match.group(1).strip() if title_match else ''

    # Keyword Matching and Pattern Recognition
    sections = {
        'Abstract': re.search(r'Abstract(.+?)Introduction', text, re.DOTALL),
        'Introduction': re.search(r'Introduction(.+?)Methodology', text, re.DOTALL),
        'Literature Review': re.search(r'Literature Review(.+?)Results', text, re.DOTALL),
        'Methodology': re.search(r'Methodology(.+?)Discussion', text, re.DOTALL),
        'Results': re.search(r'Results(.+?)Conclusion', text, re.DOTALL),
        'Discussion': re.search(r'Discussion(.+)', text, re.DOTALL)
    }

    # Extract section content
    extracted_content = {
        'Title': title,
        'Abstract': sections['Abstract'].group(1).strip() if sections['Abstract'] else '',
        'Introduction': sections['Introduction'].group(1).strip() if sections['Introduction'] else '',
        'Methodology': sections['Methodology'].group(1).strip() if sections['Methodology'] else '',
         }

    return extracted_content

def remove_duplicate_sentences(paragraph):
    # Split the paragraph into sentences
    sentences = paragraph.split('. ')

    # Remove any leading/trailing whitespace from each sentence
    sentences = [sentence.strip() for sentence in sentences]

    # Remove duplicate sentences while preserving the order
    unique_sentences = []
    for sentence in sentences:
        if sentence not in unique_sentences:
            unique_sentences.append(sentence)

    # Join the unique sentences to form a single paragraph
    unique_paragraph = '. '.join(unique_sentences)

    return unique_paragraph


# %%
def generate_summary(paragraph, n):
    sentences = nltk.sent_tokenize(paragraph)
    words = [nltk.word_tokenize(sentence) for sentence in sentences]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    summarize_text = []
    similarity_matrix = cosine_similarity(tfidf_matrix)
    graph = nx.Graph()
    num_sentences = similarity_matrix.shape[0]
    for i in range(num_sentences):
        for j in range(i+1, num_sentences):
            similarity = similarity_matrix[i,j]
            graph.add_edge(i, j, weight=similarity)
    pagerank_scores = nx.pagerank(graph)
    ranked_sentences = sorted(((score, index) for index, score in pagerank_scores.items()), reverse=True)
    ranked_sentences_list = []
    for score, index in ranked_sentences:
        ranked_sentences_list.append(sentences[index])
    num_elements = len(ranked_sentences_list)
    if(num_elements > n):
        joined_str = "".join(ranked_sentences_list[:n])
    else:
        joined_str = "".join(ranked_sentences_list[:num_elements])
    return joined_str 

# %%
def abstractive_summary(text, max_length=512, stride=256):
    # Load the T5 tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')

    # Split the input text into smaller segments
    text_segments = [text[i:i+max_length] for i in range(0, len(text), stride)]

    # Generate a summary for each text segment
    summaries = []
    for segment in text_segments:
        # Encode the segment using the tokenizer
        input_ids = tokenizer.encode("summarize: " + segment, return_tensors='pt')

        # Generate the summary using the T5 model
        summary_ids = model.generate(input_ids=input_ids,
                                      max_length=20,
                                      num_beams=5,
                                      early_stopping=True)

        # Decode the summary tokens and append to the list of summaries
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    # Combine the generated summaries into a single string
    summary = ' '.join(summaries)
    return summary

# %%


# %%
def create_soup(url2):
    headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
    r = requests.get(url2,headers=headers)
    htmlContent = r.content
    soup = BeautifulSoup(htmlContent, 'html.parser')
    return soup
def search_engine(keyword):
    headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
    url = 'https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q= '+keyword+'research papers'
    response=requests.get(url,headers=headers)
    page_contents = response.text
    doc = BeautifulSoup(page_contents,'html.parser')
    ############
    div_tags = doc.find_all('div', class_='gs_rs')
    text_list = [' '.join([token.get_text(strip=True) if token.name != 'b' else f' {token.get_text(strip=True)} ' for token in div_tag.contents]) for div_tag in div_tags]
    ############
    paper_names_tag = doc.select('[data-lid]')
    paper_names = []
    paper_name = []
    for tag in paper_names_tag:
        paper_names.append(tag.select('h3')[0].get_text())
    for i in range(len(paper_names)):
        title = paper_names[i].replace('[BOOK][B] ', '').replace('[PDF][PDF] ', '').replace('[HTML][HTML] ', '') 
        paper_name.append(title)
    links = paper_names_tag[0].find('h3')
    link_tag = doc.find_all('h3',{"class" : "gs_rt"})
    links = []
    for i in range(len(link_tag)) :
        links.append(link_tag[i].a['href'])
    Abstract = []
    j = 0
    for i in links:
        try:
            abstract = "none"
            url1 = i

            if "www.hbs.edu" in url1:
                soup = create_soup(url1)
                abstract=soup.find("div",{"class":"description-content add-underline"}).get_text()
            elif"www.tandfonline.com" in url1:
                soup = create_soup(url1)
                for tag in soup.find_all("meta"):
                    if tag.get("name", None) == "Description":
                        abstract = tag.get("content", None)
            elif "meridian.allenpress.com" in url1:
                soup = create_soup(url1)
                for tag in soup.find_all("meta"):
                    if tag.get("property", None) == "og:description":
                        abstract = tag.get("content", None)
            elif "ieee.org" in url1:
                soup = create_soup(url1)
                for tag in soup.find_all("meta"):
                    if tag.get("property", None) == "og:description":
                        abstract = tag.get("content", None)
            elif "www.mdpi.com" in url1:
                soup = create_soup(url1)
                for tag in soup.find_all("meta"):
                    if tag.get("property", None) == "og:description":
                        abstract = tag.get("content", None)
            elif "journals.plos.org" in url1:
                soup = create_soup(url1)
                abstract = soup.find_all('p')[11].get_text()  
            elif "scitation.org" in url1:
                soup = create_soup(url1)
                for tag in soup.find_all("meta"):
                    if tag.get("name", None) == "citation_abstract":
                        abstract = tag.get("content", None)
            elif "peerj.com" in url1: 
                soup = create_soup(url1)
                for tag in soup.find_all("meta"):
                    if tag.get("name", None) == "description":
                        abstract = tag.get("content", None)
            elif "direct.mit.edu" in url1:
                soup = create_soup(url1)
                for tag in soup.find_all("meta"):
                    if tag.get("property", None) == "og:description":
                        abstract = tag.get("content", None)
            elif "content.iospress.com" in url1:
                soup = create_soup(url1)
                abstract = soup.find("h1").get("data-abstract")
            elif "aip.scitation.org" in url1:
                soup = create_soup(url1)
                for tag in soup.find_all("meta"):
                    if tag.get("name", None) == "citation_abstract":
                        abstract = tag.get("content", None)
            elif "taylorfrancis.com" in url1:
                soup = create_soup(url1)
                for tag in soup.find_all("meta"):
                    if tag.get("name", None) == "citation_abstract":
                        abstract = tag.get("content", None)    
            elif "www.nature.com"in url1:
                soup = create_soup(url1)
                for tag in soup.find_all("meta"):
                    if tag.get("name", None) == "dc.description":
                        abstract = tag.get("content", None)
            elif "pubs.rsc.org" in url1:
                soup = create_soup(url1)
                for tag in soup.find_all("meta"):
                    if tag.get("name", None) == "description":
                        abstract = tag.get("content", None)

            else:
                if "springer.com" in url1:

                    if ".pdf" in url1:
                        abstract = "none"
                    elif "link.springer.com/content/pdf" in url1:
                        soup = create_soup(url1)
                        abstract=soup.find("section",{"data-title":"Reviews"}).get_text()
                    else:
                        soup = create_soup(url1)
                        abstract=soup.find("div",{"class":"c-article-section__content"}).get_text()

                else:
                    if "acm.org" in url1:
                        soup = create_soup(url1)
                        abstract=soup.find("div",{"class":"abstractSection abstractInFull"}).get_text()
                    elif "www.nowpublishers.com" in url1:
                        soup = create_soup(url1)
                        abstract = soup.find_all('p')[2].get_text()
                    else:
                        if "proceedings.mlr.press" in url1:
                            soup = create_soup(url1)
                            abstract=soup.find("div",{"class":"abstract"}).get_text()
                        elif "proceedings.neurips.cc" in url1:
                            soup = create_soup(url1)
                            abstract = soup.find_all('p')[3].get_text()
                        else:
                            if "thecvf.com" in url1:
                                soup = create_soup(url1)
                                abstract=soup.find("div",id="abstract").get_text()
                            else:
                                if "arxiv.org" in url1:
                                    soup = create_soup(url1)
                                    for tag in soup.find_all("meta"):
                                        if tag.get("name", None) == "citation_abstract":
                                            abstract = tag.get("content", None)
                                else:
                                    if "iopscience.iop.org" in url1:
                                        soup = create_soup(url1)
                                        abstract = soup.find_all('p')[12].get_text()
                                    else:
                                        if "projecteuclid.org" in url1:
                                            soup = create_soup(url1)
                                            for tag in soup.find_all("meta"):
                                                if tag.get("name", None) == "citation_abstract":
                                                    abstract = tag.get("content", None)
                                        else:
                                            if "emerald.com" in url1 :
                                                soup = create_soup(url1)
                                                for tag in soup.find_all("meta"):
                                                    if tag.get("name", None) == "dc.Description":
                                                        abstract = tag.get("content", None)
                                            else:
                                                if "ojs.aaai.org" in url1 :
                                                    soup = create_soup(url1)
                                                    for tag in soup.find_all("meta"):
                                                        if tag.get("name", None) == "DC.Description":
                                                            abstract = tag.get("content", None)
                                                elif "www.ingentaconnect.com" in url1:
                                                    soup = create_soup(url1)
                                                    abstract=soup.find("div",id="Abst").get_text()
                                                elif "academic.oup.com" in url1:
                                                    soup = create_soup(url1)
                                                    for tag in soup.find_all("meta"):
                                                        if tag.get("property", None) == "og:description":
                                                            abstract = tag.get("content", None)
                                                elif "bmcbioinformatics.biomedcentral.com" in url1:
                                                    soup = create_soup(url1)
                                                    for tag in soup.find_all("meta"):
                                                        if tag.get("property", None) == "description":
                                                            abstract = tag.get("content", None)
                                                else:
                                                    abstract  = text_list[j]
        except:
            pass
                       
        Abstract.append(abstract)
        j = j+1
    s = summary_csv(keyword,paper_name,links,Abstract,doc)
    return s   

# %%
def search_engine1(keyword,keyword1):
    headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
    url = 'https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q= '+keyword+'research papers'
    response=requests.get(url,headers=headers)
    page_contents = response.text
    doc = BeautifulSoup(page_contents,'html.parser')
    ############
    div_tags = doc.find_all('div', class_='gs_rs')
    text_list = [' '.join([token.get_text(strip=True) if token.name != 'b' else f' {token.get_text(strip=True)} ' for token in div_tag.contents]) for div_tag in div_tags]
    ############
    paper_names_tag = doc.select('[data-lid]')
    paper_names = []
    paper_name = []
    for tag in paper_names_tag:
        paper_names.append(tag.select('h3')[0].get_text())
    for i in range(len(paper_names)):
        title = paper_names[i].replace('[BOOK][B] ', '').replace('[PDF][PDF] ', '').replace('[HTML][HTML] ', '') 
        paper_name.append(title)
    links = paper_names_tag[0].find('h3')
    link_tag = doc.find_all('h3',{"class" : "gs_rt"})
    links = []
    for i in range(len(link_tag)) :
        links.append(link_tag[i].a['href'])
    Abstract = []
    j = 0
    for i in links:
        try:
            abstract = "none"
            url1 = i

            if "www.hbs.edu" in url1:
                soup = create_soup(url1)
                abstract=soup.find("div",{"class":"description-content add-underline"}).get_text()
            elif"www.tandfonline.com" in url1:
                soup = create_soup(url1)
                for tag in soup.find_all("meta"):
                    if tag.get("name", None) == "Description":
                        abstract = tag.get("content", None)
            elif "meridian.allenpress.com" in url1:
                soup = create_soup(url1)
                for tag in soup.find_all("meta"):
                    if tag.get("property", None) == "og:description":
                        abstract = tag.get("content", None)
            elif "ieee.org" in url1:
                soup = create_soup(url1)
                for tag in soup.find_all("meta"):
                    if tag.get("property", None) == "og:description":
                        abstract = tag.get("content", None)
            elif "www.mdpi.com" in url1:
                soup = create_soup(url1)
                for tag in soup.find_all("meta"):
                    if tag.get("property", None) == "og:description":
                        abstract = tag.get("content", None)
            elif "journals.plos.org" in url1:
                soup = create_soup(url1)
                abstract = soup.find_all('p')[11].get_text()  
            elif "scitation.org" in url1:
                soup = create_soup(url1)
                for tag in soup.find_all("meta"):
                    if tag.get("name", None) == "citation_abstract":
                        abstract = tag.get("content", None)
            elif "peerj.com" in url1: 
                soup = create_soup(url1)
                for tag in soup.find_all("meta"):
                    if tag.get("name", None) == "description":
                        abstract = tag.get("content", None)
            elif "direct.mit.edu" in url1:
                soup = create_soup(url1)
                for tag in soup.find_all("meta"):
                    if tag.get("property", None) == "og:description":
                        abstract = tag.get("content", None)
            elif "content.iospress.com" in url1:
                soup = create_soup(url1)
                abstract = soup.find("h1").get("data-abstract")
            elif "aip.scitation.org" in url1:
                soup = create_soup(url1)
                for tag in soup.find_all("meta"):
                    if tag.get("name", None) == "citation_abstract":
                        abstract = tag.get("content", None)
            elif "taylorfrancis.com" in url1:
                soup = create_soup(url1)
                for tag in soup.find_all("meta"):
                    if tag.get("name", None) == "citation_abstract":
                        abstract = tag.get("content", None)    
            elif "www.nature.com"in url1:
                soup = create_soup(url1)
                for tag in soup.find_all("meta"):
                    if tag.get("name", None) == "dc.description":
                        abstract = tag.get("content", None)
            elif "pubs.rsc.org" in url1:
                soup = create_soup(url1)
                for tag in soup.find_all("meta"):
                    if tag.get("name", None) == "description":
                        abstract = tag.get("content", None)

            else:
                if "springer.com" in url1:

                    if ".pdf" in url1:
                        abstract = "none"
                    elif "link.springer.com/content/pdf" in url1:
                        soup = create_soup(url1)
                        abstract=soup.find("section",{"data-title":"Reviews"}).get_text()
                    else:
                        soup = create_soup(url1)
                        abstract=soup.find("div",{"class":"c-article-section__content"}).get_text()

                else:
                    if "acm.org" in url1:
                        soup = create_soup(url1)
                        abstract=soup.find("div",{"class":"abstractSection abstractInFull"}).get_text()
                    elif "www.nowpublishers.com" in url1:
                        soup = create_soup(url1)
                        abstract = soup.find_all('p')[2].get_text()
                    else:
                        if "proceedings.mlr.press" in url1:
                            soup = create_soup(url1)
                            abstract=soup.find("div",{"class":"abstract"}).get_text()
                        elif "proceedings.neurips.cc" in url1:
                            soup = create_soup(url1)
                            abstract = soup.find_all('p')[3].get_text()
                        else:
                            if "thecvf.com" in url1:
                                soup = create_soup(url1)
                                abstract=soup.find("div",id="abstract").get_text()
                            else:
                                if "arxiv.org" in url1:
                                    soup = create_soup(url1)
                                    for tag in soup.find_all("meta"):
                                        if tag.get("name", None) == "citation_abstract":
                                            abstract = tag.get("content", None)
                                else:
                                    if "iopscience.iop.org" in url1:
                                        soup = create_soup(url1)
                                        abstract = soup.find_all('p')[12].get_text()
                                    else:
                                        if "projecteuclid.org" in url1:
                                            soup = create_soup(url1)
                                            for tag in soup.find_all("meta"):
                                                if tag.get("name", None) == "citation_abstract":
                                                    abstract = tag.get("content", None)
                                        else:
                                            if "emerald.com" in url1 :
                                                soup = create_soup(url1)
                                                for tag in soup.find_all("meta"):
                                                    if tag.get("name", None) == "dc.Description":
                                                        abstract = tag.get("content", None)
                                            else:
                                                if "ojs.aaai.org" in url1 :
                                                    soup = create_soup(url1)
                                                    for tag in soup.find_all("meta"):
                                                        if tag.get("name", None) == "DC.Description":
                                                            abstract = tag.get("content", None)
                                                elif "www.ingentaconnect.com" in url1:
                                                    soup = create_soup(url1)
                                                    abstract=soup.find("div",id="Abst").get_text()
                                                elif "academic.oup.com" in url1:
                                                    soup = create_soup(url1)
                                                    for tag in soup.find_all("meta"):
                                                        if tag.get("property", None) == "og:description":
                                                            abstract = tag.get("content", None)
                                                elif "bmcbioinformatics.biomedcentral.com" in url1:
                                                    soup = create_soup(url1)
                                                    for tag in soup.find_all("meta"):
                                                        if tag.get("property", None) == "description":
                                                            abstract = tag.get("content", None)
                                                else:
                                                    abstract  = text_list[j]
        except:
            pass
                       
        Abstract.append(abstract)
        j = j+1
    s = summary_csv1(keyword,paper_name,links,Abstract,doc,keyword1)
    return s

# %%
def summary_csv(keyword,paper_name,links,Abstract,doc):
    year_of_publish = []
    papers = doc.select('.gs_a')
    for paper in papers:
        paper_text = paper.text.strip()
        year_match = re.search(r'\b\d{4}\b', paper_text)
        if year_match:
            year = year_match.group(0)
            year_of_publish.append(year)
    Summary = []
    for i in Abstract:
        if i == "none":
            Summary.append(i)
        else:
            char_count = len(i)
            if (char_count <= 110):
                Summary.append(i)
            else:
                Summary.append(generate_summary(i, 1))
    authors_list = []
    authors_tag = doc.find_all("div", {"class": "gs_a"})
    
    for i in authors_tag:
          authors_list.append(i.text)
    if os.path.isfile('searched_data.csv'):
        old_data = pd.read_csv('searched_data.csv')
        l = old_data.iloc[-1, 0]
        index = np.zeros(10, dtype=int)
        for i in range(10):
            index[i] = l+1;
            l = l+1;
        new_data = {
        'INDEX':index,
        'SEARCHED KEYWORD' : keyword,
        'PAPER TITLE' : paper_name,
        'AUTHER NAME' : authors_list,
        'URL' : links,
        'YEAR': year_of_publish,
        'ABSTRACT':Abstract,
        'SUMMARY':Summary

            }
        new_df = pd.DataFrame(new_data)
        new_df.to_csv('searched_data.csv', mode='a', index=False, header = False)
        return new_df

    else:
        index = [0,1,2,3,4,5,6,7,8,9]
        new_data = {
        'INDEX':index,
        'SEARCHED KEYWORD' : keyword,
        'PAPER TITLE' : paper_name,
        'AUTHER NAME' : authors_list,
        'URL' : links,
        'YEAR': year_of_publish,
        'ABSTRACT':Abstract,
        'SUMMARY':Summary
            }
        papers_df = pd.DataFrame(new_data)
        papers_df.to_csv('searched_data.csv', mode='a', index=False, header = True)
        return papers_df

# %%
def summary_csv1(keyword,paper_name,links,Abstract,doc,keyword1):
    year_of_publish = []
    papers = doc.select('.gs_a')
    for paper in papers:
        paper_text = paper.text.strip()
        year_match = re.search(r'\b\d{4}\b', paper_text)
        if year_match:
            year = year_match.group(0)
            year_of_publish.append(year)
    Summary = []
    for i in Abstract:
        if i == "none":
            Summary.append(i)
        else:
            char_count = len(i)
            if (char_count <= 110):
                Summary.append(i)
            else:
                Summary.append(generate_summary(i, 1))
    authors_list = []
    authors_tag = doc.find_all("div", {"class": "gs_a"})
    
    for i in authors_tag:
          authors_list.append(i.text)
    if os.path.isfile(keyword1):
        old_data = pd.read_csv(keyword1)
        l = old_data.iloc[-1, 0]
        index = np.zeros(10, dtype=int)
        for i in range(10):
            index[i] = l+1;
            l = l+1;
        new_data = {
        'INDEX':index,
        'SEARCHED KEYWORD' : keyword,
        'PAPER TITLE' : paper_name,
        'AUTHER NAME' : authors_list,
        'URL' : links,
        'YEAR': year_of_publish,
        'ABSTRACT':Abstract,
        'SUMMARY':Summary}
        new_df = pd.DataFrame(new_data)
        new_df.to_csv(keyword1, mode='a', index=False, header = False)
        return new_df

    else:
        index = [0,1,2,3,4,5,6,7,8,9]
        new_data = {
        'INDEX':index,
        'SEARCHED KEYWORD' : keyword,
        'PAPER TITLE' : paper_name,
        'AUTHER NAME' : authors_list,
        'URL' : links,
        'YEAR': year_of_publish,
        'ABSTRACT':Abstract,
        'SUMMARY':Summary
            }
        papers_df = pd.DataFrame(new_data)
        papers_df.to_csv(keyword1, mode='a', index=False, header = True)
        return papers_df

# %%
class KeywordInputApp1:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Scholar Search Engine')

        # Set window size to fill screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")

        # Create a canvas to put the scrollable frame in
        canvas = tk.Canvas(self.root, width=screen_width, height=screen_height)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a scrollable frame to put the UI elements in
        scrollable_frame = tk.Frame(canvas)

        # Add a scrollbar to the canvas to scroll the frame
        scrollbar = tk.Scrollbar(self.root, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=scrollbar.set)

        # Bind the canvas to the scrollbar
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')

        # Add UI elements to the scrollable frame
        UI_label_font = ("AkayaKanadaka", 40)
        self.UI_label = tk.Label(scrollable_frame, text='Scholar Search Engine', font=UI_label_font)
        self.UI_label.pack(padx=20, pady=20, anchor="center")
        label_font = ("Arial", 20)
        
        self.keyword_label = tk.Label(scrollable_frame, text='Enter the keyword:', font=label_font)
        self.keyword_label.pack(padx=20, pady=20, anchor="center")

        self.keyword_entry = tk.Entry(scrollable_frame, width=50, font=label_font)
        self.keyword_entry.pack(pady=20, anchor="center")
        
        self.keyword_label = tk.Label(scrollable_frame, text='Enter a new CSV file name only if you wish to save the search results in a new file:', font=label_font)
        self.keyword_label.pack(padx=20, pady=20, anchor="center")
        
        self.keyword_entry1 = tk.Entry(scrollable_frame, width=50, font=label_font)
        self.keyword_entry1.pack(pady=20, anchor="center")
        
        self.submit_button = tk.Button(scrollable_frame, text='Submit', font=label_font, command=lambda: self.submit())
        self.submit_button.pack(padx=20, pady=20, anchor="center")

        
        self.num_results_label = tk.Label(scrollable_frame, text='', font=("Arial", 14))
        self.num_results_label.pack(pady=10, anchor='w')

        self.result_canvas = tk.Canvas(scrollable_frame, width=screen_width, height=screen_height, bd=0, highlightthickness=0)
        self.result_canvas.pack(pady=20, anchor="w")

        self.result_frame = tk.Frame(self.result_canvas)
        self.result_canvas.create_window((0, 0), window=self.result_frame, anchor='nw')

        # update the result canvas scroll region after adding new search results
        self.result_frame.bind('<Configure>', lambda e: self.result_canvas.configure(scrollregion=self.result_canvas.bbox('all')))

        self.root.mainloop()
        
    def submit(self):
        keyword = self.keyword_entry.get()
        keyword1 = self.keyword_entry1.get()
        if(keyword1 == ""):
            df = search_engine(keyword)
            # Create a Canvas widget and set its size
            #root =tk.Tk()
            canvas_toplevel = tk.Toplevel(self.root)
            canvas_toplevel.title(f'Search Results ')
            canvas_toplevel.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}")

            # Create a Canvas widget inside the Toplevel widget and set its size
            canvas = tk.Canvas(canvas_toplevel)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            # Create a Scrollbar widget and associate it with the Canvas
            scrollbar = tk.Scrollbar(canvas , orient="vertical", command=canvas.yview)
            canvas.configure(yscrollcommand=scrollbar.set)

            # Add widgets to the Canvas
            frame = tk.Frame(canvas)
            result_label = tk.Label(frame,text=f'Search Results for "{keyword}"', font=("AkayaKanadaka",40))
            result_label.pack(padx=20, pady=20, anchor="center")
            for i, row in df.iterrows():
                title = row['PAPER TITLE']
                abstract = row['ABSTRACT']
                url = row['URL']
                title_label = tk.Label(frame, text=title, font=("Arial", 16), fg='blue', cursor='hand2')
                title_label.pack(pady=5, padx=5, anchor='w')
                title_label.bind('<Button-1>', lambda e, u=url: webbrowser.open_new(u))
                abstract_label = tk.Label(frame, text=abstract, font=("Arial", 12), wraplength=1000, justify='left')
                abstract_label.pack(pady=5, padx=5, anchor='w')
            # Pack the widgets onto the window
            scrollbar.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)
            canvas.create_window((0,0), window=frame, anchor="nw")
            # Add a button widget for clustering similar topics
            cluster_button = tk.Button(frame, text="Similar Topics", font=("Arial", 12),command=lambda: self.cluster_papers(df))
            cluster_button.pack(pady=10)

            # Pack the widgets onto the window
            scrollbar.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)
            canvas.create_window((0, 0), window=frame, anchor="nw")

            # Configure the Canvas to resize with the window
            frame.bind("<Configure>", lambda event, canvas=canvas: canvas.configure(scrollregion=canvas.bbox("all")))

            root.mainloop()


        else:
            if not keyword1.endswith(".csv"):
                messagebox.showerror("Invalid CSV Name", "Invalid CSV name. Enter a string ending with '.csv'.")
            else:
                df = search_engine1(keyword,keyword1)
                # Create a Canvas widget and set its size
                #root =tk.Tk()
                canvas_toplevel = tk.Toplevel(self.root)
                canvas_toplevel.title(f'Search Results ')
                canvas_toplevel.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}")

                # Create a Canvas widget inside the Toplevel widget and set its size
                canvas = tk.Canvas(canvas_toplevel)
                canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                # Create a Scrollbar widget and associate it with the Canvas
                scrollbar = tk.Scrollbar(canvas , orient="vertical", command=canvas.yview)
                canvas.configure(yscrollcommand=scrollbar.set)

                # Add widgets to the Canvas
                frame = tk.Frame(canvas)
                result_label = tk.Label(frame,text=f'Search Results for "{keyword}"', font=("AkayaKanadaka",40))
                result_label.pack(padx=20, pady=20, anchor="center")
                for i, row in df.iterrows():
                    title = row['PAPER TITLE']
                    abstract = row['ABSTRACT']
                    url = row['URL']
                    title_label = tk.Label(frame, text=title, font=("Arial", 16), fg='blue', cursor='hand2')
                    title_label.pack(pady=5, padx=5, anchor='w')
                    title_label.bind('<Button-1>', lambda e, u=url: webbrowser.open_new(u))
                    abstract_label = tk.Label(frame, text=abstract, font=("Arial", 12), wraplength=1000, justify='left')
                    abstract_label.pack(pady=5, padx=5, anchor='w')
                # Pack the widgets onto the window
                scrollbar.pack(side="right", fill="y")
                canvas.pack(side="left", fill="both", expand=True)
                canvas.create_window((0,0), window=frame, anchor="nw")
                
                # Add a button widget for clustering similar topics
                cluster_button = tk.Button(frame, text="Similar Topics", font=("Arial", 12),command=lambda: self.cluster_papers(df))
                cluster_button.pack(pady=10)

                # Pack the widgets onto the window
                scrollbar.pack(side="right", fill="y")
                canvas.pack(side="left", fill="both", expand=True)
                canvas.create_window((0, 0), window=frame, anchor="nw")

                # Configure the Canvas to resize with the window
                frame.bind("<Configure>", lambda event, canvas=canvas: canvas.configure(scrollregion=canvas.bbox("all")))

                root.mainloop()
                
    def cluster_papers(self,df):
        
        abstracts = df['ABSTRACT'].fillna('')  # fill NaN with empty string
        titles = df['PAPER TITLE'].fillna('')  # fill NaN with empty string
        text = titles + ' ' + abstracts  # Combine title and abstract
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(text)
        distance_matrix = pairwise_distances(tfidf_matrix, metric='cosine')
        similarity_matrix = 1 - distance_matrix
        similarity_threshold = 0.1  # Set the similarity threshold here
        max_similar_papers = 3  # Set the maximum number of similar papers to show here

        # Create a Tkinter window
        window = tk.Tk()
        window.title("Similar Papers")

        # Create a scrolled text widget to display the results
        results_text = scrolledtext.ScrolledText(window, width=50, height=20)
        results_text.pack()

        for i, row in df.iterrows():
            paper_index = row['INDEX']
            paper_title = row['PAPER TITLE']
            similar_papers = []
            for j, sim_score in sorted(enumerate(similarity_matrix[i]), key=lambda x: x[1], reverse=True):
                if j != i and sim_score > similarity_threshold:
                    similar_papers.append((df.iloc[j]['INDEX'], sim_score))
                    if len(similar_papers) >= max_similar_papers:
                        break
            if len(similar_papers) > 0:
                results_text.insert(tk.END, f'Papers similar to "{paper_title}" (index {paper_index}):\n')
                for p in similar_papers:
                    results_text.insert(tk.END, f'- {p[0]} (similarity score: {p[1]:.2f})\n')
                results_text.insert(tk.END, '\n')
            print("Similar paper")
        # Start the Tkinter main loop
        window.mainloop()



# %%
class KeywordInputApp():
    global total
    global kg_df_g
    global sliced
    global paragraph
    total = pd.read_csv('sorted_arxiv4L_with_year.csv')
    #total['Year'] = 2022
    # Generate random values between 2000 and 2022
    
    
    total['title'] = total['title'].str.casefold()
    total['author'] = total['author'].str.casefold()
    total['index_col'] = total.index
    sliced = total[:4000]
    
    # submit buttons:
              
    
    def submit11(self): 
        app = KeywordInputApp1()
         
    
    def submit3(self):
        dfw = sliced
        keyword = self.keyword_entry.get()
        
        def KGT(m):
            if(type(m)!=int):
                m = int(m)
            #print(m)
            #print(dfw[m:m+1])
            cso_df =  dfw[m:m+1]
            cso_df['union'] = ''
            #cso_df.rename(columns = {'ABSTRACT':'abstract','PAPER TITLE':'title'}, inplace = True)
            d = cso_df[['abstract','title','union']]
            d.to_csv('d.csv')
            
            csv_filename = 'd.csv'
            i = -1
            with open(csv_filename) as f:
                reader = csv.DictReader(f)

                for row in reader:
                    i = i+1
                    cc = cp(workers = 1, modules = "both", enhancement = "first", explanation = True)
                    result = cc.batch_run(row)
                    cso_df.loc[i, 'union'] = str(result['abstract']['union']+result['title']['union'])
                    #print(result['abstract']['union']+result['title']['union'])

            candidate_sentences = cso_df
            candidate_sentences['union'][0] = candidate_sentences['union'][0:1].replace("'","")
            candidate_sentences['union'][0] = candidate_sentences['union'][0:1].replace("[","")
            candidate_sentences['union'][0] = candidate_sentences['union'][0:1].replace("]","")
            candidate_sentences['title'][0] = candidate_sentences['title'][0:1].replace("\n","")
            candidate_sentences['author'][0] = candidate_sentences['author'][0:1].replace("and", ",")
            
            
            articles = ['A ', 'An ', 'The ',' study of ', 'On ']
            problem = [' based on ', ' on the Size of ', ' by means of ', ' for ', ' with ',' in ',' of ', ' over ', ' on ', ]
    
            text = candidate_sentences['title'][0:1].str.casefold()
            text = text.to_string()
            text = text[5:]
            
            for i in articles:
                if i in text:
                    text = text.replace(i, ' ')     
            
            for i in problem:
                if i in text:
                    text = text.replace(i, ',',1)
                    break 
            
            Data = text
            string = text
            if len(string)<20:
                pos =  string.index(" ") 
            else:
                pos = string.index(' ', string.index(" ") + 1)
            string = string[:pos] + ' \n ' + string[pos + 1:] 
            text = string
        #print("Hi from 40" + text)
            text = pd.Series(text)
            
            df_s = pd.Series({'source' : []})
            df_e = pd.Series({'edge' : 1})
            df_t = pd.Series({'target': []})
            dflink = pd.DataFrame()
            
            df_s = df_s.append(candidate_sentences['Year'][0:1], ignore_index=True)
            df_s = df_s.append(candidate_sentences['author'][0:1].str.split(",", n = 1, expand = True)[0], ignore_index=True)
            
            Data = Data[5:]
            flag=0
            print(Data)
            title_prob = list(Data.split(",")[0].split(" "))
            print(title_prob)
            for i in range(len(title_prob)):
                if candidate_sentences['union'][i:i+1].str.contains(title_prob[i]).any():
                    df_s = df_s.append(text.str.split(",", n = 1, expand = True)[0],ignore_index=True)
                    df_s = df_s.append(text.str.split(",", n = 1, expand = True)[1],ignore_index=True)
                    flag = 1
                    break
                if flag!=1:
                    df_s = df_s.append(text.str.split(",", n = 1, expand = True)[1],ignore_index=True)
                    df_s = df_s.append(text.str.split(",", n = 1, expand = True)[0], ignore_index=True)
                    break
                
           
    
            df_s = df_s.drop(0)
            df_s.index = np.arange(0,len(df_s))
        #x = candidate_sentences['title'][0].replace("\n ", "")

            df_e = pd.concat([df_e]*4, ignore_index=True)
    
    #print(df_s)
    
            x = candidate_sentences['title'][0]
            string = x.to_string()
            if len(string)<15:
                pos = string.index(" ") + 1
           
    
            string = string[:pos] + ' \n ' + string[pos + 1:] 
            #x = string
            x = candidate_sentences['title'][0:1]
            x = pd.Series(x)
            df_t = pd.concat([x]*4, ignore_index=True)
    
            string = df_s[2]
            if len(string)>10:
                pos = string.index(' ') 
                string = string[:pos] + ' \n ' + string[pos + 1:] 
                df_s[2] = string
                
    
            
            
            source = [i for i in df_s]
            relations = [i for i in df_e]
            target = [i for i in df_t]
    
            
            kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})
    
            # create a directed-graph from a dataframe
            G=nx.from_pandas_edgelist(kg_df, "source", "target",edge_attr=True, create_using=nx.MultiDiGraph())
    
            # Let’s plot the network:

      
            window = Tk()
            window.title("Graph")
           
            
           
            f = plt.figure(figsize=(14,5), dpi=100)
            pos = nx.spring_layout(G)
            d = dict(G.degree)

            center_node =  1 
            edge_nodes = set(G) - {center_node}

            pos = nx.circular_layout(G.subgraph(edge_nodes))
            pos[center_node] = np.array([0, 0])

            # Ensures the nodes around the circle are evenly distributed
            nx.draw(G, pos=pos, with_labels=True, node_shape="s", node_color="none", bbox=dict(facecolor="white", edgecolor='black', boxstyle='round,pad=0.8'))

            #Set the geometry
            window.geometry("950x300")

            window.eval('tk::PlaceWindow . center')
            # create matplotlib canvas using figure `f` and assign to widget `window`
            canvas = FigureCanvasTkAgg(f, window)
            
            # get canvas as tkinter's widget and `grid` in widget `window`
            canvas.get_tk_widget().grid(row=1, column=5, padx=25, pady=40)
            
            
            x,y=pos[kg_df['source'][0]]
            plt.text(x,y+0.14,s= 'Year', weight='bold', fontsize=14, color="green", bbox=dict(facecolor='white', edgecolor='white', alpha=0.0),horizontalalignment='center')
            CloseButton = tk.Button(window, text='Year', command = lambda m = 1:open_year(kg_df['target'][0],kg_df['source'][0]))
            CloseButton.place(x=x+240,y=y+500)
            
            x,y=pos[kg_df['source'][1]]
            plt.text(x,y+0.2,s= 'First Author', weight='bold', fontsize=14, color="green", bbox=dict(facecolor='white', edgecolor='white', alpha=0.0),horizontalalignment='center')
            CloseButton = tk.Button(window, text='Author', command = lambda m = 1: author_search(kg_df['source'][1]))
            CloseButton.place(x=x+280,y=y+500)
            
            x,y=pos[kg_df['source'][2]]
            plt.text(x,y+0.25,s='Technique', weight='bold', color="green", fontsize=14, bbox=dict(facecolor='white', edgecolor='white', alpha=0.0),horizontalalignment='center')
            CloseButton = tk.Button(window, text='Technique', command = lambda m = 1:open_url(kg_df['source'][2]))
            CloseButton.place(x=x+330,y=y+500)
            
            
            x,y=pos[kg_df['source'][3]]
            plt.text(x,y+0.2,s='Problem', weight='bold', fontsize=14, color="green", bbox=dict(facecolor='white', edgecolor='white', alpha=0.3),horizontalalignment='center')
            CloseButton = tk.Button(window, text='Problem', command = lambda m = 1:open_url(kg_df['source'][3]))
            CloseButton.place(x=x+400,y=y+500)
            
            x,y=pos[kg_df['target'][0]]
            plt.text(x,y+0.22,s='TITLE ', weight='bold', color="green", fontsize=14, bbox=dict(facecolor='white', edgecolor='white', alpha=0.0),horizontalalignment='center')
            CloseButton = tk.Button(window, text='Title', command = lambda m = 1:open_url(kg_df['target'][0]))
            CloseButton.place(x=x+460,y=y+500)
         
            def open_year(title,year):
                result_df = kg_df[:1]
                result_df = result_df.drop(0)
                input_string = title
                words = input_string.split()  # Split the string into individual words
                substrings = []

                 #Generate substrings of different lengths
                for i in range(len(words)):
                    for j in range(i + 1, len(words) + 1):
                        substring = ' '.join(words[i:j])  # Join the selected words with spaces
                        substrings.append(substring)

                result = substrings
                rev = []
                for i in range(0,input_string.count(" ")):
                    rev1 = result[i*input_string.count(" "):(i+1)*input_string.count(" ")+1]
                    rev2 = rev1[::-1] 
                    rev = rev + rev2

                for ele in rev:
                    searching = ''
                    searching = searching + ele
                    G_new=nx.from_pandas_edgelist(kg_df_g[kg_df_g['edge'].str.find(searching)!=-1], "source", "target", 
                                  edge_attr=True, create_using=nx.MultiDiGraph())
                    df1 = nx.to_pandas_edgelist(G_new)
                    # Check if df1 is not empty
                    if not df1.empty:
                    # Append df2 to df1
                        result_df = pd.concat([result_df, df1])
                    else:
                        continue
                
                #plt.figure(figsize=(12,12))
                pos = nx.spring_layout(G_new, k = 0.5) # k regulates the distance between nodes
                result_df.rename(columns = {'edge':'title'}, inplace = True)
                dset = pd.merge(result_df, sliced, how='inner', on=['title'])
                dset = dset.drop_duplicates(subset="title")
                dset = dset[dset.Year == year]
                dten = dset[:10]
                dten = dten.reset_index(drop=True)
                #----------change####################################:
                # Set window size to fill screen
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
                self.root.geometry(f"{screen_width}x{screen_height}")
                canvas = tk.Canvas(self.root, width=screen_width, height=screen_height)
                canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                text = tk.Text(self.root)
                text.pack(fill="both", expand=True)
                #-------change----
        
                # Create a Canvas widget and set its size
                canvas_toplevel = tk.Toplevel(self.root)

                canvas_toplevel.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}")

                # Create a Canvas widget inside the Toplevel widget and set its size
                canvas = tk.Canvas(canvas_toplevel)
                canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                # Create a Scrollbar widget and associate it with the Canvas
                scrollbar = tk.Scrollbar(canvas , orient="vertical", command=canvas.yview)
                canvas.configure(yscrollcommand=scrollbar.set)

                # Add widgets to the Canvas
                frame = tk.Frame(canvas)
                result_label = tk.Label(frame,text=f'Search Results', font=("AkayaKanadaka",40))
                result_label.pack(padx=20, pady=20, anchor="center")
        
                for i, row in dten.iterrows():
                    k=30*i
                    item = "this is item {}".format(i+1)
                    text.insert("end", item + "\t\n")
                    text.insert("end", item + "\t\n")                   
                    button = tk.Button(canvas, text=f"Visualise {i+1}", padx=2, pady=2,cursor="left_ptr",
                               bd=1, highlightthickness=0,
                                   command = lambda m=dten['index_col'][i:i+1]: KGT(m))
              
                    button.place(x=1200+i, y=125+k)
               
                    text.insert("end", "\n")
                    #change:---
                    title = row['title']
                    abstract = row['abstract']
                    #url = row['URL']
                    title_label = tk.Label(frame, text=title, font=("Arial", 10), fg='blue', cursor='hand2')
                    title_label.pack(pady=5, padx=5, anchor='w')
                    ###########################################
            
                    #title_label.bind('<Button-1>', lambda e, u=url: webbrowser.open_new(u))
                    abstract_label = tk.Label(frame, text=abstract, font=("Arial", 9), wraplength=5000, justify='left')
                    abstract_label.pack(pady=5, padx=5, anchor='w')
                    # Pack the widgets onto the window
                scrollbar.pack(side="right", fill="y")
                canvas.pack(side="left", fill="both", expand=True)
                canvas.create_window((0,0), window=frame, anchor="nw")
        
                # Configure the Canvas to resize with the window
                frame.bind("<Configure>", lambda event, canvas=canvas: canvas.configure(scrollregion=canvas.bbox("all")))
                self.root.mainloop()
                
            
            
            
            # defination of open_url 
        
            def open_url(search):
                
                if(type(search)!=str):
                    search = str(search)
   
                #dg = nx.to_pandas_edgelist(G_new)
 
                result_df = kg_df[:1]
                
                result_df = result_df.drop(0)

                input_string = search
                words = input_string.split()  # Split the string into individual words
                substrings = []

                 #Generate substrings of different lengths
                for i in range(len(words)):
                    for j in range(i + 1, len(words) + 1):
                        substring = ' '.join(words[i:j])  # Join the selected words with spaces
                        substrings.append(substring)

                result = substrings
                rev = []
                for i in range(0,input_string.count(" ")):
                    rev1 = result[i*input_string.count(" "):(i+1)*input_string.count(" ")+1]
                    rev2 = rev1[::-1] 
                    rev = rev + rev2

                for ele in rev:
                    searching = ''
                    searching = searching + ele
                    G_new_url=nx.from_pandas_edgelist(kg_df_g[kg_df_g['edge'].str.find(searching)!=-1], "source", "target", 
                                  edge_attr=True, create_using=nx.MultiDiGraph())
                    df1 = nx.to_pandas_edgelist(G_new_url)
                    # Check if df1 is not empty
                    if not df1.empty:
                    # Append df2 to df1
                        result_df = pd.concat([result_df, df1])
                    else:
                        continue
                
                plt.figure(figsize=(12,12))
                pos = nx.spring_layout(G_new_url, k = 0.5) # k regulates the distance between nodes
                result_df.rename(columns = {'edge':'title'}, inplace = True)
                dset_t = pd.merge(result_df, sliced, how='inner', on=['title'])
                dset_t = dset_t.drop_duplicates(subset="title")
                dten_t = dset_t[:10]
                dten_t = dten_t.reset_index(drop=True)
                
                #----------change####################################:
                # Set window size to fill screen
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
                self.root.geometry(f"{screen_width}x{screen_height}")
                canvas = tk.Canvas(self.root, width=screen_width, height=screen_height)
                canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                text = tk.Text(self.root)
                text.pack(fill="both", expand=True)
                #-------change----
        
                # Create a Canvas widget and set its size
                canvas_toplevel = tk.Toplevel(self.root)

                canvas_toplevel.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}")

                # Create a Canvas widget inside the Toplevel widget and set its size
                canvas = tk.Canvas(canvas_toplevel)
                canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                # Create a Scrollbar widget and associate it with the Canvas
                scrollbar = tk.Scrollbar(canvas , orient="vertical", command=canvas.yview)
                canvas.configure(yscrollcommand=scrollbar.set)

                # Add widgets to the Canvas
                frame = tk.Frame(canvas)
                result_label = tk.Label(frame,text=f'Search Results', font=("AkayaKanadaka",40))
                result_label.pack(padx=20, pady=20, anchor="center")
        
                for i, row in dten_t.iterrows():
                    k=30*i
                    item = "this is item {}".format(i+1)
                    text.insert("end", item + "\t\n")
                    text.insert("end", item + "\t\n")                   
                    button = tk.Button(canvas, text=f"Visualise {i+1}", padx=2, pady=2,cursor="left_ptr", bd=1, highlightthickness=0,
                                   command = lambda m=dten_t['index_col'][i:i+1]: KGT(m))
              
                    button.place(x=1200+i, y=125+k)
               
                    text.insert("end", "\n")
                    #change:---
                    title = row['title']
                    abstract = row['abstract']
                    #url = row['URL']
                    title_label = tk.Label(frame, text=title, font=("Arial", 10), fg='blue', cursor='hand2')
                    title_label.pack(pady=5, padx=5, anchor='w')
                    ###########################################
            
                    #title_label.bind('<Button-1>', lambda e, u=url: webbrowser.open_new(u))
                    abstract_label = tk.Label(frame, text=abstract, font=("Arial", 9), wraplength=1000, justify='left')
                    abstract_label.pack(pady=5, padx=5, anchor='w')
                    # Pack the widgets onto the window
                scrollbar.pack(side="right", fill="y")
                canvas.pack(side="left", fill="both", expand=True)
                canvas.create_window((0,0), window=frame, anchor="nw")
        
                # Configure the Canvas to resize with the window
                frame.bind("<Configure>", lambda event, canvas=canvas: canvas.configure(scrollregion=canvas.bbox("all")))
                self.root.mainloop()
    # change here----
                
            
            
            # search on dataframe using KG:
            def author_search(search):
                if(type(search)!=str):
                    search = str(search)
                 
        
                G=nx.from_pandas_edgelist(kg_df_a[kg_df_a['edge'].str.find(search)!=-1], "source", "target",edge_attr=True, create_using=nx.MultiDiGraph())
                result_df = nx.to_pandas_edgelist(G)
                #nx.draw(G, with_labels=True, node_color='green', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
                result_df.rename(columns = {'edge':'author'}, inplace = True)
                #result_df = result_df.drop_duplicates(subset="title")
                d_a = pd.merge(result_df, sliced, how='inner', on=['author'])
                d_a = d_a.drop_duplicates(subset="title")
                dten_a = d_a[:10]

                #----------change####################################:
                # Set window size to fill screen
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
                self.root.geometry(f"{screen_width}x{screen_height}")
                canvas = tk.Canvas(self.root, width=screen_width, height=screen_height)
                canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                text = tk.Text(self.root)
                text.pack(fill="both", expand=True)
                #-------change----
        
                # Create a Canvas widget and set its size
                canvas_toplevel = tk.Toplevel(self.root)

                canvas_toplevel.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}")

                # Create a Canvas widget inside the Toplevel widget and set its size
                canvas = tk.Canvas(canvas_toplevel)
                canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                # Create a Scrollbar widget and associate it with the Canvas
                scrollbar = tk.Scrollbar(canvas , orient="vertical", command=canvas.yview)
                canvas.configure(yscrollcommand=scrollbar.set)

                # Add widgets to the Canvas
                frame = tk.Frame(canvas)
                result_label = tk.Label(frame,text=f'Search Results', font=("AkayaKanadaka",40))
                result_label.pack(padx=20, pady=20, anchor="center")
        
                for i, row in dten_a.iterrows():
                    #change:---
                    k=30*i
                    item = "this is item {}".format(i+1)
                    text.insert("end", item + "\t\n")
                    text.insert("end", item + "\t\n")                   
                    button = tk.Button(canvas, text = f"Visualise {i+1}", padx=2, pady=2,
                               cursor="left_ptr",
                               bd=1, highlightthickness=0,
                                   command = lambda m=int(dten_a['index_col'][i:i+1]): KGT(m))
              
                    button.place(x=1200+i, y=125+k)
               
                    text.insert("end", "\n")
                    #change:---
                    title = row['title']
                    abstract = row['abstract']
                    #url = row['URL']
                    title_label = tk.Label(frame, text=title, font=("Arial", 10), fg='blue', cursor='hand2')
                    title_label.pack(pady=5, padx=5, anchor='w')
                    ###########################################
            
                    #title_label.bind('<Button-1>', lambda e, u=url: webbrowser.open_new(u))
                    abstract_label = tk.Label(frame, text=abstract, font=("Arial", 9), wraplength=10000, justify='left')
                    abstract_label.pack(pady=5, padx=5, anchor='w')
                    # Pack the widgets onto the window
                scrollbar.pack(side="right", fill="y")
                canvas.pack(side="left", fill="both", expand=True)
                canvas.create_window((0,0), window=frame, anchor="nw")
                
                # Configure the Canvas to resize with the window
                frame.bind("<Configure>", lambda event, canvas=canvas: canvas.configure(scrollregion=canvas.bbox("all")))
                self.root.mainloop()
    
    
    
            
            cso_df = dfw[m:m+1]
            #cso_df.rename(columns = {'ABSTRACT':'abstract','PAPER TITLE':'title'}, inplace = True)
            #d = cso_df[['abstract','title','union']]
            #d.to_csv('d.csv')
            
            candidate_sentences = cso_df
            candidate_sentences['title'][0] = candidate_sentences['title'][0:1].replace("\n","")
            candidate_sentences['author'][0] = candidate_sentences['author'][0:1].replace("and", ",")
            
    
            articles = ['A ', 'An ', 'The ', 'a ', 'an ', 'the ',' study of ', 'On ']
            problem = [' based on ', ' on the Size of ', ' by means of ', ' for ', ' with ',' in ',' of ', ' over ', ' on ', ]
    
            text = candidate_sentences['title'][0:1].str.casefold()
            text = text.to_string()
            text = text[5:]
            #text = text.replace("\n ", "")
            for i in articles:
                if i in text:
                    text = text.replace(i, ' ')     
            
            for i in problem:
                if i in text:
                    text = text.replace(i, ',',1)
                    break 
            
            Data = text
            string = text
           
    
            # x = pd.Series()  
            df_s = pd.Series({'source' : []})
            df_e = pd.Series({'edge' : 1})
            df_t = pd.Series({'target': []})
            dflink = pd.DataFrame()
    
            
            #df_s = df_s.append(candidate_sentences['union'][i:i+1], ignore_index=True)
            df_s = df_s.append(candidate_sentences['Year'][0:1],ignore_index=True)
            df_s = df_s.append(candidate_sentences['author'][0:1].str.split(",", n = 1, expand = True)[0], ignore_index=True)
            
            Data = Data[5:]
            flag=0
            title_prob = list(Data.split(",")[0].split(" "))

            
            a = text.split(",",1)[0]
            a = pd.Series(a)
            b = text.split(",",1)[1]
            b = pd.Series(b)
            df_s = df_s.append(a,ignore_index=True)
            df_s = df_s.append(b,ignore_index=True)
    
            df_s = df_s.drop(0)
            df_s.index = np.arange(0,len(df_s))
            #x = candidate_sentences['title'][0].replace("\n ", "")

            df_e = pd.concat([df_e]*4, ignore_index=True)
    
    
            #x = string
            x = candidate_sentences['title'][0:1]
            x = pd.Series(x)
            df_t = pd.concat([x]*4, ignore_index=True)
    
            string = df_s[2]
            if len(string)>10:
                pos = string.index(' ') 
                string = string[:pos] + ' \n ' + string[pos + 1:] 
                df_s[2] = string
    
            source = [i for i in df_s]
            relations = [i for i in df_e]
            target = [i for i in df_t]
    
            kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})
    
            # create a directed-graph from a dataframe
            G=nx.from_pandas_edgelist(kg_df, "source", "target",edge_attr=True, create_using=nx.MultiDiGraph())
    
            # Let’s plot the network:

            f = plt.figure(figsize=(14,5), dpi=100)
            pos = nx.spring_layout(G)
            d = dict(G.degree)

            center_node =  1 
            edge_nodes = set(G) - {center_node}

            pos = nx.circular_layout(G.subgraph(edge_nodes))
            pos[center_node] = np.array([0, 0])

            # Ensures the nodes around the circle are evenly distributed
            nx.draw(G, pos=pos, with_labels=True, node_shape="s", node_color="none", bbox=dict(facecolor="white", edgecolor='black', boxstyle='round,pad=0.8'))

            #Set the geometry
            window.geometry("950x500")

            window.eval('tk::PlaceWindow . center')
            # create matplotlib canvas using figure `f` and assign to widget `window`
            canvas = FigureCanvasTkAgg(f, window)
            
            # get canvas as tkinter's widget and `grid` in widget `window`
            canvas.get_tk_widget().grid(row=1, column=5, padx=25, pady=40)
            
            
            x,y=pos[kg_df['source'][0]]
            plt.text(x,y+0.14,s= 'Year', weight='bold', fontsize=14, color="green", bbox=dict(facecolor='white', edgecolor='white', alpha=0.0),horizontalalignment='center')
            CloseButton = tk.Button(window, text='Year', command = lambda m = 1:open_year(kg_df['target'][0],kg_df['source'][0]))
            CloseButton.place(x=x+240,y=y+500)
            
            x,y=pos[kg_df['source'][1]]
            plt.text(x,y+0.2,s= 'First Author', weight='bold', fontsize=14, color="green", bbox=dict(facecolor='white', edgecolor='white', alpha=0.0),horizontalalignment='center')
            CloseButton = tk.Button(window, text='Author', command = lambda m = 1: author_search(kg_df['source'][1]))
            CloseButton.place(x=x+280,y=y+500)
            
            x,y=pos[kg_df['source'][2]]
            plt.text(x,y+0.25,s='Technique', weight='bold', color="green", fontsize=14, bbox=dict(facecolor='white', edgecolor='white', alpha=0.0),horizontalalignment='center')
            CloseButton = tk.Button(window, text='Technique', command = lambda m = 1:open_url(kg_df['source'][2]))
            CloseButton.place(x=x+330,y=y+500)
            
            
            x,y=pos[kg_df['source'][3]]
            plt.text(x,y+0.2,s='Problem', weight='bold', fontsize=14, color="green", bbox=dict(facecolor='white', edgecolor='white', alpha=0.3),horizontalalignment='center')
            CloseButton = tk.Button(window, text='Problem', command = lambda m = 1:open_url(kg_df['source'][3]))
            CloseButton.place(x=x+400,y=y+500)
            
            x,y=pos[kg_df['target'][0]]
            plt.text(x,y+0.22,s='TITLE ', weight='bold', color="green", fontsize=14, bbox=dict(facecolor='white', edgecolor='white', alpha=0.0),horizontalalignment='center')
            CloseButton = tk.Button(window, text='Title', command = lambda m = 1:open_url(kg_df['target'][0]))
            CloseButton.place(x=x+460,y=y+500)
            
             
            
             
            
            #----------change####################################:
            # Set window size to fill screen
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            self.root.geometry(f"{screen_width}x{screen_height}")
            canvas = tk.Canvas(self.root, width=screen_width, height=screen_height)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            text = tk.Text(self.root)
            text.pack(fill="both", expand=True)
            #-------change----
        
            # Create a Canvas widget and set its size
            canvas_toplevel = tk.Toplevel(self.root)

            canvas_toplevel.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}")

            # Create a Canvas widget inside the Toplevel widget and set its size
            canvas = tk.Canvas(canvas_toplevel)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            # Create a Scrollbar widget and associate it with the Canvas
            scrollbar = tk.Scrollbar(canvas , orient="vertical", command=canvas.yview)
            canvas.configure(yscrollcommand=scrollbar.set)

            # Add widgets to the Canvas
            frame = tk.Frame(canvas)
            result_label = tk.Label(frame,text=f'Search Results', font=("AkayaKanadaka",40))
            result_label.pack(padx=20, pady=20, anchor="center")
            self.root.mainloop()
            
# change line 535  
        candidate_sentences = sliced
        df_s = pd.Series({'source' : []})
        df_e = pd.Series({'edge' : []})
        df_t = pd.Series({})
        df_se = pd.DataFrame()
        for i in range(0,len(candidate_sentences)): 
          # For Author
            e = pd.DataFrame()
            new = candidate_sentences['author'][i:i+1].str.split(",", n = 20, expand = True)
            sr = new.squeeze()
            if type(sr)==str:
                sr = pd.Series(sr)
            df_e = df_e.append(sr, ignore_index=True)
  
          # For Year
            y = pd.DataFrame()
            y = pd.concat([candidate_sentences['Year'][i:i+1]]*(sr.count()), ignore_index=True)
            df_s = df_s.append(y, ignore_index=True)
  
          # For Title
            t = pd.DataFrame() 
            t =  pd.concat([candidate_sentences['title'][i:i+1]]*(sr.count()), ignore_index=True)
            df_t = df_t.append(t, ignore_index=True)
        df_e = df_e.drop(0)
        df_e = df_e.reset_index()
        df_e = df_e.drop('index', axis=1)
        df_e = df_e.squeeze()

        df_s = df_s.drop(0)
        df_s = df_s.reset_index()
        df_s = df_s.drop('index', axis=1)
        df_s = df_s.squeeze()

        for j in range(0, len(df_t)):
            if type(df_t[j]) == float:
                df_t[j] = str(df_t[j])
            if df_t[j].startswith(' ') == True:
                df_t[j] = df_t[j].replace(' ', '', 1)
            else:
                continue
        source = [i for i in tqdm(df_s)]
        relations = [i for i in tqdm(df_e)] # authors
        target = [i for i in tqdm(df_t)]    # titles
        kg_df_a = pd.DataFrame({'source':source, 'target':target, 'edge':relations})

        # create a directed-graph from a dataframe
        G=nx.from_pandas_edgelist(kg_df_a, "source", "target",edge_attr=True, create_using=nx.MultiDiGraph())




# author kg ends===        
        # =============== main KG ====================#
        candidate_sentences = sliced
        df_s = pd.Series({'source' : []})
        df_e = pd.Series({'edge' : []})
        df_t = pd.Series({})
        df_se = pd.DataFrame()
        for i in range(0,len(candidate_sentences)): 
          # For Author
          e = pd.DataFrame()
          new = candidate_sentences['author'][i:i+1].str.split(",", n = 20, expand = True)
          sr = new.squeeze()
          if type(sr)==str:
            sr = pd.Series(sr)
          df_e = df_e.append(sr, ignore_index=True)
  
          # For Year
          y = pd.DataFrame()
          y = pd.concat([candidate_sentences['Year'][i:i+1]]*(sr.count()), ignore_index=True)
          df_s = df_s.append(y, ignore_index=True)
  
          # For Title
          t = pd.DataFrame() 
          t =  pd.concat([candidate_sentences['title'][i:i+1]]*(sr.count()), ignore_index=True)
          df_t = df_t.append(t, ignore_index=True)
        df_e = df_e.drop(0)
        df_e = df_e.reset_index()
        df_e = df_e.drop('index', axis=1)
        df_e = df_e.squeeze()

        df_s = df_s.drop(0)
        df_s = df_s.reset_index()
        df_s = df_s.drop('index', axis=1)
        df_s = df_s.squeeze()

        for j in range(0, len(df_t)):
            if type(df_t[j]) == float:
                df_t[j] = str(df_t[j])
            if df_t[j].startswith(' ') == True:
                df_t[j] = df_t[j].replace(' ', '', 1)
            else:
                continue
        source = [i for i in tqdm(df_s)]
        relations = [i for i in tqdm(df_t)]
        target = [i for i in tqdm(df_e)]
        kg_df_g = pd.DataFrame({'source':source, 'target':target, 'edge':relations})

        # create a directed-graph from a dataframe
        G_kg=nx.from_pandas_edgelist(kg_df_g, "source", "target",edge_attr=True, create_using=nx.MultiDiGraph())
        
        result_df = kg_df_g[:1]
        result_df = result_df.drop(0)

        input_string = keyword
        words = input_string.split()  # Split the string into individual words
        substrings = []

         #Generate substrings of different lengths
        for i in range(len(words)):
            for j in range(i + 1, len(words) + 1):
                substring = ' '.join(words[i:j])  # Join the selected words with spaces
                substrings.append(substring)

        result = substrings
        rev = []
        for i in range(0,input_string.count(" ")):
            rev1 = result[i*input_string.count(" "):(i+1)*input_string.count(" ")+1]
            rev2 = rev1[::-1] 
            rev = rev + rev2

        for ele in rev:
            search = ''
            search = search + ele
            G_kg=nx.from_pandas_edgelist(kg_df_g[kg_df_g['edge'].str.find(search)!=-1], "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())
            df1 = nx.to_pandas_edgelist(G_kg)
            # Check if df1 is not empty
            if not df1.empty:
            # Append df2 to df1
                result_df = pd.concat([result_df, df1])
            else:
                continue

        result_df.rename(columns = {'edge':'title'}, inplace = True)
        dset_m = pd.merge( result_df, sliced, how='inner', on=['title'])
        dset_m = dset_m.drop_duplicates(subset="title")
        
        
                
        
        #----------change####################################:
         # Set window size to fill screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")
        canvas = tk.Canvas(self.root, width=screen_width, height=screen_height)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text = tk.Text(self.root)
        text.pack(fill="both", expand=True)
        
        
        # Create a Canvas widget and set its size
        canvas_toplevel = tk.Toplevel(self.root)

        canvas_toplevel.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}")

        # Create a Canvas widget inside the Toplevel widget and set its size
        canvas = tk.Canvas(canvas_toplevel)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Create a Scrollbar widget and associate it with the Canvas
        scrollbar = tk.Scrollbar(canvas , orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        # Add widgets to the Canvas
        frame = tk.Frame(canvas)
        result_label = tk.Label(frame,text=f'Search Results', font=("AkayaKanadaka",20))
        result_label.pack(padx=20, pady=20, anchor="center")
        dset_10 = dset_m[:10]
        dset_10 = dset_10.reset_index(drop=True)
        for i, row in dset_10.iterrows():
            #change:---
            k=30*i
            item = "this is item {}".format(i+1)
            text.insert("end", item + "\t\n")
            text.insert("end", item + "\t\n")                   
            button = tk.Button(canvas, text=f"Visualise {i+1}", padx=2, pady=2,
                       cursor="left_ptr",
                       bd=1, highlightthickness=0,
                       command = lambda m=dset_10['index_col'][i:i+1]: KGT(m))
              
            button.place(x=1200+i, y=125+k)
               
            text.insert("end", "\n")
            #change:---
            title = row['title']
            abstract = row['abstract']
            #url = row['URL']
            title_label = tk.Label(frame, text=title, font=("Arial", 10), fg='blue', cursor='hand2')
            title_label.pack(padx=5, pady=5, anchor='w')
            ###########################################
            
            #title_label.bind('<Button-1>', lambda e, u=url: webbrowser.open_new(u))
            abstract_label = tk.Label(frame, text=abstract, font=("Arial", 9), wraplength=10000, justify='left')
            abstract_label.pack(padx=5, pady=5, anchor='w')
        # Pack the widgets onto the window
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.create_window((0,0), window=frame, anchor="nw")
        
        # Configure the Canvas to resize with the window
        frame.bind("<Configure>", lambda event, canvas=canvas: canvas.configure(scrollregion=canvas.bbox("all")))
        
    
        self.root.mainloop()
        
 # change line 689   
   

    def submit65(self):
        self.root = tk.Tk()
        self.root.title('Extractive Summary')
        # Set window size to fill screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")
        keyword = self.keyword_entry.get()
        
        if os.path.isfile(keyword):
            content = extract_sections_from_pdf(keyword)
            paragraph = ' '.join(content.values())
            paragraph = remove_duplicate_sentences(paragraph)
            text1 = generate_summary(paragraph,10)
            text3 = generate_summary(paragraph,20)
            text2=text1+"\n\n\n" + "Your summaries is available in summary.txt"
            abstract_label = tk.Label(self.root, text=text2, font=("Arial", 12), wraplength=1000, justify='left')
            abstract_label.pack(pady=5, padx=5, anchor='w')
            text2 = abstractive_summary(text3)
            with open('summary.txt', 'w', encoding="utf-8") as f:
                # Write the Extrative_Summary heading and content
                f.write("Extractive Summary:\n\n")
                f.write(text1 + "\n\n")
                # Write the Abstractive_Summary heading and content
                f.write("Abstractive Summary:\n\n")
                f.write(text2 + "\n\n")          
        else:
            messagebox.showerror("Invalid PDF Name", "no such pdf file available in the system")
            
            
        
    def __init__(self):
        
        self.root = tk.Tk()
        
        # Front page:
        from PIL import Image, ImageTk
        
        self.root.title('Summarization and Scholar Search Engine')

        # Set window size to fill screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")

        # Create a canvas to put the scrollable frame in
        canvas = tk.Canvas(self.root, width=screen_width, height=screen_height)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create a scrollable frame to put the UI elements in
        scrollable_frame = tk.Frame(canvas)

        # Add a scrollbar to the canvas to scroll the frame
        scrollbar = tk.Scrollbar(self.root, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=scrollbar.set)

        # Bind the canvas to the scrollbar
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')

        # Add UI elements to the scrollable frame
           # UI_label_font = ("AkayaKanadaka", 40)
        img = Image.open(r"pic_sum.png")
        photo = ImageTk.PhotoImage(img, master=scrollable_frame)
        
        # here, image option is used to
        # set image on button
       
        self.photo_Button=tk.Button(scrollable_frame, image = photo)

        #self.UI_label = tk.Label(scrollable_frame, text='Summarization', font=UI_label_font)
        #self.UI_label = tk.Label(scrollable_frame, image = test)
        self.photo_Button.pack(padx=20, pady=20, anchor="center" )

        label_font = ("Arial", 10)
        self.keyword_label = tk.Label(scrollable_frame, text='Enter your File Name:', font=label_font)
        self.keyword_label.pack(padx=10, pady=10, anchor="center")

        self.keyword_entry = tk.Entry(scrollable_frame, width=30, font=label_font)
        self.keyword_entry.pack(pady=10, anchor="center")

        #self.submit55_button = tk.Button(scrollable_frame, text='Verify Text', font=label_font, command=lambda: self.submit55())
        #self.submit55_button.pack(padx=10, pady=10, anchor="center")
        
        self.submit65_button = tk.Button(scrollable_frame, text='Summary', font=label_font, command=lambda: self.submit65())
        self.submit65_button.pack(padx=10, pady=10, anchor="center")
        self.keyword_label = tk.Label(scrollable_frame, text='(estimated time:2-5 min)', font=label_font)
        self.keyword_label.pack(padx=10, pady=10, anchor="center")
        
        ###################################################################################
        self.submit11_button = tk.Button(scrollable_frame, text='Scholar Search', font=label_font, command=lambda: self.submit11())
        self.submit11_button.pack(padx=10, pady=5, anchor="sw")
        
        self.num_results_label = tk.Label(scrollable_frame, text='', font=("Arial", 14))
        self.num_results_label.pack(pady=10, anchor='w')

        self.result_canvas = tk.Canvas(scrollable_frame, width=screen_width, height=screen_height, bd=0, highlightthickness=0)
        self.result_canvas.pack(pady=10, anchor="w")

        self.result_frame = tk.Frame(self.result_canvas)
        self.result_canvas.create_window((0, 0), window=self.result_frame, anchor='nw')

        # update the result canvas scroll region after adding new search results
        self.result_frame.bind('<Configure>', lambda e: self.result_canvas.configure(scrollregion=self.result_canvas.bbox('all')))        
        
        
        
        
        # Create Knowledge Graph for entire dataframe once:
        dfw = sliced
        dfw['index_col'] = dfw.index
        dfw['title'] = dfw['title'].str.casefold()
        dfw['author'] = dfw['author'].str.casefold()
        
        candidate_sentences = dfw
        
         
        df_s = pd.Series({'source' : []})
        df_e = pd.Series({'edge' : []})
        df_t = pd.Series({})
        df_se = pd.DataFrame()
        for i in range(0,len(candidate_sentences)): 
        # For Author
            e = pd.DataFrame()
            new = candidate_sentences['author'][i:i+1].str.split(",", n = 20, expand = True)
            sr = new.squeeze()
            if type(sr)==str:
                sr = pd.Series(sr)
            df_e = df_e.append(sr, ignore_index=True)
  
            # For Year
            y = pd.DataFrame()
            y = pd.concat([candidate_sentences['Year'][i:i+1]]*(sr.count()), ignore_index=True)
            df_s = df_s.append(y, ignore_index=True)
  
            # For Title
            t = pd.DataFrame() 
            t =  pd.concat([candidate_sentences['title'][i:i+1]]*(sr.count()), ignore_index=True)
            df_t = df_t.append(t, ignore_index=True)
        df_e = df_e.drop(0)
        df_e = df_e.reset_index()
        df_e = df_e.drop('index', axis=1)
        df_e = df_e.squeeze()

        df_s = df_s.drop(0)
        df_s = df_s.reset_index()
        df_s = df_s.drop('index', axis=1)
        df_s = df_s.squeeze()

        for j in range(0, len(df_t)):
            if type(df_t[j]) == float:
                df_t[j] = str(df_t[j])
            if df_t[j].startswith(' ') == True:
                df_t[j] = df_t[j].replace(' ', '', 1)
            else:
                continue
        source = [i for i in tqdm(df_s)]     # contains year.
        relations = [i for i in tqdm(df_t)]  # contains title
        target = [i for i in tqdm(df_e)]     # contains author
        
        kg_dfs = pd.DataFrame({'source':source, 'target':target, 'edge':relations})
        kg_dfs['edge'] = kg_dfs['edge'].str.casefold()
        kg_dfs['target'] = kg_dfs['target'].str.casefold()
        # create a directed-graph from a dataframe
        G_new=nx.from_pandas_edgelist(kg_dfs, "source", "target",edge_attr=True, create_using=nx.MultiDiGraph())
            
        self.root.mainloop()       



# %%
from tkinter import *
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import webbrowser
from cso_classifier import CSOClassifier as cp
app = KeywordInputApp()

# %%
