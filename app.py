# Core PKG's
import streamlit as st

# Pacotes Adicionais/ Summarization PKG's
# TextRank algotihtm
from gensim.summarization import summarize

# LexRank algorithm
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from langdetect import detect

import nltk
# nltk.data.find('tokenizers/punkt_tab')
nltk.download('punkt_tab')
from nltk import word_tokenize,sent_tokenize



# Exploratory Data Analysis PKG's
import pandas as pd

# Data visualization PKG's
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Configurando página streamlit
st.set_page_config(page_title="Summarize App", page_icon=":shark:", layout="wide")

## CARALHOOOO ESSA BUDEGA DEU CERTOOOO PEGAAAAAAAA !!!!!!!!!!
## Função para o Sumy Summarization/ LexRank
def sumy_summarizer(docx, num=2):
    language = detect(docx)
    if language == 'en':
        tokenizer = Tokenizer("english")
    elif language == 'pt':
        tokenizer = Tokenizer("portuguese")
    else:
        raise ValueError("Idioma não suportado: " + language)
    
    parser = PlaintextParser.from_string(docx, tokenizer)
    lex_summarizer = LexRankSummarizer()
    resumo = lex_summarizer(parser.document, num)
    lista_resumo = [str(frase) for frase in resumo]
    result = ' '.join(lista_resumo)
    
    return result


def main(): 
    """ Summarization simples com Streamlit """
    
    st.title("Summarize app")
    menu = ["Principal", "Sobre"]
    escolha = st.sidebar.selectbox("Menu", menu)
    
    if escolha == "Principal":
        st.subheader("Summarization")
        texto_puro = st.text_area("Cole o texto aqui")
        if st.button("Summarize"):
            
            with st.expander("Texto Original"):
                st.write(texto_puro)
        
            # Meu Layout
            c1, c2 = st.columns(2)
            with c1:
                with st.expander("LexRank Summary"):
                   meu_resumo = sumy_summarizer(texto_puro)
                   st.write(meu_resumo)
            
            with c2:
                with st.expander("TextRank Summary"):
                    meu_resumo = summarize(texto_puro)
                    st.write(meu_resumo)
    else:
        st.subheader("Sobre")

    
    
if __name__ == '__main__':
    main()