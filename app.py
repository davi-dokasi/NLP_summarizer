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

# Validação Summarização
from rouge import Rouge

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



def valida_summary(summary, reference):
    r = Rouge()
    eval_score = r.get_scores(summary, reference)
    eval_score_df = pd.DataFrame(eval_score[0])
    return eval_score_df

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
                   document_len = {'Original':len(texto_puro), 
                                   'Summary':len(meu_resumo)}
                   st.write(document_len)
                   st.write(meu_resumo)
                   
                   # Rouge (Recall-Oriented Understudy for Gisting Evaluation)
                   # Rouge Score é uma métrica de avaliação de sumarização
                   # O score é calculado comparando o resumo gerado com o resumo de referência
                   # Rouge-n -> Rouge-1 mede a similaridade (sobreposição) entre palavras unigramas
                   # Rouge-2 mede a similaridade entre bigramas
                   # Rouge-L mede a similaridade entre sequências de palavras
                   
                   st.info("Rouge Score")
                   score = valida_summary(meu_resumo, texto_puro)
                   
                   # explicação de (r, p, f)
                   # r (recall) -> proporção de palavras no resumo de referência que estão no resumo gerado (em resumo quando do conteúdo relevante foi capturado pelo resumo gerado)
                   # p (precision) -> proporção de palavras no resumo gerado que estão no resumo de referência (em resumo, quanto do conteúdo gerado é realmente relevante)
                   # f (F-score) -> média harmônica entre r e p. Útil para quando se quer uma métrica única consideranto tanto precisão quanto recall
                   
                   
                   # Interpretação (r =0.5, p = 0.6, f = 0.55)
                   # Rouge-n(1 é unigrama, 2 é bigrama) 
                   # r = 50% dos unigramos nos resumos de referência estão presentes no resumo gerado
                   # p = 60% dos unigramos no resumo gerado estão no de referência
                   # f = 55% média harmônica entre r e p
                   
                   # Rouge-l
                   # 50% da sequência mais longa de palavras no resumo de referência estão presentes no resumo gerado
                   # 60% da sequência mais longa de palavras no resumo gerado estão no resumo de referência
                   # f = 55% média harmônica entre r e p
                   
                   st.dataframe(score.T)
                   score['metrics'] = score.index
                   fig = px.bar(score, x='metrics', 
                                y='rouge-1', title='Rouge-1 Score')
                   st.plotly_chart(fig)
                   
            with c2:
                with st.expander("TextRank Summary"):
                    meu_resumo = summarize(texto_puro)
                    document_len = {'Original':len(texto_puro), 
                                    'Summary':len(meu_resumo)}
                    st.write(document_len)
                    st.write(meu_resumo)
                    
                    st.info("Rouge Score")
                    score = valida_summary(meu_resumo, texto_puro)
                    st.dataframe(score.T)
                    
                    score['metrics'] = score.index
                    fig = px.bar(score, x='metrics', 
                                y='rouge-1', title='Rouge-1 Score')
                    st.plotly_chart(fig)
    
    
    else:
        st.subheader("Sobre")

    
    
if __name__ == '__main__':
    main()