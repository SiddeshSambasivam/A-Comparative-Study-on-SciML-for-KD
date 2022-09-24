import streamlit as st

import pathlib
import sys

sys.path.append(str(pathlib.Path().absolute()).split("/src")[0])
from src.dashboard.pages import render_algebraic_page, render_differential_page

def main():    

    st.write("## A Comparative Study of Machine Learning Algorithms for Knowledge Discovery")
    st.write("Given a dataset $(x,y)$, $x$ is a set of input variables, and $y$ is a set of output variables, the aim of symbolic regression is to identify a function $f : x \Rightarrow y$ that **best fits the dataset.**")
    st.sidebar.write('## Scientific Machine Learning for Knowledge Discovery')
    st.sidebar.write(
        "This web app is a demo for the final year thesis, _[Scientific Machine Learning for Knowledge Discovery]()_ by **Siddesh Sambasivam Suseela**."
    )

    alg_fam = st.sidebar.radio(
        "Select the type of knowledge discovery algorithms",
        ("Algebraic representation", "Differential representation"),
    )

    if alg_fam == "Algebraic representation":        
        render_algebraic_page()
    elif alg_fam == "Differential representation":        
        render_differential_page()


if __name__ == "__main__":    
    st.set_page_config(
        page_title="Symbolic Regression", 
        page_icon=":chart_with_upwards_trend:",         
    )
    main()