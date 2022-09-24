from typing import List, Set
import streamlit as st
import sympy as sp

from src.dataset import Equation
from src.dashboard.model_handler import get_model

def render_algebraic_page():
    """Render the algebraic representation page"""
    
    st.write("### Algebraic form of representation")

    model = st.radio(
        "Choose a model:",
        (
            "AI-Feynman",
            "Deep Symbolic Regression",
            "Genetic Programming",
            "Neural Guided Genetic Programming",
            "Neural Symbolic Regression",
        )
    )

    modelToName = {
        "AI-Feynman":"aifeynman",
        "Deep Symbolic Regression":"dsr",
        "Genetic Programming":"gplearn",
        "Neural Guided Genetic Programming":"dsr-gp",
        "Neural Symbolic Regression":"nesymres",
    }

    func = st.text_input(
        "Please enter a function, with variables ranging from x0 to x3", 
        "2*sin(x0) + 5*x1",         
        key="input_eq"
    )

    def generate_support_dict(keys: Set[str]) -> dict:
        support = {}
        for k in keys:
            support[k] = {
                "max": 5,
                "min": -5,
            }
            
        return support

    number_of_pts = st.slider(
        "Pick the number of support points", 
        min_value=10, 
        max_value=500, 
        value=50, 
        step=10
    )

    noise = st.slider(
        "Pick the noise level", 
        min_value=0.0, 
        max_value=0.1, 
        value=0.0, 
        step=0.01
    )

    try:
        expr = sp.parse_expr(func)
        support = generate_support_dict(expr.free_symbols)

        eq = Equation(
            expr=expr, 
            variables=expr.free_symbols,
            support=support, 
            number_of_points=number_of_pts
            )

    except Exception as e:
        st.error(f"Please enter a valid sympy function: {str(e)}")
        return     
    
    x,y = eq.get_data(noise)
    st.write("The following equation is provided to generate the dataset:")
    st.latex(sp.latex(eq.expr))

    if st.button("Train model"):
        
        if model == "AI-Feynman":
            model = get_model(modelToName[model])
        elif model == "Neural Guided Genetic Programming":
            model = get_model(modelToName[model])
        elif model == "Neural Symbolic Regression":
            model = get_model(modelToName[model])
        elif model == "Deep Symbolic Regression":
            model = get_model(modelToName[model])
        elif model == "Genetic Programming":
            model = get_model(modelToName[model])

        with st.spinner("Fitting the data..."):
            model.fit(x,y)
        
        st.write("The following equation is predicted by the model:")
        st.latex(sp.latex(model.equation()))
        st.write(f"Accuracy of model: {str(model.score(x,y))}")
    





def render_differential_page():
    """Render the differential representation page"""

    st.write("### Differential form of representation")
