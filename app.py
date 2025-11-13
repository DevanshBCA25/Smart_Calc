import os
import ast
import operator as op
import streamlit as st
import sympy as sp
import math
import numpy as np
from streamlit_drawable_canvas import st_canvas

try:
    import openai
except Exception:
    openai = "OPEN_API_KEY"

st.set_page_config(page_title="Math Assistant", page_icon="🧮", layout="wide")

st.title("SmartCalc")
st.caption("Compute, visualize, and reason through mathematical problems with AI-powered explanations.")

st.sidebar.header("⚙️ Select Mode")
mode = st.sidebar.selectbox(
    "Choose Calculation Mode",
    ["Basic Expression", "Symbolic Math", "Matrix Operations", "Determinant", "Inverse", "Eigenvalues"]
)
precision = st.sidebar.slider("Result Precision (decimals)", 0, 12, 2)
ask_llm = st.sidebar.checkbox("💬 Ask AI for Step-by-Step Explanation")
st.sidebar.markdown("---")
st.sidebar.info("Use the whiteboard below to sketch matrices, formulas, or rough work.")

st.subheader("✏️ Interactive Math Whiteboard")
st.write("You can use this whiteboard to write matrices, equations, or visualize math problems.")
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=2,
    stroke_color="#000000",
    background_color="#ffffff",
    height=300,
    width=800,
    drawing_mode="freedraw",
    key="canvas",
)
st.markdown("---")

SAFE_OPERATORS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod,
    ast.USub: op.neg, ast.UAdd: op.pos,
}
SAFE_FUNCTIONS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "sqrt": math.sqrt, "log": math.log, "abs": abs,
}

def safe_eval(node):
    """Safe numeric evaluation with AST."""
    if isinstance(node, ast.Expression): return safe_eval(node.body)
    if isinstance(node, ast.Constant): return node.value
    if isinstance(node, ast.BinOp):
        left, right = safe_eval(node.left), safe_eval(node.right)
        return SAFE_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp):
        return SAFE_OPERATORS[type(node.op)](safe_eval(node.operand))
    if isinstance(node, ast.Call):
        func_name = getattr(node.func, "id", None)
        if func_name in SAFE_FUNCTIONS:
            args = [safe_eval(arg) for arg in node.args]
            return SAFE_FUNCTIONS[func_name](*args)
        raise ValueError(f"Function {func_name} not allowed.")
    raise ValueError("Unsupported expression.")

def evaluate_expression(expr: str):
    return safe_eval(ast.parse(expr, mode="eval"))

if mode in ["Basic Expression", "Symbolic Math"]:
    expr = st.text_input("🔢 Enter your expression:", "(2+3)*4 - 5/2")

elif mode in ["Matrix Operations", "Determinant", "Inverse", "Eigenvalues"]:
    expr = st.text_area(
        "🧮 Enter matrix elements (comma separated, semicolon for new row):",
        "1,2,3;4,5,6;7,8,9")

if st.button("Compute"):
    if mode == "Basic Expression":
        try:
            result = evaluate_expression(expr)
            display = round(result, precision) if isinstance(result, float) else result
            st.success(f"✅ Result: {display}")
        except Exception as e:
            st.error(f"Invalid expression: {e}")
    elif mode == "Symbolic Math":
        try:
            x, y, z = sp.symbols('x y z')
            symbolic_result = sp.simplify(expr)
            st.success("🧠 Simplified Expression:")
            st.latex(sp.latex(symbolic_result))
        except Exception as e:
            st.error(f"Symbolic error: {e}")
    else:
        try:
            rows = [list(map(float, r.split(","))) for r in expr.strip().split(";")]
            M = sp.Matrix(rows)
            st.write("### 📊 Input Matrix:")
            st.latex(sp.latex(M))
            if mode == "Matrix Operations":
                st.write("**Transpose:**")
                st.latex(sp.latex(M.T))
                st.write("**Rank:**", M.rank())
                st.write("**Determinant:**", M.det())
            elif mode == "Determinant":
                st.success(f"Determinant = {M.det()}")
            elif mode == "Inverse":
                if M.det() != 0:
                    st.write("**Inverse Matrix:**")
                    st.latex(sp.latex(M.inv()))
                else:
                    st.error("Matrix is singular, no inverse exists.")
            elif mode == "Eigenvalues":
                eig_vals = M.eigenvals()
                st.write("**Eigenvalues:**")
                st.latex(sp.latex(eig_vals))
        except Exception as e:
            st.error(f"Matrix error: {e}")

    if ask_llm:
        api_key = os.getenv("OPEN_API_KEY")
        if not api_key:
            st.warning("⚠️ Please set your OPENAI_API_KEY to use LLM explanations.")
        elif openai is None:
            st.warning("Install openai library to enable this feature.")
        else:
            prompt = (
                f"Explain step-by-step how to solve this mathematical problem:\n{expr}\n"
                f"Mode: {mode}. Include reasoning and results clearly."
            )
            try:
                openai.api_key = api_key
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a math tutor and reasoning assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                explanation = response["choices"][0]["message"]["content"].strip()
                st.markdown("### 💬 AI Explanation:")
                st.write(explanation)
            except Exception as e:
                st.error(f"LLM Error: {e}")

st.markdown("---")
st.caption("SmartCalc LLM — AI-powered math reasoning tool built with Streamlit and OpenAI.")


