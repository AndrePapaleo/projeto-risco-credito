from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import traceback # Para ver o erro real no terminal se acontecer

app = Flask(__name__)

# --- 1. CARREGAMENTO E TREINAMENTO DA IA (Ao iniciar) ---
print("Carregando modelo e dados na memória...")

modelo = None
encoders = {}
colunas_texto = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

try:
    # Carregar dados
    df = pd.read_csv('credit_risk_dataset.csv')
    
    # Limpeza básica
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

    # Tratamento de textos (Label Encoding)
    for col in colunas_texto:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Separar dados
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    # Treinar o modelo
    modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
    modelo.fit(X, y)
    print("Modelo treinado com sucesso!")

except Exception as e:
    print(f"ERRO CRÍTICO AO TREINAR: {e}")

# --- 2. AS ROTAS DO SITE ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not modelo:
        return jsonify({'erro': 'O modelo de IA não foi carregado corretamente.'})

    dados = request.json
    
    try:
        # --- CORREÇÃO: Converter Texto para Números ---
        idade = int(dados['idade'])
        renda = float(dados['renda'])
        moradia = dados['moradia']
        tempo_emprego = float(dados['tempo_emprego'])
        motivo = dados['motivo']
        valor = float(dados['valor'])
        juros =
