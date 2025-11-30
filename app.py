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
        juros = float(dados['juros'])
        score = dados['score']
        calote = dados['calote']
        tempo_historico = int(dados['tempo_historico'])
        
        # Calcular percentual da renda
        percentual_renda = valor / renda if renda > 0 else 0
        
        # Criar tabela com o novo cliente
        novo_cliente = pd.DataFrame([{
            'person_age': idade,
            'person_income': renda,
            'person_home_ownership': moradia, 
            'person_emp_length': tempo_emprego,
            'loan_intent': motivo,
            'loan_grade': score, 
            'loan_amnt': valor,
            'loan_int_rate': juros,
            'loan_status': 0, # Coluna dummy apenas para manter estrutura, será removida/ignorada se necessário, mas aqui a ordem importa
            'loan_percent_income': percentual_renda,
            'cb_person_default_on_file': calote, 
            'cb_person_cred_hist_length': tempo_historico
        }])
        
        # Remover a coluna loan_status que não existe na entrada do modelo, 
        # mas precisamos garantir que as colunas estejam na mesma ordem do treinamento X
        # A ordem do CSV é: age, income, home, emp_length, intent, grade, amnt, int_rate, status, percent, default, hist
        # O X não tem status. Vamos reorganizar:
        
        novo_cliente = novo_cliente[[
            'person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
            'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate',
            'loan_percent_income', 'cb_person_default_on_file', 'cb_person_cred_hist_length'
        ]]

        # Transformar textos em códigos usando os encoders salvos
        for col in colunas_texto:
            # Tratamento de erro caso venha uma categoria que não existia no treino
            try:
                novo_cliente[col] = encoders[col].transform(novo_cliente[col])
            except:
                # Se der erro (ex: valor desconhecido), usa o valor mais comum (transforma no primeiro da lista)
                novo_cliente[col] = 0

        # Previsão
        probabilidade = modelo.predict_proba(novo_cliente)[0][1]
        
        # Regra de Negócio
        decisao = "CRÉDITO APROVADO" if probabilidade <= 0.30 else "CRÉDITO NEGADO"
        
        return jsonify({
            'decisao': decisao,
            'risco': f"{probabilidade:.1%}"
        })

    except Exception as e:
        # Isso vai imprimir o erro exato no seu PowerShell para sabermos o que houve
        traceback.print_exc() 
        return jsonify({'erro': f"Erro no processamento: {str(e)}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
