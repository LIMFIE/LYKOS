# Bibliotecas

import os
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu as om
from scipy.optimize import minimize
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


# Agora você pode usar norm.ppf() na sua função calcular_var_parametrico


# Base de Dados CSV

if "2_carteira.csv" not in os.listdir():

    dict1 = {
        "Data": [],
        "Categoria": [],
        "País": [],
        "Setor": [],
        "Ativo": [],
        "Operação": [],
        "Quantidade": [],
        "Preço": [],
        "Verificado": [],
    }
    carteira = pd.DataFrame(dict1).set_index("Data")
    carteira.to_csv("2_carteira.csv")

if "3_patrimonio.csv" not in os.listdir():

    dict1 = {
        "Data": [],
        "Categoria": [],
        "País": [],
        "Setor": [],
        "Ativo": [],
        "Operação": [],
        "Patrimônio": [],
    }
    patrimonio = pd.DataFrame(dict1).set_index("Data")
    patrimonio.to_csv("3_patrimonio.csv")

# Leitura de CSV

df_ativos = pd.read_csv("1_ativos.csv")
carteira = pd.read_csv("2_carteira.csv")

# Configuração básica da página

st.set_page_config(page_icon="wolf", page_title="LIMFIE", layout="wide")
st.sidebar.title("FUNDO LYKOS")

# Configuração básica da Sidebar

with st.sidebar:
    pagina = om(
        menu_title=None,
        options=["Carteira", "Posição", "Resultados"],
    )
# Inicializar variáveis globais
carteira = pd.DataFrame()
patrimonio = pd.DataFrame()

# Carregar dados de carteira e patrimônio, se existirem
try:
    carteira = pd.read_csv("2_carteira.csv")
except FileNotFoundError:
    carteira = pd.DataFrame(columns=[
        "Data", "Categoria", "País", "Setor", "Ativo", "Operação", "Quantidade", "Preço", "Verificado"
    ])

try:
    patrimonio = pd.read_csv("3_patrimonio.csv")
    patrimonio["Data"] = pd.to_datetime(patrimonio["Data"]).dt.date
except FileNotFoundError:
    patrimonio = pd.DataFrame(columns=[
        "Data", "Categoria", "País", "Setor", "Ativo", "Operação", "Valor"
    ])

if pagina == "Carteira":

    # Criação do setor de adicionar ativos
    st.header("Adicionar Ativos: ")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        categoria = st.selectbox(
            "Categoria do ativo: ", options=df_ativos["Tipo"].unique()
        )

    with col2:
        pais = st.selectbox(
            "País do ativo: ",
            options=df_ativos.set_index("Tipo").loc[categoria, "País"].unique(),
        )

    with col3:
        setor = st.selectbox(
            "Selecione o setor:",
            options=df_ativos.set_index("Tipo")
            .loc[categoria]
            .reset_index()
            .set_index("País")
            .loc[pais, "Setor"]
            .unique(),
        )

    with col4:
        ativo = st.selectbox(
            "Selecione o ativo:",
            options=df_ativos.set_index("Tipo")
            .loc[categoria]
            .reset_index()
            .set_index("País")
            .loc[pais]
            .reset_index()
            .set_index("Setor")
            .loc[setor, "Ativo"]
            .unique(),
        )

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        operacao = st.selectbox("Selecione a operação:", options=["Long", "Short"])

    with col6:
        quantidade = st.number_input("Quantidade:", 1, 100000, 1)

    with col7:
        valor = st.number_input("Preço da operação:", 0.0, 1000.0, 0.01)  

    with col8:

        def adicionar():
            global carteira

            # Adicionar um ativo que ainda não está na carteira
            if carteira.empty or ativo not in carteira["Ativo"].unique():
                carteira.loc[len(carteira)] = [
                    datetime.today().date(),
                    categoria,
                    pais,
                    setor,
                    ativo,
                    operacao,
                    quantidade,
                    valor,
                    False,
                ]
                carteira = carteira.reindex(columns=ordem)
                carteira.to_csv("2_carteira.csv", index=False)

            # Adicionar um ativo já existente mas com a operação contrária
            elif (
                ativo in (carteira["Ativo"].unique())
                and operacao != carteira.set_index("Ativo").loc[ativo, "Operação"]
            ):
                conta = (
                    carteira.set_index("Ativo").loc[ativo, "Quantidade"] - quantidade
                )

                # Estamos posicionado "Long" e realizamos parte da posição
                if (
                    carteira.set_index("Ativo").loc[ativo, "Operação"] == "Long"
                    and conta > 0
                ):
                    carteira = carteira.set_index("Ativo")
                    carteira.loc[ativo, "Quantidade"] = conta
                    carteira.loc[ativo, "Verificado"] = False
                    carteira.loc[ativo, "Data"] = datetime.today().date()
                    carteira = carteira.reset_index()
                    carteira = carteira.reindex(columns=ordem)
                    carteira.to_csv("2_carteira.csv", index=False)

                # Estamos posicionado "Long" mas invertemos a posição
                elif (
                    carteira.set_index("Ativo").loc[ativo, "Operação"] == "Long"
                    and conta < 0
                ):
                    carteira = carteira.set_index("Ativo")
                    carteira.loc[ativo, "Quantidade"] = abs(conta)
                    carteira.loc[ativo, "Operação"] = "Short"
                    carteira.loc[ativo, "Preço"] = valor
                    carteira.loc[ativo, "Verificado"] = False
                    carteira.loc[ativo, "Data"] = datetime.today().date()
                    carteira = carteira.reset_index()
                    carteira = carteira.reindex(columns=ordem)
                    carteira.to_csv("2_carteira.csv", index=False)

                # Estamos posicionado "Long" e realizamos toda posição
                elif (
                    carteira.set_index("Ativo").loc[ativo, "Operação"] == "Long"
                    and conta == 0
                ):
                    carteira = carteira.set_index("Ativo")
                    carteira = carteira.drop(ativo)
                    carteira = carteira.reset_index()
                    carteira = carteira.reindex(columns=ordem)
                    carteira.to_csv("2_carteira.csv", index=False)

                # Estamos posicionado "Short" e realizamos parte da posição
                elif (
                    carteira.set_index("Ativo").loc[ativo, "Operação"] == "Short"
                    and conta > 0
                ):
                    carteira = carteira.set_index("Ativo")
                    carteira.loc[ativo, "Quantidade"] = conta
                    carteira.loc[ativo, "Verificado"] = False
                    carteira.loc[ativo, "Data"] = datetime.today().date()
                    carteira = carteira.reset_index()
                    carteira = carteira.reindex(columns=ordem)
                    carteira.to_csv("2_carteira.csv", index=False)

                # Estamos posicionado "Short" mas invertemos a posição
                elif (
                    carteira.set_index("Ativo").loc[ativo, "Operação"] == "Short"
                    and conta < 0
                ):
                    carteira = carteira.set_index("Ativo")
                    carteira.loc[ativo, "Quantidade"] = abs(conta)
                    carteira.loc[ativo, "Operação"] = "Long"
                    carteira.loc[ativo, "Preço"] = valor
                    carteira.loc[ativo, "Verificado"] = False               
                    carteira.loc[ativo, "Data"] = datetime.today().date()
                    carteira = carteira.reset_index()
                    carteira = carteira.reindex(columns=ordem)
                    carteira.to_csv("2_carteira.csv", index=False)

                # Estamos posicionado "Short" e realizamos toda posição
                elif (
                    carteira.set_index("Ativo").loc[ativo, "Operação"] == "Short"
                    and conta == 0
                ):
                    carteira = carteira.set_index("Ativo")
                    carteira = carteira.drop(ativo)
                    carteira = carteira.reset_index()
                    carteira = carteira.reindex(columns=ordem)
                    carteira.to_csv("2_carteira.csv", index=False)

            # Aumento de posição e preço médio
            elif (
                ativo in (carteira["Ativo"].unique())
                and operacao == carteira.set_index("Ativo").loc[ativo, "Operação"]
            ):
                conta = (
                    carteira.set_index("Ativo").loc[ativo, "Quantidade"] + quantidade
                )

                media = (
                    (
                        carteira.set_index("Ativo").loc[ativo, "Quantidade"]
                        * carteira.set_index("Ativo").loc[ativo, "Preço"]
                    )
                    + (quantidade * valor)
                ) / (carteira.set_index("Ativo").loc[ativo, "Quantidade"] + quantidade)

                carteira = carteira.set_index("Ativo")
                carteira.loc[ativo, "Quantidade"] = conta
                carteira.loc[ativo, "Preço"] = media
                carteira.loc[ativo, "Verificado"] = False
                carteira.loc[ativo, "Data"] = datetime.today().date()
                carteira = carteira.reset_index()
                carteira = carteira.reindex(columns=ordem)
                carteira.to_csv("2_carteira.csv", index=False)

        st.write("")
        st.write("")
        bt_adicionar = st.button(
            "Adicionar", use_container_width=True, on_click=adicionar
        )

    st.divider()

    # Colocando ordem nas colunas da carteira para visualização
    ordem = [
        "Data",
        "Categoria",
        "País",
        "Setor",
        "Ativo",
        "Operação",
        "Quantidade",
        "Preço",
        "Verificado",
    ]
    carteira = carteira.reindex(columns=ordem)

    # Visualização das posições na carteira
    st.dataframe(carteira.drop(columns="Verificado"), use_container_width=True)

    # Evolução patrimonial
    for i in range(len(carteira)):

        if patrimonio.empty or (
            carteira.loc[i, "Verificado"] == False
            and carteira.loc[i, "Ativo"] not in patrimonio["Ativo"].unique()
        ):
            patrimonio.loc[len(patrimonio)] = [
                carteira.loc[i, "Data"],
                carteira.loc[i, "Categoria"],
                carteira.loc[i, "País"],
                carteira.loc[i, "Setor"],
                carteira.loc[i, "Ativo"],
                carteira.loc[i, "Operação"],
                (carteira.loc[i, "Quantidade"] * carteira.loc[i, "Preço"]),
            ]

            carteira.loc[i, "Verificado"] = True
            carteira.to_csv("2_carteira.csv", index=False)
            patrimonio.to_csv("3_patrimonio.csv", index=False)

        else:
            ultima_data_patrimonio = pd.Series(
                patrimonio.set_index("Ativo").loc[carteira.loc[i, "Ativo"], "Data"]
            ).iloc[-1]

            ultima_data_yfinance = (
                yf.Ticker(carteira.loc[i, "Ativo"]).history().index[-1]
            ).date()

            if ultima_data_patrimonio >= ultima_data_yfinance:
                pass

            elif carteira.loc[i, "Verificado"] == True:
                fechamentos = yf.download(
                    carteira.loc[i, "Ativo"], start=(ultima_data_patrimonio) + timedelta(1)
                )["Adj Close"]

                fechamentos = fechamentos * carteira.loc[i, "Quantidade"]

                for j in range(len(fechamentos)):
                    patrimonio.loc[len(patrimonio)] = [
                        fechamentos.index[j].date(),
                        carteira.loc[i, "Categoria"],
                        carteira.loc[i, "País"],
                        carteira.loc[i, "Setor"],
                        carteira.loc[i, "Ativo"],
                        carteira.loc[i, "Operação"],
                        fechamentos.iloc[j],
                    ]

                    patrimonio = patrimonio.sort_values(["Ativo", "Data"])
                    patrimonio.to_csv("3_patrimonio.csv", index=False)

            elif carteira.loc[i, "Verificado"] == False:
                patrimonio.loc[len(patrimonio)] = [
                    carteira.loc[i, "Data"],
                    carteira.loc[i, "Categoria"],
                    carteira.loc[i, "País"],
                    carteira.loc[i, "Setor"],
                    carteira.loc[i, "Ativo"],
                    carteira.loc[i, "Operação"],
                    (carteira.loc[i, "Quantidade"] * carteira.loc[i, "Preço"]),
                ]
                carteira.loc[i, "Verificado"] = True
                carteira.to_csv("2_carteira.csv", index=False)
                patrimonio = patrimonio.sort_values(["Ativo", "Data"])
                patrimonio.to_csv("3_patrimonio.csv", index=False)
# Evolução patrimonial

carteira = pd.read_csv("2_carteira.csv")
patrimonio = pd.read_csv("3_patrimonio.csv")
patrimonio["Data"] = pd.to_datetime(patrimonio["Data"]).dt.date

for i in range(len(carteira)):

    if patrimonio.empty or (
        carteira.loc[i, "Verificado"] == False
        and carteira.loc[i, "Ativo"] not in patrimonio["Ativo"].unique()
    ):
        patrimonio.loc[len(patrimonio)] = [
            carteira.loc[i, "Data"],
            carteira.loc[i, "Categoria"],
            carteira.loc[i, "País"],
            carteira.loc[i, "Setor"],
            carteira.loc[i, "Ativo"],
            carteira.loc[i, "Operação"],
            (carteira.loc[i, "Quantidade"] * carteira.loc[i, "Preço"]),
        ]

        carteira.loc[i, "Verificado"] = True
        carteira.to_csv("2_carteira.csv", index=False)
        patrimonio.to_csv("3_patrimonio.csv", index=False)

    else:

        ultima_data_patrimonio = pd.Series(
            patrimonio.set_index("Ativo").loc[carteira.loc[i, "Ativo"], "Data"]
        ).iloc[-1]

        ultima_data_yfinance = (
            yf.Ticker(carteira.loc[i, "Ativo"]).history().index[-1]
        ).date()

        if ultima_data_patrimonio >= ultima_data_yfinance:
            pass

        elif carteira.loc[i, "Verificado"] == True:

            fechamentos = yf.download(
                carteira.loc[i, "Ativo"], start=(ultima_data_patrimonio) + timedelta(1)
            )["Adj Close"]

            fechamentos = fechamentos * carteira.loc[i, "Quantidade"]

            for j in range(len(fechamentos)):

                patrimonio.loc[len(patrimonio)] = [
                    fechamentos.index[j].date(),
                    carteira.loc[i, "Categoria"],
                    carteira.loc[i, "País"],
                    carteira.loc[i, "Setor"],
                    carteira.loc[i, "Ativo"],
                    carteira.loc[i, "Operação"],
                    fechamentos.iloc[j],
                ]

                patrimonio = patrimonio.sort_values(["Ativo", "Data"])
                patrimonio.to_csv("3_patrimonio.csv", index=False)

        elif carteira.loc[i, "Verificado"] == False:

            patrimonio.loc[len(patrimonio)] = [
                carteira.loc[i, "Data"],
                carteira.loc[i, "Categoria"],
                carteira.loc[i, "País"],
                carteira.loc[i, "Setor"],
                carteira.loc[i, "Ativo"],
                carteira.loc[i, "Operação"],
                (carteira.loc[i, "Quantidade"] * carteira.loc[i, "Preço"]),
            ]
            carteira.loc[i, "Verificado"] = True
            carteira.to_csv("2_carteira.csv", index=False)
            patrimonio = patrimonio.sort_values(["Ativo", "Data"])
            patrimonio.to_csv("3_patrimonio.csv", index=False)

#########################################################################################################33
if pagina == "Posição":

    col9, col10 = st.columns(2)

    data_divisao = patrimonio.sort_values("Data", ascending=True)["Data"].iloc[-1]

    with col9:

        divisao_pais = (
            patrimonio.drop(columns=["Categoria", "Setor", "Ativo", "Operação"])
            .groupby(["Data", "País"])
            .sum()
            .loc[data_divisao]
            .reset_index()
        )

        fig_pais = px.pie(divisao_pais, "País", "Patrimônio")
        st.plotly_chart(fig_pais, use_container_width=True)

    with col10:

        divisao_setor = (
            patrimonio.drop(columns=["Categoria", "País", "Ativo", "Operação"])
            .groupby(["Data", "Setor"])
            .sum()
            .loc[data_divisao]
            .reset_index()
        )

        fig_setor = px.pie(divisao_setor, "Setor", "Patrimônio")
        st.plotly_chart(fig_setor, use_container_width=True)

    # Normalizando os dados dos ativos

    normal = patrimonio.drop(columns=["Categoria", "País", "Setor"])
    normal = (
        (
            (
                normal.groupby(["Ativo", "Operação", "Data"]).sum()
                / normal.groupby(["Ativo", "Operação"]).first()
            )
            - 1
        )
        .drop(columns="Data")
        .reset_index()
    )

    normal.loc[normal["Operação"] == "Short", "Patrimônio"] *= -1
    normal = (
        normal.drop(columns=["Ativo", "Operação"]).groupby("Data").sum().reset_index()
    )

    fig_evolucao = px.line(normal, "Data", "Patrimônio")
    st.plotly_chart(fig_evolucao, use_container_width=True)
################################################################################################################################################

def calcular_plotar_drawdown_carteira(carteira_com_pesos):
    # Data de início e fim para baixar dados
    data_fim = pd.Timestamp.now().date()
    data_inicio = data_fim - pd.DateOffset(months=36)  # 36 meses atrás
    
    # Lista para armazenar os dados de retorno acumulado de cada ativo
    retorno_acumulado_ativos = []
    # Baixar dados históricos ajustados do Yahoo Finance para cada ativo na carteira
    for index, row in carteira_com_pesos.iterrows():
        ativo = row['Ativo']
        peso = row['Peso']
        try:
            dados = yf.download(ativo, start=data_inicio, end=data_fim)['Adj Close']
            if dados.empty:
                st.warning(f"Dados vazios para o ativo {ativo}. Verifique o ticker ou tente novamente mais tarde.")
            else:
                # Preencher dados faltantes com interpolação linear
                dados = dados.interpolate(method='linear')
                
                # Garantir que todos os dados tenham o mesmo índice de datas
                datas_completas = pd.date_range(start=data_inicio, end=data_fim, freq='D')
                dados = dados.reindex(datas_completas)
                
                # Preencher novamente dados faltantes após a reindexação
                dados = dados.interpolate(method='linear')
                
                # Calcular o retorno acumulado do ativo
                dados[f'Retorno_Acumulado_{ativo}'] = (1 + dados.pct_change()).cumprod()
                
                # Salvar o retorno acumulado do ativo na lista
                retorno_acumulado_ativos.append(dados[f'Retorno_Acumulado_{ativo}'] * peso)
                
        except Exception as e:
            st.error(f"Erro ao obter dados para o ativo {ativo}: {str(e)}")

    # Calcular o retorno acumulado da carteira
    if retorno_acumulado_ativos:
        retorno_acumulado_carteira = pd.concat(retorno_acumulado_ativos, axis=1).sum(axis=1)
        retorno_acumulado_carteira.dropna(inplace=True)
        
        # Calcular o drawdown da carteira com janela de 5 dias
        max_retorno_acumulado = retorno_acumulado_carteira.rolling(window=5, min_periods=1).max()
        drawdown_carteira = (retorno_acumulado_carteira / max_retorno_acumulado) - 1
        
        # Encontrar os pontos de máximo drawdown
        drawdown_maximo = drawdown_carteira.min()
        data_drawdown_maximo = drawdown_carteira.idxmin()

        # Encontrar os pontos de mínimo drawdown
        drawdown_minimo = drawdown_carteira.max()
        data_drawdown_minimo = drawdown_carteira.idxmax()
        
        # Plotar o gráfico de drawdown da carteira com anotações
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(drawdown_carteira.index, drawdown_carteira * 100, linestyle='-', color='steelblue', linewidth=2, label='Drawdown da Carteira (5 Dias)')
        
        # Destacar o ponto de máximo drawdown
        ax.plot(data_drawdown_maximo, drawdown_maximo * 100, marker='o', color='red', markersize=10)

        # Destacar o ponto de mínimo drawdown
        ax.plot(data_drawdown_minimo, drawdown_minimo * 100, marker='o', color='green', markersize=10)

        # Anotação para o ponto de máximo drawdown
        ax.annotate(f'Máximo: {drawdown_maximo*100:.2f}% em {data_drawdown_maximo.strftime("%Y-%m-%d")}',
                    xy=(data_drawdown_maximo, drawdown_maximo * 100),
                    xytext=(data_drawdown_maximo + pd.DateOffset(days=5), (drawdown_maximo - 0.02) * 100),
                    arrowprops=dict(facecolor='black', arrowstyle='->'),
                    fontsize=10, ha='center')

        # Anotação para o ponto de mínimo drawdown
        ax.annotate(f'Mínimo: {drawdown_minimo*100:.2f}% em {data_drawdown_minimo.strftime("%Y-%m-%d")}',
                    xy=(data_drawdown_minimo, drawdown_minimo * 100),
                    xytext=(data_drawdown_minimo + pd.DateOffset(days=5), (drawdown_minimo + 0.02) * 100),
                    arrowprops=dict(facecolor='black', arrowstyle='->'),
                    fontsize=10, ha='center')

        # Estilo dos eixos e legendas
        ax.set_xlabel('Data', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12, color='steelblue')
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax.legend(loc='upper left', fontsize=12)

        # Título do gráfico
        plt.title('Drawdown da Carteira (Janela de 5 Dias) nos Últimos 36 Meses', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.warning("Não foi possível calcular o drawdown da carteira devido a problemas com os dados dos ativos.")                           

    
def calcular_pesos_por_valor(carteira):
    carteira['Valor_Investido'] = carteira['Quantidade'] * carteira['Preço']
    total_investido = carteira['Valor_Investido'].sum()
    carteira['Peso'] = carteira['Valor_Investido'] / total_investido
    return carteira




# Função para calcular correlação e covariância sem o IBOV
def calcular_correlacao_covariancia(carteira):
    df_retornos = pd.DataFrame()

    data_fim = pd.Timestamp.now().date()
    data_inicio = data_fim - pd.DateOffset(months=36)  # Busca dados para os últimos 60 meses

    # Obter dados dos ativos da carteira
    carteira_com_pesos = calcular_pesos_por_valor(carteira)
    for ativo in carteira_com_pesos['Ativo'].unique():
        try:
            ativo_data = yf.download(ativo, start=data_inicio, end=data_fim)['Adj Close']
            if ativo_data.empty:
                st.warning(f"Dados vazios para o ativo {ativo}. Verifique o ticker ou tente novamente mais tarde.")
            else:
                df_retornos[f'{ativo}_retorno'] = ativo_data.pct_change()
        except Exception as e:
            st.error(f"Erro ao obter dados para o ativo {ativo}: {str(e)}")

    df_retornos.dropna(inplace=True)

    if df_retornos.empty:
        st.warning("Não há dados disponíveis para calcular a correlação e covariância.")
        return pd.DataFrame(), pd.DataFrame(), None, None

    # Calcular as matrizes de correlação e covariância sem ponderar pelos pesos
    matriz_corr = df_retornos.corr()
    matriz_cov = df_retornos.cov()

    return df_retornos, matriz_corr, matriz_cov, carteira_com_pesos

# Função para plotar a matriz de correlação
def plotar_matriz_correlacao(matriz_corr):
    st.header('Matriz de Correlação')
    fig, ax = plt.subplots(figsize=(10, 8))  # Ajuste o tamanho da figura aqui
    sns.heatmap(matriz_corr, annot=True, ax=ax)
    st.pyplot(fig)

# Função para plotar a matriz de covariância
def plotar_matriz_covariancia(matriz_cov):
    st.header('Matriz de Covariância')
    fig, ax = plt.subplots(figsize=(10, 8))  # Ajuste o tamanho da figura aqui
    sns.heatmap(matriz_cov, annot=True, ax=ax)
    st.pyplot(fig)

# Função para calcular e plotar a variância da carteira ao longo dos últimos 60 meses usando Plotly
def calcular_e_plotar_variancia(carteira, matriz_cov):
    # Extrair os pesos do DataFrame para um array numpy
    pesos = carteira_com_pesos['Peso'].values

    # Verificar se as dimensões estão alinhadas
    if pesos.shape[0] != matriz_cov.shape[0]:
        st.error(f"Erro: O número de pesos ({pesos.shape[0]}) não corresponde ao tamanho da matriz de covariância ({matriz_cov.shape[0]}).")
        return None

    # Calcular a variância da carteira
    variancia_carteira = np.dot(pesos.T, np.dot(matriz_cov, pesos))

    # Simular dados de variância ao longo dos últimos 60 meses (substitua com dados reais)
    variancias = np.random.normal(variancia_carteira, 0.05 * variancia_carteira, size=60)

    # Criar o gráfico de variância usando Plotly
    fig = go.Figure()

    # Adicionar barras de variância mensal
    fig.add_trace(go.Bar(
        x=list(range(len(variancias))),
        y=variancias,
        name='Variância Mensal',
        marker_color='skyblue'
    ))

    # Adicionar linha de média móvel de 10 meses
    fig.add_trace(go.Scatter(
        x=list(range(len(np.convolve(variancias, np.ones(10)/10, mode='valid')))),
        y=np.convolve(variancias, np.ones(10)/10, mode='valid'),
        mode='lines',
        name='Média Móvel de 10 Meses',
        line=dict(color='orange', dash='dash')
    ))

    # Adicionar linha horizontal para a variância total
    fig.add_trace(go.Scatter(
        x=list(range(len(variancias))),
        y=[variancia_carteira]*len(variancias),
        mode='lines',
        name='Variância Total',
        line=dict(color='red', dash='dot')
    ))

    # Configurações do layout
    fig.update_layout(
        title='Variância da Carteira nos Últimos 60 Meses',
        xaxis_title='Meses',
        yaxis_title='Variância',
        legend_title='Legenda',
        template='plotly_white'
    )

    # Exibir o valor da variância na tela
    st.write(f"A variância da carteira é: {variancia_carteira}")

    # Exibir o gráfico na aplicação Streamlit
    st.plotly_chart(fig)

    return variancia_carteira


def calcular_e_plotar_var_parametrico(df_retornos, carteira_com_pesos, alpha=0.05):
    # Calcular retornos ponderados
    retornos_ponderados = df_retornos @ carteira_com_pesos['Peso'].values
    # Calcular retorno e desvio padrão da carteira
    media_retorno_carteira = retornos_ponderados.mean()
    desvio_padrao_carteira = retornos_ponderados.std()
    # Encontrar o z-score para o nível de confiança desejado
    z_score = norm.ppf(1 - alpha)
    # Calcular VaR paramétrico
    var_parametrico = -z_score * desvio_padrao_carteira
    st.write(f"O Value at Risk (VaR) paramétrico da carteira ao nível de confiança de {alpha*100}% é: {var_parametrico}")
    
    # Plotar a distribuição gaussiana dos retornos ponderados
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(retornos_ponderados, kde=True, stat='density', color='blue', bins=50, ax=ax)
    ax.axvline(var_parametrico, color='r', linestyle='-', label=f'VaR Paramétrico ({alpha*100}%)')
    ax.set_title('Distribuição dos Retornos Ponderados e VaR Paramétrico')
    ax.set_xlabel('Retornos Ponderados')
    ax.set_ylabel('Densidade')
    ax.legend()
    plt.tight_layout()

    # Exibir o gráfico no Streamlit
    st.pyplot(fig)

    return var_parametrico


def calcular_beta_carteira(carteira, ticker_referencia='^BVSP'):
    # Data de início e fim para baixar dados (3 anos para ter dados suficientes)
    data_fim = pd.Timestamp.now().date()
    data_inicio = data_fim - pd.DateOffset(months=60)

    # Obter dados históricos do índice de referência
    try:
        dados_referencia = yf.download(ticker_referencia, start=data_inicio, end=data_fim)['Adj Close']
        if dados_referencia.empty:
            st.warning(f"Dados vazios para o índice de referência {ticker_referencia}. Verifique o ticker ou tente novamente mais tarde.")
            return None
        else:
            retornos_referencia = dados_referencia.pct_change().dropna()
    except Exception as e:
        st.error(f"Erro ao obter dados para o índice de referência {ticker_referencia}: {str(e)}")
        return None

    # Calcular os retornos da carteira
    df_retornos, _, _, carteira_com_pesos = calcular_correlacao_covariancia(carteira)

    if df_retornos.empty:
        st.warning("Não foi possível calcular os retornos da carteira.")
        return None

    retornos_carteira = df_retornos @ carteira_com_pesos['Peso'].values

    # Alinhar as séries temporais
    retornos_carteira = retornos_carteira.loc[retornos_carteira.index.intersection(retornos_referencia.index)]
    retornos_referencia = retornos_referencia.loc[retornos_referencia.index.intersection(retornos_carteira.index)]

    # Calcular a covariância entre os retornos da carteira e os retornos do índice de referência
    cov_carteira_referencia = np.cov(retornos_carteira, retornos_referencia)[0, 1]

    # Calcular a variância dos retornos do índice de referência
    var_referencia = np.var(retornos_referencia)

    # Calcular o beta da carteira
    beta_carteira = cov_carteira_referencia / var_referencia

    # Exibir o beta em uma caixa destacada
    st.info(f"Beta da Carteira: {beta_carteira:.4f}")

    return beta_carteira


# Configuração para desativar o aviso PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Função para calcular os retornos da carteira
def calcular_retornos_carteira(carteira, data_inicio, data_fim):
    df_retornos = pd.DataFrame()

    for ativo in carteira['Ativo'].unique():
        try:
            # Obter os dados do Yahoo Finance para cada ativo na carteira
            dados = yf.download(ativo, start=data_inicio, end=data_fim)['Adj Close']
            if dados.empty:
                st.warning(f"Dados vazios para o ativo {ativo}. Verifique o ticker ou tente novamente mais tarde.")
            else:
                # Calcular os retornos percentuais
                df_retornos[f'{ativo}_retorno'] = dados.pct_change() * 100  # Retornos em porcentagem
        except Exception as e:
            st.error(f"Erro ao obter dados para o ativo {ativo}: {str(e)}")

    df_retornos.dropna(inplace=True)
    return df_retornos

# Função para calcular a média dos retornos e a matriz de covariância
def calcular_media_covariancia(retornos_hist):
    media_retornos = retornos_hist.mean() * 100  # Média dos retornos em porcentagem
    matriz_covariancia = retornos_hist.cov() * 100  # Covariância em porcentagem
    return media_retornos, matriz_covariancia

# Função para maximizar o índice de Sharpe
def maximizar_indice_sharpe(pesos, media_retornos, matriz_covariancia):
    retorno = np.dot(media_retornos, pesos)
    risco = np.sqrt(np.dot(pesos.T, np.dot(matriz_covariancia, pesos)))
    indice_sharpe = -retorno / risco  # Maximizar o negativo do índice de Sharpe
    return indice_sharpe

# Função para encontrar a carteira ótima
def encontrar_carteira_otima(retornos_hist):
    media_retornos, matriz_covariancia = calcular_media_covariancia(retornos_hist)
    
    num_ativos = len(media_retornos)
    pesos_iniciais = np.ones(num_ativos) / num_ativos  # Pesos iniciais iguais
    
    # Restrição: a soma dos pesos deve ser 1
    restricao = ({'type': 'eq', 'fun': lambda pesos: np.sum(pesos) - 1})
    
    # Limites dos pesos (0.01 <= peso <= 1 para cada ativo)
    limites_pesos = tuple((0.01, 0.4) for _ in range(num_ativos))
    
    # Minimização do negativo do índice de Sharpe
    resultado = minimize(maximizar_indice_sharpe, pesos_iniciais, args=(media_retornos, matriz_covariancia),
                         method='SLSQP', bounds=limites_pesos, constraints=restricao)
    
    pesos_otimos = resultado.x
    retorno_otimo = np.dot(media_retornos, pesos_otimos)
    risco_otimo = np.sqrt(np.dot(pesos_otimos.T, np.dot(matriz_covariancia, pesos_otimos)))
    
    return risco_otimo, retorno_otimo, pesos_otimos

# Função para simular a fronteira eficiente de Markowitz
def simular_fronteira_eficiente(retornos_hist, num_simulacoes=100000):
    media_retornos, matriz_covariancia = calcular_media_covariancia(retornos_hist)
    
    num_ativos = len(media_retornos)
    
    resultados = []
    pesos_simulados = []
    
    for _ in range(num_simulacoes):
        pesos = np.random.dirichlet(np.ones(num_ativos))
        retorno_portfolio = np.sum(pesos * media_retornos)
        risco_portfolio = np.sqrt(np.dot(pesos.T, np.dot(matriz_covariancia, pesos)))
        sharpe_ratio = retorno_portfolio / risco_portfolio
        resultados.append([risco_portfolio, retorno_portfolio, sharpe_ratio])
        pesos_simulados.append(pesos)
    
    resultados = np.array(resultados)
    pesos_simulados = np.array(pesos_simulados)
    return resultados, pesos_simulados, matriz_covariancia

# Função para plotar a fronteira eficiente com a carteira ótima
def plotar_fronteira_eficiente(resultados, risco_otimo, retorno_otimo, pesos_otimos, matriz_covariancia, media_retornos):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    risco_portfolio = resultados[:, 0]
    retorno_portfolio = resultados[:, 1]
    
    # Plotar todos os portfólios simulados
    scatter = ax.scatter(risco_portfolio, retorno_portfolio, c=retorno_portfolio / risco_portfolio, marker='o', cmap='viridis', label='Simulação Monte Carlo')
    
    ax.set_title('Fronteira Eficiente de Markowitz (Simulação de Monte Carlo)')
    ax.set_xlabel('Risco (Desvio Padrão)')
    ax.set_ylabel('Retorno Esperado')
    fig.colorbar(scatter, ax=ax, label='Índice de Sharpe')
    
    # Plotar a carteira ótima
    ax.scatter(risco_otimo, retorno_otimo, marker='*', color='r', s=200, label='Carteira Ótima')
    
    ax.legend()
 # Exibir os pesos ótimos em formato de lista
    st.subheader('Pesos Ótimos da Carteira:')
    for i, peso in enumerate(pesos_otimos):
        ativo_nome = carteira['Ativo'].iloc[i]
        st.text(f"{ativo_nome}: {peso:.2f}")

    st.pyplot(fig)


# Exemplo de uso na página 'Resultados'
if pagina == 'Resultados':
    # Supondo que você já tenha calculado df_retornos e carteira_com_pesos
    # Aqui você deve chamar a função calcular_correlacao_covariancia para obter esses dados
    df_retornos, matriz_corr, matriz_cov, carteira_com_pesos = calcular_correlacao_covariancia(carteira)    
    # Excluir a coluna do IBOV, se houver, antes de calcular a matriz de covariância
    if 'IBOV' in df_retornos.columns:
        df_retornos = df_retornos.drop(columns=['IBOV'])
        matriz_cov = df_retornos.cov()  # Recalcular a matriz de covariância
    
     # Calcular o beta da carteira
    beta_carteira = calcular_beta_carteira(carteira)

    calcular_plotar_drawdown_carteira(carteira)
    
    data_fim = pd.Timestamp.now().date()
    data_inicio = data_fim - pd.DateOffset(months=36)  # Exemplo de 5 anos de dados históricos
        
    # Calcular retornos da carteira
    retornos_carteira = calcular_retornos_carteira(carteira, data_inicio, data_fim)
    
    # Calcular média dos retornos e matriz de covariância
    media_retornos, matriz_covariancia = calcular_media_covariancia(retornos_carteira)
    
    # Encontrar a carteira ótima
    risco_otimo, retorno_otimo, pesos_otimos = encontrar_carteira_otima(retornos_carteira)
    
    # Simular fronteira eficiente
    resultados_simulacao, _, _ = simular_fronteira_eficiente(retornos_carteira)
    
    # Plotar fronteira eficiente com a carteira ótima
    plotar_fronteira_eficiente(resultados_simulacao, risco_otimo, retorno_otimo, pesos_otimos, matriz_covariancia, media_retornos)

    # Plotar e exibir a matriz de correlação
    st.subheader('Matriz de Correlação')
    fig_corr, ax_corr = plt.subplots()
    sns.heatmap(matriz_corr, annot=True, ax=ax_corr, cmap='coolwarm', cbar=True, square=True, linewidths=.5)
    ax_corr.set_title('Matriz de Correlação')
    st.pyplot(fig_corr)

    # Plotar e exibir a matriz de covariância escalada
    st.subheader('Matriz de Covariância Escalada')
    fig_cov, ax_cov = plt.subplots()
    matriz_cov_escalada = matriz_cov * 1e5  # Multiplica cada valor por 100,000
    sns.heatmap(matriz_cov_escalada, annot=True, ax=ax_cov, cmap='coolwarm', cbar=True, square=True, linewidths=.5)
    ax_cov.set_title('Matriz de Covariância Escalada')
    st.pyplot(fig_cov)

    # Chamar a função para calcular e plotar a variância da carteira
    variancia_carteira = calcular_e_plotar_variancia(carteira_com_pesos, matriz_cov)
        
    # Calcular e plotar VaR Paramétrico
    calcular_e_plotar_var_parametrico(df_retornos, carteira_com_pesos, alpha=0.05)
    
    # Ajustar o layout para evitar sobreposição
    plt.tight_layout()

