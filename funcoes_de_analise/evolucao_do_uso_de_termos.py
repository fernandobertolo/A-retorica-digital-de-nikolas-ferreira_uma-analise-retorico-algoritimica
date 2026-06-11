import pandas as pd
import numpy as np
import re
import spacy
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objects as go


def evolucao_do_uso_de_termos(lista_de_listas_termos,
                              transcricoes_dados_dict,
                              nome_do_campo_data="data_criacao",
                              nome_do_campo_texto="transcricao",
                              categorias_gramaticais={"NOUN", "PROPN"},
                              intervalo="ME",
                              linha_tendencia=False):
    # 1. Normalização Estrutural da Entrada
    lista_normalizada = []
    for item in lista_de_listas_termos:
        if isinstance(item, list):
            lista_normalizada.append(item)
        elif isinstance(item, str):
            lista_normalizada.append([item])

    termos_unicos = []
    for sublista in lista_normalizada:
        for t in sublista:
            t_lower = t.lower()
            if t_lower not in termos_unicos:
                termos_unicos.append(t_lower)

    padroes_regex = {termo: re.compile(rf'\b{re.escape(termo)}\b') for termo in termos_unicos}

    # 2. Construção do DataFrame Base
    df = pd.DataFrame(transcricoes_dados_dict)
    df[nome_do_campo_data] = pd.to_datetime(df[nome_do_campo_data])
    df = df.sort_values(nome_do_campo_data)

    nlp = spacy.load("pt_core_news_lg")
    dados_transcricoes = df[nome_do_campo_texto].astype(str).tolist()
    documentos_processados = nlp.pipe(dados_transcricoes, disable=["parser", "ner"], batch_size=50)

    # 3. Processamento Estruturado em Matriz
    dados_contagem = []
    for doc in documentos_processados:
        total_validos = sum(1 for token in doc if token.pos_ in categorias_gramaticais)
        total_validos = total_validos if total_validos > 0 else 1

        texto_limpo = doc.text.lower()

        linha_contagem = {'total_tokens_validos': total_validos}
        for termo in termos_unicos:
            linha_contagem[termo] = len(padroes_regex[termo].findall(texto_limpo))

        dados_contagem.append(linha_contagem)

    df_contagens = pd.DataFrame(dados_contagem, index=df.index)
    df = pd.concat([df, df_contagens], axis=1)

    for termo in termos_unicos:
        df[termo] = (df[termo] / df['total_tokens_validos']) * 100

    df = df.drop(columns=[nome_do_campo_texto, 'total_tokens_validos'])

    # 4. Agrupamento Temporal
    df.set_index(nome_do_campo_data, inplace=True)
    df_agrupado = df[termos_unicos].resample(intervalo).mean()
    df_longo = df_agrupado.reset_index().melt(id_vars=nome_do_campo_data, var_name='termo', value_name='frequencia')

    # 5. Mapeamento de Clusters e Cores
    cores = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF", "#C0C0C0", "#808080",
        "#800000", "#808000", "#008000", "#800080", "#008080", "#000080", "#FF4500", "#FFA500",
        "#FFD700", "#ADFF2F", "#32CD32", "#00FA9A", "#00CED1", "#4682B4", "#1E90FF", "#4169E1",
        "#8A2BE2", "#9932CC", "#C71585", "#FF1493", "#FF69B4", "#FFC0CB", "#F5DEB3", "#D2B48C",
        "#BC8F8F", "#A0522D", "#8B4513", "#2F4F4F", "#708090", "#778899", "#B0C4DE", "#E6E6FA",
        "#DDA0DD", "#EE82EE", "#DA70D6", "#BA55D3", "#9370DB", "#6A5ACD", "#483D8B", "#20B2AA",
        "#000000", "#FFFFFF"
    ]

    grupos = {}
    cores_base = {}

    for i, sublista in enumerate(lista_normalizada):
        nome_cluster = f"Cluster {i + 1}"
        grupos[nome_cluster] = [t.lower() for t in sublista]
        cores_base[nome_cluster] = cores[i % len(cores)]

    def mapear_conjunto(termo):
        for conjunto, palavras in grupos.items():
            if termo in palavras:
                return conjunto
        return "Outro"

    df_longo['conjunto'] = df_longo['termo'].apply(mapear_conjunto)
    mapa_cores_termos = {termo: cores_base[mapear_conjunto(termo)] for termo in termos_unicos}

    # 6. Visualização Plotly
    df_longo = df_longo.sort_values(['data_criacao', 'conjunto', 'termo'])

    fig = px.area(
        df_longo,
        x="data_criacao",
        y="frequencia",
        color="termo",
        color_discrete_map=mapa_cores_termos,
        title="Evolução Temporal do Uso de Termos por Conjunto",
        line_shape="spline",
        labels={'frequencia': 'Proporção no Vocabulário (%)', 'data_criacao': 'Período', 'termo': 'Termo Extraído'}
    )

    fig.update_traces(line=dict(smoothing=1.0, width=0.3))

    # 7. Adição de Linhas de Tendência por Cluster
    if linha_tendencia:
        # Agrupa a soma de frequências de todos os termos dentro do mesmo cluster
        df_tendencia = df_longo.groupby(['data_criacao', 'conjunto'])['frequencia'].sum().reset_index()

        for conjunto in df_tendencia['conjunto'].unique():
            df_sub = df_tendencia[df_tendencia['conjunto'] == conjunto].dropna()

            # Necessário mínimo de 2 pontos para traçar uma reta
            if len(df_sub) > 1:
                # Conversão de data para formato numérico (ordinal) para regressão
                x_num = pd.to_numeric(df_sub['data_criacao'].map(pd.Timestamp.toordinal))
                y = df_sub['frequencia']

                # Regressão linear polinomial de grau 1
                coeficientes = np.polyfit(x_num, y, 1)
                polinomio = np.poly1d(coeficientes)

                fig.add_trace(go.Scatter(
                    x=df_sub['data_criacao'],
                    y=polinomio(x_num),
                    mode='lines',
                    line=dict(dash='dash', color=cores_base.get(conjunto, '#000000'), width=3),
                    name=f'Tendência ({conjunto})',
                    hoverinfo='skip'
                ))

    fig.update_layout(hovermode="x unified")
    fig.write_html("evolucao_termos_clusters.html", include_plotlyjs="cdn")

    return fig




def evolucao_do_uso_de_termos_barras(lista_de_listas_termos,
                              transcricoes_dados_dict,
                              nome_do_campo_data="data_criacao",
                              nome_do_campo_texto="transcricao",
                              categorias_gramaticais={"NOUN", "PROPN"},
                              intervalo="ME",
                              linha_tendencia=False):
    # 1. Normalização Estrutural da Entrada
    lista_normalizada = []
    for item in lista_de_listas_termos:
        if isinstance(item, list):
            lista_normalizada.append(item)
        elif isinstance(item, str):
            lista_normalizada.append([item])

    termos_unicos = []
    for sublista in lista_normalizada:
        for t in sublista:
            t_lower = t.lower()
            if t_lower not in termos_unicos:
                termos_unicos.append(t_lower)

    padroes_regex = {termo: re.compile(rf'\b{re.escape(termo)}\b') for termo in termos_unicos}

    # 2. Construção do DataFrame Base
    df = pd.DataFrame(transcricoes_dados_dict)
    df[nome_do_campo_data] = pd.to_datetime(df[nome_do_campo_data])
    df = df.sort_values(nome_do_campo_data)

    nlp = spacy.load("pt_core_news_lg")
    dados_transcricoes = df[nome_do_campo_texto].astype(str).tolist()
    documentos_processados = nlp.pipe(dados_transcricoes, disable=["parser", "ner"], batch_size=50)

    # 3. Processamento Estruturado em Matriz
    dados_contagem = []
    for doc in documentos_processados:
        total_validos = sum(1 for token in doc if token.pos_ in categorias_gramaticais)
        total_validos = total_validos if total_validos > 0 else 1

        texto_limpo = doc.text.lower()

        linha_contagem = {'total_tokens_validos': total_validos}
        for termo in termos_unicos:
            linha_contagem[termo] = len(padroes_regex[termo].findall(texto_limpo))

        dados_contagem.append(linha_contagem)

    df_contagens = pd.DataFrame(dados_contagem, index=df.index)
    df = pd.concat([df, df_contagens], axis=1)

    for termo in termos_unicos:
        df[termo] = (df[termo] / df['total_tokens_validos']) * 100

    df = df.drop(columns=[nome_do_campo_texto, 'total_tokens_validos'])

    # 4. Agrupamento Temporal
    df.set_index(nome_do_campo_data, inplace=True)
    df_agrupado = df[termos_unicos].resample(intervalo).mean()
    df_longo = df_agrupado.reset_index().melt(id_vars=nome_do_campo_data, var_name='termo', value_name='frequencia')

    # 5. Mapeamento de Clusters e Cores
    cores = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF", "#C0C0C0", "#808080",
        "#800000", "#808000", "#008000", "#800080", "#008080", "#000080", "#FF4500", "#FFA500",
        "#FFD700", "#ADFF2F", "#32CD32", "#00FA9A", "#00CED1", "#4682B4", "#1E90FF", "#4169E1",
        "#8A2BE2", "#9932CC", "#C71585", "#FF1493", "#FF69B4", "#FFC0CB", "#F5DEB3", "#D2B48C",
        "#BC8F8F", "#A0522D", "#8B4513", "#2F4F4F", "#708090", "#778899", "#B0C4DE", "#E6E6FA",
        "#DDA0DD", "#EE82EE", "#DA70D6", "#BA55D3", "#9370DB", "#6A5ACD", "#483D8B", "#20B2AA",
        "#000000", "#FFFFFF"
    ]

    grupos = {}
    cores_base = {}

    for i, sublista in enumerate(lista_normalizada):
        nome_cluster = f"Cluster {i + 1}"
        grupos[nome_cluster] = [t.lower() for t in sublista]
        cores_base[nome_cluster] = cores[i % len(cores)]

    def mapear_conjunto(termo):
        for conjunto, palavras in grupos.items():
            if termo in palavras:
                return conjunto
        return "Outro"

    df_longo['conjunto'] = df_longo['termo'].apply(mapear_conjunto)
    mapa_cores_termos = {termo: cores_base[mapear_conjunto(termo)] for termo in termos_unicos}

    # 6. Visualização Plotly
    df_longo = df_longo.sort_values(['data_criacao', 'conjunto', 'termo'])

    # Alterado para px.bar e removido line_shape
    fig = px.bar(
        df_longo,
        x="data_criacao",
        y="frequencia",
        color="termo",
        color_discrete_map=mapa_cores_termos,
        title="Evolução Temporal do Uso de Termos por Conjunto",
        labels={'frequencia': 'Proporção no Vocabulário (%)', 'data_criacao': 'Período', 'termo': 'Termo Extraído'}
    )

    fig.update_traces(marker_line_width=0.3, marker_line_color='black')

    # 7. Adição de Linhas de Tendência por Cluster
    if linha_tendencia:
        # Agrupa a soma de frequências de todos os termos dentro do mesmo cluster
        df_tendencia = df_longo.groupby(['data_criacao', 'conjunto'])['frequencia'].sum().reset_index()

        for conjunto in df_tendencia['conjunto'].unique():
            df_sub = df_tendencia[df_tendencia['conjunto'] == conjunto].dropna()

            # Necessário mínimo de 2 pontos para traçar uma reta
            if len(df_sub) > 1:
                # Conversão de data para formato numérico (ordinal) para regressão
                x_num = pd.to_numeric(df_sub['data_criacao'].map(pd.Timestamp.toordinal))
                y = df_sub['frequencia']

                # Regressão linear polinomial de grau 1
                coeficientes = np.polyfit(x_num, y, 1)
                polinomio = np.poly1d(coeficientes)

                fig.add_trace(go.Scatter(
                    x=df_sub['data_criacao'],
                    y=polinomio(x_num),
                    mode='lines',
                    line=dict(dash='dash', color=cores_base.get(conjunto, '#000000'), width=6),
                    name=f'Tendência ({conjunto})',
                    hoverinfo='skip'
                ))

    fig.update_layout(hovermode="x unified")
    fig.write_html("evolucao_termos_clusters.html", include_plotlyjs="cdn")

    return fig