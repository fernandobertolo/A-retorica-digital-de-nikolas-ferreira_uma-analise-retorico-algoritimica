import pandas as pd
import spacy
import plotly.express as px


def analisar_complexidade_lexical(
        transcricoes_dados_dict,
        nome_campo_data="data_criacao",
        nome_campo_texto="transcricao",
        intervalo="ME",
        arquivo_saida="complexidade_lexical_temporal.html"
):
    """
    Processa dados textuais para calcular e plotar a evolução temporal
    da complexidade lexical (volume total de substantivos vs. léxico único).
    """
    # 1. Carregamento do modelo
    nlp = spacy.load("pt_core_news_lg")

    # 2. Conversão e ordenação do DataFrame
    df = pd.DataFrame(transcricoes_dados_dict)
    df[nome_campo_data] = pd.to_datetime(df[nome_campo_data])
    df = df.sort_values(nome_campo_data)

    lista_total_palavras = []
    lista_palavras_unicas = []

    # 3. Processamento linguístico otimizado
    dados_transcricoes = df[nome_campo_texto].astype(str).tolist()
    documentos_processados = nlp.pipe(dados_transcricoes, disable=["parser", "ner"], batch_size=50)

    for doc in documentos_processados:
        lemas_validos = [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in ["NOUN", "PROPN"] and token.is_alpha
        ]

        lista_total_palavras.append(len(lemas_validos))
        lista_palavras_unicas.append(len(set(lemas_validos)))

    # 4. Inserção de métricas e limpeza de memória
    df['total_palavras'] = lista_total_palavras
    df['palavras_unicas'] = lista_palavras_unicas
    df = df.drop(columns=[nome_campo_texto])

    # 5. Agregação Temporal
    df.set_index(nome_campo_data, inplace=True)
    df_temporal = df[['total_palavras', 'palavras_unicas']].resample(intervalo).mean().reset_index()

    # 6. Reestruturação estrutural (formato longo)
    df_longo = df_temporal.melt(
        id_vars=nome_campo_data,
        value_vars=['total_palavras', 'palavras_unicas'],
        var_name='metrica',
        value_name='quantidade'
    )

    mapa_nomes = {
        'total_palavras': 'Total de Substantivos (Volume)',
        'palavras_unicas': 'Léxico Único (Riqueza Vocabular)'
    }
    df_longo['metrica'] = df_longo['metrica'].map(mapa_nomes)

    # 7. Renderização Gráfica
    fig = px.line(
        df_longo,
        x=nome_campo_data,
        y="quantidade",
        color="metrica",
        color_discrete_map={
            'Total de Substantivos (Volume)': '#2c3e50',
            'Léxico Único (Riqueza Vocabular)': '#e67e22'
        },
        title="Evolução da Complexidade Lexical: Volume Total vs. Palavras Únicas (Média por Vídeo)",
        labels={'quantidade': 'Quantidade de Palavras', nome_campo_data: 'Período', 'metrica': 'Métrica Lexical'},
        markers=True
    )

    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # 8. Exportação
    fig.write_html(arquivo_saida, include_plotlyjs="cdn")

    return fig