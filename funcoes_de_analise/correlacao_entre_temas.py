import pandas as pd


def correlacao_entre_temas(dados, metodo="pearson", caminho_heatmap=None):
    """
    Calcula a correlação entre temas a partir das pontuações por documento.

    Cada documento é uma observação e cada tema é uma variável. A função monta a matriz
    (documentos x temas), calcula a matriz de correlação entre os temas e resume:
      - para cada tema, qual outro é o mais e o menos correlacionado;
      - o índice de "isolamento" (correlação média, em módulo, com os demais -> quanto
        menor, mais o tema aparece sozinho);
      - quantos documentos têm valor > 0 no tema (para avaliar esparsidade: um tema quase
        sempre zero pode parecer "isolado" apenas por ter pouca variância).

    Args:
        dados (dict | list): Ou um dict {titulo: {tema: valor}} (um por documento), ou uma
            lista de dicts {tema: valor}.
        metodo (str): Método de correlação do resumo/matriz retornados: 'pearson' (linear)
            ou 'spearman' (por postos, mais robusto a dados esparsos/assimétricos como
            TF-IDF com muitos zeros). Padrão: 'pearson'.
        caminho_heatmap (str | None): Se informado, salva em .html um heatmap interativo
            comparando as matrizes de correlação de Pearson e Spearman lado a lado.

    Returns:
        (pandas.DataFrame, pandas.DataFrame): (matriz_correlacao, resumo)
          - matriz_correlacao: correlação tema x tema (pelo `metodo` escolhido).
          - resumo: por tema, o mais/menos correlacionado, o isolamento e o número de
            documentos com valor > 0.
    """
    # 1. Monta a matriz documentos x temas
    if isinstance(dados, dict):
        df = pd.DataFrame.from_dict(dados, orient="index")
    else:
        df = pd.DataFrame(list(dados))

    total_docs = len(df)

    # 2. Matriz de correlação entre os temas
    corr = df.corr(method=metodo)

    # 3. Resumo por tema
    n_positivos = (df > 0).sum()  # documentos com valor > 0 por tema
    linhas = []
    for tema in corr.columns:
        outros = corr[tema].drop(tema)
        linhas.append({
            "tema": tema,
            "n_docs_positivos": int(n_positivos[tema]),
            "pct_docs_positivos": f"{n_positivos[tema] / total_docs * 100:.1f}%",
            "mais_correlacionado": outros.idxmax(),
            "corr_max": round(outros.max(), 3),
            "mais_anticorrelacionado": outros.idxmin(),
            "corr_min": round(outros.min(), 3),
            "isolamento": round(outros.abs().mean(), 3),
        })
    resumo = pd.DataFrame(linhas).sort_values("isolamento").reset_index(drop=True)

    # 4. Heatmap comparando Pearson e Spearman lado a lado
    if caminho_heatmap:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        corr_pearson = df.corr(method="pearson")
        corr_spearman = df.corr(method="spearman")

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Pearson", "Spearman"),
                             horizontal_spacing=0.15)
        for col, matriz in enumerate([corr_pearson, corr_spearman], start=1):
            fig.add_trace(
                go.Heatmap(
                    z=matriz.values,
                    x=list(matriz.columns),
                    y=list(matriz.index),
                    coloraxis="coloraxis",
                    texttemplate="%{z:.2f}",
                    textfont={"size": 9},
                ),
                row=1, col=col,
            )
        fig.update_layout(
            coloraxis=dict(colorscale="RdBu_r", cmin=-1, cmax=1,
                           colorbar=dict(title="Correlação")),
            title="Correlação entre temas — Pearson vs Spearman",
        )
        # Eixo Y invertido para a diagonal ir do topo-esquerda ao fundo-direita
        fig.update_yaxes(autorange="reversed")
        fig.write_html(caminho_heatmap, include_plotlyjs="cdn")
        print(f"Heatmap (Pearson vs Spearman) salvo em: {caminho_heatmap}")

    return corr, resumo
