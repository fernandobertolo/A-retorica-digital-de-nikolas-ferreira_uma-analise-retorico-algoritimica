import os
import datetime
import textwrap
import pprint  # Para imprimir o resultado de forma legível
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import plotly.express as px
import plotly.io as pio
from umap import UMAP

# Define o tema do Plotly para funcionar bem em diferentes ambientes (como notebooks)
pio.templates.default = "plotly_white"


def juntar_dados_videos(
        lista_metadados: list[dict],
        lista_transcricoes: list[dict],
        chave_id_meta: str = 'id_vd',
        chave_id_trans: str = 'id_video',
        manter_sem_correspondencia: bool = True
) -> list[dict]:
    """
    Junta duas listas de dicionários (metadados e transcrições de vídeos)
    em uma única lista, combinando as informações pelo ID do vídeo.

    A função utiliza um mapa de busca para realizar a junção de forma eficiente (O(N+M)),
    o que é ideal para listas grandes.

    Args:
        lista_metadados (list[dict]): A lista principal com os metadados dos vídeos.
        lista_transcricoes (list[dict]): A lista com as transcrições e seus IDs.
        chave_id_meta (str): O nome da chave de ID na lista de metadados.
                             Default: 'id_vd'.
        chave_id_trans (str): O nome da chave de ID na lista de transcrições.
                              Default: 'id_video'.
        manter_sem_correspondencia (bool): Se True, vídeos da lista de metadados que
                                           não têm uma transcrição correspondente serão
                                           incluídos no resultado final (com o campo
                                           'transcricao' como None). Se False, serão omitidos.
                                           Default: True.

    Returns:
        list[dict]: Uma nova lista de dicionários, onde cada dicionário contém
                    os metadados originais mais a chave 'transcricao' correspondente.

    Exemplo de saída:
    [{'coments': 0,
        'data_criacao': datetime.date(2025, 5, 14),
        'id_ch': 'UCh91hhxrIUkPldxPy70a3sw',
        'id_vd': 'Zzzrk3jm_dE',
        'likes': 1,
        'titulo': 'Jornalista detona a Janja',
        'transcricao': 'A primeira dama deveria ter uma postura mais reservada em '
                 'viagens internacionais...',
    'views': 16}]
    """
    print("Iniciando a junção dos dados...")

    # Passo 1: Criar um mapa de busca (dicionário) a partir da lista de transcrições.
    # Isso torna a busca pela transcrição extremamente rápida (complexidade O(1) em média).
    # A chave do mapa será o ID do vídeo e o valor será a transcrição.
    mapa_transcricoes = {
        item[chave_id_trans]: item['transcricao'] for item in lista_transcricoes
    }
    print(f"Mapa de busca com {len(mapa_transcricoes)} transcrições criado.")

    lista_final = []
    videos_com_transcricao = 0

    # Passo 2: Iterar pela lista de metadados e juntar com as informações do mapa.
    for video_meta in lista_metadados:
        # Pega o ID do vídeo da lista de metadados
        id_do_video = video_meta.get(chave_id_meta)

        # Busca a transcrição no mapa usando o ID.
        # O método .get() é seguro, pois retorna None se o ID não for encontrado.
        transcricao = mapa_transcricoes.get(id_do_video)

        if transcricao is not None:
            # Se encontrou a transcrição, cria uma cópia do dicionário de metadados
            # para não alterar a lista original (boa prática).
            video_completo = video_meta.copy()
            video_completo['transcricao'] = transcricao
            lista_final.append(video_completo)
            videos_com_transcricao += 1
        elif manter_sem_correspondencia:
            # Se não encontrou, mas queremos manter o vídeo mesmo assim.
            video_completo = video_meta.copy()
            video_completo['transcricao'] = id_do_video  # Adiciona 'transcricao' como None
            lista_final.append(video_completo)

    print(f"Junção concluída. {len(lista_final)} vídeos na lista final.")
    print(f"({videos_com_transcricao} vídeos foram enriquecidos com transcrição).")
    return lista_final


def analise_bertopic_videos(
        dados_videos: list[dict],
        chave_titulo: str = 'titulo',
        chave_texto: str = 'transcricao',
        modelo_embedding: str = 'all-MiniLM-L6-v2',
        min_topic_size: int = 3,
        language: str = 'portuguese'
):
    """
    Realiza uma análise de tópicos com BERTopic em uma lista de vídeos.

    A função extrai transcrições de uma lista de dicionários, treina um modelo
    BERTopic para encontrar e agrupar tópicos, e exibe um gráfico interativo
    da similaridade dos documentos (vídeos), usando os títulos dos vídeos como rótulos.

    Args:
        dados_videos (list[dict]): Uma lista de dicionários, onde cada dicionário
                                   representa um vídeo e contém pelo menos um título e uma transcrição.
        chave_titulo (str): A chave no dicionário que corresponde ao título do vídeo.
                            Default: 'nome_do_video'.
        chave_texto (str): A chave no dicionário que corresponde à transcrição.
                           Default: 'transcricao'.
        modelo_embedding (str): O nome do modelo de sentence-transformer a ser usado
                                para criar os embeddings dos textos.
                                Default: 'all-MiniLM-L6-v2'.
        min_topic_size (int): O número mínimo de documentos para formar um tópico.
                              Aumentar este valor reduz o número de tópicos.
                              Default: 3.
        language (str): O idioma dos textos para otimizar o processamento (ex: remoção de stopwords).
                        Use 'multilingual' para múltiplos idiomas.
                        Default: 'portuguese'.

    Returns:
        bertopic.BERTopic: O objeto do modelo BERTopic treinado, que pode ser usado
                           para análises futuras.
    """
    print("1. Preparando os dados...")
    # Extrai as transcrições e os títulos das estruturas de dados de entrada
    try:
        textos = [video[chave_texto] for video in dados_videos]
        titulos = [video[chave_titulo] for video in dados_videos]
    except KeyError as e:
        print(f"Erro: A chave {e} não foi encontrada nos dicionários. Verifique os nomes das chaves.")
        return None

    if not textos:
        print("Erro: A lista de textos para análise está vazia.")
        return None

    print(f"2. Inicializando o modelo BERTopic com o embedding '{modelo_embedding}'...")
    # Para consistência, usamos um modelo de embedding específico.
    # 'all-MiniLM-L6-v2' é rápido e tem boa performance.
    embedding_model = SentenceTransformer(modelo_embedding)

    # Cria a instância do modelo BERTopic
    # - language='portuguese': Usa uma lista de stopwords em português.
    # - min_topic_size: Controla a granularidade dos tópicos.
    # - calculate_probabilities=True: Necessário para algumas visualizações.
    # - verbose=True: Mostra o progresso do treinamento.
    topic_model = BERTopic(
        embedding_model=embedding_model,
        language=language,
        min_topic_size=min_topic_size,
        calculate_probabilities=True,
        verbose=True
    )

    print("\n3. Treinando o modelo e agrupando os textos... (Isso pode levar alguns minutos)")
    # Treina o modelo com as transcrições
    topics, probabilities = topic_model.fit_transform(textos)

    print("\n4. Gerando o gráfico de similaridade interativo...")
    # Gera a visualização dos documentos.
    # Cada ponto é um vídeo, posicionado de acordo com sua similaridade semântica.
    # A cor indica o tópico atribuído.
    # O parâmetro 'custom_labels' é crucial para mostrar os títulos dos vídeos.
    figura_interativa = topic_model.visualize_documents(
        textos,
        custom_labels=titulos,
        title="<b>Similaridade Semântica entre Vídeos</b>"
    )

    # Exibe o gráfico
    figura_interativa.show()

    print("\nAnálise concluída!")

    # Retorna o modelo treinado para que possa ser inspecionado posteriormente
    return topic_model



def clusterizacao_BERT(
        sentencas: list[str],
        caminho_html: str = 'clusterizacao_bert.html',
        modelo_embedding: str = 'all-MiniLM-L6-v2',
        min_topic_size: int = 15,
        nr_topics=None,
        tamanho_ponto: int = 8,
        largura_hover: int = 80,
        dimensoes: int = 2,
        language: str = 'portuguese'
):
    """
    Agrupa uma lista de sentenças em tópicos semânticos usando BERTopic.

    A função gera embeddings para cada sentença, treina um modelo BERTopic para
    descobrir e agrupar os tópicos e salva um gráfico interativo da similaridade
    semântica entre os documentos em um arquivo HTML.

    Args:
        sentencas (list[str]): Lista de textos a serem clusterizados.
        caminho_html (str): Caminho do arquivo HTML interativo a ser salvo.
            Default: 'clusterizacao_bert.html'.
        modelo_embedding (str): Nome do modelo sentence-transformer usado para
            gerar os embeddings. Default: 'all-MiniLM-L6-v2'.
        min_topic_size (int): Número mínimo de documentos para formar um tópico.
            Valores maiores reduzem o número de tópicos. Default: 15.
        nr_topics (int | str | None): Reduz o número de tópicos após o
            treinamento. Use um inteiro para um alvo exato ou 'auto' para
            redução automática. None mantém todos os tópicos. Default: None.
        tamanho_ponto (int): Tamanho dos marcadores (pontos) no gráfico.
            Default: 8.
        largura_hover (int): Largura máxima (em caracteres) de cada linha do
            texto exibido no mouse over, antes de quebrar para a linha seguinte.
            Default: 80.
        dimensoes (int): Dimensionalidade da visualização: 2 para o gráfico 2D
            padrão do BERTopic ou 3 para um scatter 3D interativo (UMAP 3D +
            Plotly). Default: 2.
        language (str): Idioma dos textos para otimizar o processamento (ex.:
            remoção de stopwords). Use 'multilingual' para múltiplos idiomas.
            Default: 'portuguese'.

    Returns:
        bertopic.BERTopic | None: O modelo BERTopic treinado, ou None se a lista
            de sentenças estiver vazia.
    """
    if not sentencas:
        print("Erro: a lista de sentenças para análise está vazia.")
        return None

    print(f"1. Inicializando o modelo BERTopic com o embedding '{modelo_embedding}'...")
    embedding_model = SentenceTransformer(modelo_embedding)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        language=language,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        calculate_probabilities=True,
        verbose=True
    )

    # Gera os embeddings uma vez e os reaproveita no treino e na projeção 3D.
    print("\n2. Gerando embeddings...")
    embeddings = embedding_model.encode(sentencas, show_progress_bar=True)

    print("\n3. Treinando o modelo e agrupando os textos... (Isso pode levar alguns minutos)")
    topics, probabilities = topic_model.fit_transform(sentencas, embeddings)

    # Quebra o texto do hover em várias linhas (Plotly usa <br>), tornando
    # textos longos legíveis em vez de uma única linha esticada.
    def _quebrar_hover(texto):
        if not texto:
            return texto
        return "<br>".join(textwrap.wrap(str(texto), width=largura_hover))

    print("\n4. Gerando o gráfico de similaridade interativo...")
    if dimensoes == 3:
        # Projeta os embeddings em 3 dimensões e monta um scatter 3D.
        # Espelha os parametros que o BERTopic usa no visualize_documents (2D):
        # n_neighbors=10 e min_dist=0.0 compactam cada cluster e afastam os grupos,
        # evitando que os topicos se fundam visualmente (o que fazia o 3D parecer ter menos clusters).
        reduzido = UMAP(
            n_components=3,
            n_neighbors=10,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        ).fit_transform(embeddings)
        mapa_nomes = topic_model.get_topic_info().set_index("Topic")["Name"].to_dict()

        df = pd.DataFrame({
            "x": reduzido[:, 0],
            "y": reduzido[:, 1],
            "z": reduzido[:, 2],
            "topico": [f"{t}: {mapa_nomes.get(t, '')}" for t in topics],
            "texto": [_quebrar_hover(s) for s in sentencas],
        })

        # Concatena paletas qualitativas (26 + 24 + 24 = 74 cores distintas) para
        # evitar repeticao de cor mesmo com muitos topicos.
        paleta = (px.colors.qualitative.Alphabet
                  + px.colors.qualitative.Dark24
                  + px.colors.qualitative.Light24)

        figura_interativa = px.scatter_3d(
            df, x="x", y="y", z="z", color="topico",
            custom_data=["texto"],
            color_discrete_sequence=paleta,
            title="<b>Similaridade Semântica entre Sentenças (3D)</b>",
        )
        figura_interativa.update_traces(
            marker=dict(size=tamanho_ponto),
            hovertemplate="%{customdata[0]}<extra></extra>",
        )
    else:
        # hide_annotations=True remove os rótulos fixos sobre o gráfico (poluição visual);
        # hide_document_hover=False mantém o texto da sentença ao passar o mouse.
        figura_interativa = topic_model.visualize_documents(
            sentencas,
            embeddings=embeddings,
            hide_annotations=True,
            hide_document_hover=False,
            title="<b>Similaridade Semântica entre Sentenças</b>"
        )

        # Aumenta o tamanho dos pontos sem afetar os rótulos de texto.
        figura_interativa.update_traces(marker=dict(size=tamanho_ponto),
                                        selector=dict(mode='markers'))

        for trace in figura_interativa.data:
            hover = getattr(trace, "hovertext", None)
            if hover is None:
                continue
            if isinstance(hover, str):
                trace.hovertext = _quebrar_hover(hover)
            else:
                # hovertext costuma ser um array (NumPy/lista) com um texto por ponto.
                trace.hovertext = [_quebrar_hover(t) for t in hover]

    figura_interativa.write_html(caminho_html)
    print(f"   Gráfico salvo em: {os.path.abspath(caminho_html)}")

    print("\nAnálise concluída!")
    return topic_model