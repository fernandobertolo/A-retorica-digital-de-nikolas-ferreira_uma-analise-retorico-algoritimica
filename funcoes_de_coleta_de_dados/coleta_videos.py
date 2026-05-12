# 2 - CAPTURA VIDEOS
# Conjunto de funções que tem o objetivos de baixar N vídeos de um canal do youtube
import time, re
from funcoes_de_coleta_de_dados._controle_chaves_API import chave_api
from datetime import datetime
from googleapiclient.errors import HttpError



# Funções assessoras
def get_uploads_playlist_id(youtube, channel_id):
    """Retorna o ID da playlist de uploads do canal"""
    resposta = youtube.channels().list(
        part="contentDetails",
        id=channel_id
    ).execute()

    items = resposta.get("items", [])
    if not items:
        raise Exception("Canal não encontrado ou sem vídeos.")

    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

def get_video_ids_from_playlist(youtube, playlist_id, max_videos=50):
    """Coleta todos os IDs dos vídeos da playlist"""
    video_ids = []
    next_page_token = None

    while True:
        resposta = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        ).execute()

        for item in resposta["items"]:
            video_ids.append(item["contentDetails"]["videoId"])
            if len(video_ids) >= max_videos:
                return video_ids

        next_page_token = resposta.get("nextPageToken")
        if not next_page_token:
            break

    return video_ids

def get_video_details(youtube, video_ids):
    """Pega os detalhes dos vídeos a partir dos IDs, convertendo tipos"""
    dados = []

    for i in range(0, len(video_ids), 50):
        ids = video_ids[i:i + 50]
        resposta = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(ids)
        ).execute()

        for item in resposta["items"]:
            content_details = item.get("contentDetails", {})  # Correção: campo correto
            snippet = item.get("snippet", {})
            statistics = item.get("statistics", {})
            data_criacao_iso = snippet.get("publishedAt")
            data_criacao = datetime.strptime(data_criacao_iso, "%Y-%m-%dT%H:%M:%SZ").date() if data_criacao_iso else None

            # Coleta da duração original e conversão
            duracao_raw = content_details.get("duration", "")
            duracao_segundos = converter_duracao_para_segundos(duracao_raw)

            views = int(statistics.get("viewCount", 0))
            likes = int(statistics.get("likeCount", 0))
            coments = int(statistics.get("commentCount", 0))


            dados.append({
                "id_vd": item["id"],
                "id_ch": snippet.get("channelId", ""),
                "titulo": snippet.get("title", ""),
                "descricao": snippet.get("description", ""),
                "data_criacao": data_criacao,
                "views": views,
                "likes": likes,
                "coments": coments,
                "duracao": duracao_segundos,
                "transcricao": ""
            })

        time.sleep(0.5)  # Para evitar limites de cota

    return dados

def converter_duracao_para_segundos(duracao_iso):
    """Converte o formato ISO 8601 de duração do YouTube para total de segundos."""
    if not duracao_iso:
        return 0
    # Regex para extrair horas, minutos e segundos
    pattern = re.compile(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?')
    match = pattern.match(duracao_iso)
    if not match:
        return 0

    horas = int(match.group(1) or 0)
    minutos = int(match.group(2) or 0)
    segundos = int(match.group(3) or 0)

    return horas * 3600 + minutos * 60 + segundos


# Funções principais
def coleta_videos_de_um_canal(channel_id, limite=50):
    youtube = chave_api()
    playlist_id = get_uploads_playlist_id(youtube, channel_id)
    video_ids = get_video_ids_from_playlist(youtube, playlist_id, max_videos=limite)
    dados = get_video_details(youtube, video_ids)
    return dados

def coleta_video_por_handle(handle, maxResults=1, ordenador="relevance"):
    """
    Realiza busca de vídeos no YouTube via Data API v3 com enriquecimento de estatísticas.

    O processo ocorre em duas etapas para otimização de cota:
    1. 'search.list': Localiza vídeos por termo/handle (Custo: 100).
    2. 'videos.list': Recupera métricas (views, likes, comments) em lote (Custo: 1).

    Args:
        handle (str): Termo de pesquisa, identificador do canal (@handle) ou palavras-chave.
        maxResults (int, opcional): Quantidade de resultados (1 a 50). Padrão é 1.
        ordenador (str, opcional): Critério de ordenação da API (ex: 'date', 'viewCount',
            'relevance'). Padrão é 'relevance'.

    Returns:
        list[dict]: Lista de dicionários contendo os metadados e estatísticas.
            Retorna uma lista vazia ou None em caso de erro crítico.

        Exemplo de um dicionário de vídeo no retorno:
        {
            "id_vd": "zSnt_m7XyIs",
            "id_ch": "UC_x5XG1OV2P6uqz76xPyjZA",
            "titulo": "Análise de Redes Digitais em 2026",
            "canal": "Canal de Inteligência Online",
            "descricao": "Discussão sobre o impacto das novas redes sociais...",
            "data_criacao": datetime.date(2026, 4, 10),
            "views": 15420,
            "likes": 890,
            "coments": 45
        }

    Custo de Cota:
        - Execução padrão: 101 unidades (100 de busca + 1 de estatísticas em lote).
        - O custo de 101 é fixo para buscas que retornam entre 1 e 50 vídeos.

    Raises:
        HttpError: Erros específicos da API do YouTube são tratados internamente,
            registrando o log e retornando o estado atual dos dados.
    """
    try:
        youtube = chave_api()
    except Exception as e:
        print(f"Erro ao autenticar a API: {e}")
        return None, 0

    # 1. Busca os vídeos (Custo: 100 unidades)
    try:
        resposta = youtube.search().list(
            part="snippet",
            q=handle,
            type="video",
            maxResults=maxResults,
            order=ordenador
        ).execute()


    except HttpError as e:
        print(f"[YouTube API] Erro HTTP: {e.resp.status} - {getattr(e, 'error_details', e)}")
        return None
    except Exception as e:
        print(f"[Erro inesperado] {type(e).__name__}: {e}")
        return None

    if not resposta.get("items"):
        print(f"Nenhum vídeo encontrado para: {handle}")
        return []

    videos = []

    # Processamento inicial dos dados da busca
    try:
        for video in resposta["items"]:
            if video['id']['kind'] == 'youtube#video':
                data_str = video['snippet']['publishTime']
                try:
                    data = datetime.fromisoformat(data_str.replace("Z", "+00:00")).date()
                except Exception:
                    data = data_str

                videos.append({
                    "id_vd": video["id"]['videoId'],
                    "id_ch": video['snippet']["channelId"],
                    "titulo": video['snippet']["title"],
                    "canal": video['snippet']['channelTitle'],
                    "descricao": video['snippet']['description'],
                    "data_criacao": data,
                    # Inicializa métricas com 0
                    "views": 0, "likes": 0, "coments": 0
                })
    except Exception as e:
        print(f"[Erro ao processar dados] {e}")

    # 2. Busca estatísticas em LOTE (Batch) (Custo: 1 unidade para cada 50 vídeos)
    # Isso é muito mais barato do que fazer um loop chamando a API um por um.
    if videos:
        ids_videos = [v["id_vd"] for v in videos]
        ids_string = ",".join(ids_videos)  # "id1,id2,id3..."

        try:
            resposta_estat = youtube.videos().list(
                part="statistics",
                id=ids_string
            ).execute()

            # Cria um dicionário para acesso rápido: {'id_video': {stats}}
            stats_map = {item['id']: item.get('statistics', {}) for item in resposta_estat.get('items', [])}

            def to_int(val):
                try:
                    return int(val)
                except:
                    return 0

            # Atualiza a lista original com as estatísticas
            for video in videos:
                stats = stats_map.get(video["id_vd"], {})
                video["views"] = to_int(stats.get('viewCount'))
                video["likes"] = to_int(stats.get('likeCount'))
                video["coments"] = to_int(stats.get('commentCount'))

        except HttpError as e:
            print(f"Erro ao buscar estatísticas: {e}")
            # Retorna o que conseguiu até agora, mas avisa do erro
            return videos
    return videos
