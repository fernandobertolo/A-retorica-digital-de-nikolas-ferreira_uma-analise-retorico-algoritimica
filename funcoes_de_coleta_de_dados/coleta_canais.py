# 1 - CAPTURA CANAIS
# Conjunto de funções que tem o objtivo de baixar dados de canais do youtube
import datetime
from funcoes_de_coleta_de_dados._controle_chaves_API import chave_api
from googleapiclient.errors import HttpError



def coleta_canais_por_handle(handle, maxResults=1, ordenador="relevance"):
    """
    Busca canais no YouTube por handle ou nome do canal e retorna informações básicas e estatísticas.

    :param handle (str): Pode ser uma arroba (@) ou o nome do canal.
    :param maxResults (int): Número máximo de canais a buscar.
    :param ordenador (str): Em qual ordem a API vai buscar os vídeos.
    :return: lista de dicionários com informações dos canais (com tipos corretos) ou None em caso de erro.
    """
    try:
        youtube = chave_api()
    except Exception as e:
        print(f"Erro ao autenticar a API: {e}")
        return None

    try:
        resposta = youtube.search().list(
            part="snippet",
            q=handle,
            type="channel",
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
        print(f"Nenhum canal encontrado para: {handle}")
        return []

    canais = []

    def to_int_safe(val):
        try:
            return int(val)
        except:
            return 0

    for item in resposta["items"]:
        if item['id']['kind'] == 'youtube#channel':
            canal_id = item['id']['channelId']

            # Valores básicos
            titulo = item['snippet']['title']
            descricao = item['snippet']['description']
            data_raw = item['snippet']['publishedAt']

            try:
                data_criacao = datetime.fromisoformat(data_raw.replace("Z", "+00:00")).date()
            except Exception:
                data_criacao = None  # Em caso de erro, pode usar None

            # Valores default
            seguidores = views = n_videos = 0
            autor = "N/A"

            # Busca estatísticas
            try:
                resposta_stats = youtube.channels().list(
                    part="snippet,statistics",
                    id=canal_id
                ).execute()
                dados = resposta_stats['items'][0]

                stats = dados.get("statistics", {})
                snippet = dados.get("snippet", {})

                seguidores = to_int_safe(stats.get("subscriberCount"))
                views = to_int_safe(stats.get("viewCount"))
                n_videos = to_int_safe(stats.get("videoCount"))
                autor = snippet.get("customUrl") or snippet.get("handle") or "N/A"

            except Exception as e:
                print(f"[Erro ao buscar estatísticas para canal {canal_id}] {type(e).__name__}: {e}")

            canais.append({
                "id_ch": canal_id,               # str
                "titulo": titulo,                # str
                "descricao": descricao,          # str
                "data_criacao": data_criacao,    # datetime.date
                "seguidores": seguidores,        # int
                "views": views,                  # int
                "n_videos": n_videos,            # int
                "autor": autor                   # str
            })

    return canais

def coleta_canal_por_id(id_cn):
    """
    Busca informações de um canal do YouTube a partir do seu ID.

    :param id_cn (str): ID do canal no YouTube (ex.: 'UC_x5XG1OV2P6uZZ5FSM9Ttw').
    :return: Uma lista com um dicionário contendo as informações do canal com tipos corretos, ou None em caso de erro.
    """
    try:
        youtube = chave_api()
    except Exception as e:
        print(f"Erro ao autenticar a API: {e}")
        return None

    try:
        resposta = youtube.channels().list(
            part="snippet,statistics",
            id=id_cn
        ).execute()
    except HttpError as e:
        print(f"[YouTube API] Erro HTTP: {e.resp.status} - {getattr(e, 'error_details', e)}")
        return None
    except Exception as e:
        print(f"[Erro inesperado] {type(e).__name__}: {e}")
        return None

    if not resposta.get("items"):
        print(f"Nenhum canal encontrado para o ID: {id_cn}")
        return None

    def to_int_safe(val):
        try:
            return int(val)
        except:
            return 0

    try:
        item = resposta["items"][0]
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})

        data_raw = snippet.get("publishedAt", "")
        try:
            data_criacao = datetime.fromisoformat(data_raw.replace("Z", "+00:00")).date()
        except Exception:
            data_criacao = None

        canal = {
            "id_ch": item.get("id", ""),                            # str
            "titulo": snippet.get("title", ""),                     # str
            "descricao": snippet.get("description", ""),           # str
            "data_criacao": data_criacao,                           # datetime.date
            "autor": snippet.get("customUrl") or snippet.get("handle") or "N/A",  # str
            "seguidores": to_int_safe(stats.get("subscriberCount")),  # int
            "views": to_int_safe(stats.get("viewCount")),             # int
            "n_videos": to_int_safe(stats.get("videoCount"))          # int
        }

        return [canal]

    except Exception as e:
        print(f"[Erro ao processar os dados do canal] {type(e).__name__}: {e}")
        return None