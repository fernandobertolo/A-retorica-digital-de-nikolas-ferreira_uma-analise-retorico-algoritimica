# 2 - CAPTURA VIDEOS
# Conjunto de funções que tem o objetivos de baixar N vídeos de um canal do youtube
from datetime import datetime
import time
from _controle_chaves_API import chave_api


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
            snippet = item.get("snippet", {})
            statistics = item.get("statistics", {})

            data_criacao_iso = snippet.get("publishedAt")
            data_criacao = datetime.strptime(data_criacao_iso, "%Y-%m-%dT%H:%M:%SZ").date() if data_criacao_iso else None

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
                "transcricao": ""
            })

        time.sleep(0.5)  # Para evitar limites de cota

    return dados


# Função principal
def coleta_videos_de_um_canal(channel_id, limite=50):
    youtube = chave_api()
    playlist_id = get_uploads_playlist_id(youtube, channel_id)
    video_ids = get_video_ids_from_playlist(youtube, playlist_id, max_videos=limite)
    dados = get_video_details(youtube, video_ids)
    return dados
