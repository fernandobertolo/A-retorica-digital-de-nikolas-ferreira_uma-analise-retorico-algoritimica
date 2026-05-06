# 3 - CAPTURA TRANSCRICOES
import datetime

import concurrent, os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig

# Carrega as variáveis do arquivo .env
load_dotenv()


def coleta_transcricoes(lista_de_videos):
    """
    Coleta várias transcrições de uma vez usando processamento paralelo
    :param lista_de_videos: Lista de dicionários dos vídeos gerado pela função coleta_videos de coleta_videos
    :return: Uma lista de dicionários o índice das transcrições
    """


    # ETAPA 1 - VERIFICAçÃO
    # Verifica quais vídeos dos dicionários possuem transcrição
    # Essa parte é necessária por que, muitas vezes, por diversos motivos, a função não consegue
    # coletar todas as transcrições em uma tentativa só

    videos_sem_transcricoes = [v for v in lista_de_videos if not v["transcricao"]]
    videos_com_transcricoes = [v for v in lista_de_videos if v["transcricao"]]
    print(f"Verificação das transcrições:\n{len(videos_sem_transcricoes)} - Sem\n{len(videos_com_transcricoes)} - Com")


    # O limite de 50 conexões simultâneas é definido em max_workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(coleta_transcricao, v): v for v in videos_sem_transcricoes}

        i = 0

        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                videos_com_transcricoes.append(res)
                i+=1
                print(f'Transcrição {i} de {len(lista_de_videos)}')
            except Exception as e:
                print(f"Erro crítico na thread: {e}")

    print("\n--- Processo concluído ---")
    return videos_com_transcricoes



def coleta_transcricao(video):
    """
       Recupera a transcrição de um vídeo do YouTube via API e a anexa ao dicionário do vídeo.

        A função utiliza a biblioteca `youtube_transcript_api` com configuração de proxy
        Webshare para contornar restrições de IP. Ela busca legendas preferencialmente
        em português ('pt') ou inglês ('en').

        Args:
            video (dict): Dicionário contendo os dados do vídeo.
                Deve obrigatoriamente possuir a chave 'id_vd' (ID do YouTube).

        Returns:
            dict: O dicionário original atualizado com a chave 'transcricao' contendo
                o texto completo concatenado, caso a operação seja bem-sucedida.
            None: Retorna None caso ocorra qualquer erro na recuperação da transcrição
                (ex: legendas desativadas, erro de conexão ou proxy).
        """
    id_vd = video.get("id_vd")
    PROXY_USERNAME =  os.getenv("PROXY_USERNAME")
    PROXY_PASSWORD =  os.getenv("PROXY_PASSWORD")

    ytt_api = YouTubeTranscriptApi(
    proxy_config=WebshareProxyConfig(
        proxy_username=PROXY_USERNAME,
        proxy_password=PROXY_PASSWORD,
    )
)
    # retrieve (recupera) the available transcripts
    try:
        transcript = ytt_api.fetch(id_vd, languages=['pt', 'en'])
        str = ""
        for trecho in transcript:
            str += trecho.text + " "
        video["transcricao"] = str
        return video
    except Exception as e:
        print(f"❌ Erro ao baixar a transcrição: {e}")
        return video