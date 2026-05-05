from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig


def obter_transcricao(video):
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

        Raises:
            Exception: Captura exceções genéricas da API, registrando-as no log
                de depuração e exibindo uma mensagem no console.

        Example:
            >>> video = {"id_vd": "abc12345", "titulo": "Exemplo"}
            >>> resultado = obter_transcricao(video)
            >>> if resultado: print(resultado['transcricao'])
        """
    id_vd = video.get("id_vd")

    ytt_api = YouTubeTranscriptApi(
    proxy_config=WebshareProxyConfig(
        proxy_username="xxxxxxx",
        proxy_password="xxxxxxxx",
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
        logger.debug(f"❌ Erro ao baixar a transcrição: {e}")
        return None



def op6_buscar_transcricoes_de_videos_selecionados():
    """
    Busca transcrições de vídeos previamente selecionados (armazenados em `videos_selecionados`)
    e as insere no banco de dados MySQL.

    A função itera sobre os vídeos, baixa a transcrição para cada um (se disponível),
    armazena os resultados em uma lista e depois insere na tabela `transcricao_v02`.
    """
    import webbrowser
    from funcoes_entrada._buscador_transcricoes import obter_transcricao
    global videos_selecionados, transcricoes_selecionadas
    if not videos_selecionados:
        print("Nenhum vídeo selecionado.")
        return

    # webbrowser.open_new_tab("https://dashboard.webshare.io/dashboard")

    # 1 - Seleciona as transcrições já existentes dos vídeos selecionados
    transcricoes_ja_existentes = seleciona_transcricoes_por_video(videos_selecionados)

    ###
    print(f'Número de transcricoes já existentes {len(transcricoes_ja_existentes)}.')
    ###

    # 1.1 - Adiciona as transcrições já existentes às transcrilções selecionadas sem repetições
    adiciona_a_selecao_de_transcricoes(transcricoes_ja_existentes)
    print("---------")
    logger.info(f"Adicionado à tabela 'transcricoes selecionadas' = {len(transcricoes_ja_existentes)}")

    # 2 - Verifica quais transcrições ainda não foram baixadas
    lista_nao_verificada = videos_selecionados
    lista_verificada = transcricoes_selecionadas
    # 2.1 lista de dicts com os vídeos sem transcrição
    lista_videos_sem_transcricao = filtrar_itens_unicos(lista_nao_verificada, lista_verificada, "id_vd", "id_video")

    # 3 - Baixa as transcrições dos vídeos sem transcricao
    print(f"Transcrições selecionadas: {len(transcricoes_selecionadas)}")
    print(f"Iniciando busca de transcrições para {len(lista_videos_sem_transcricao)} vídeos...\n")

    # 3.1 função que encapsula o processamento de cada vídeos no multithread, ela que obtém as transcrições
    def processar_video(video_info):
        """
        Função auxiliar para processar um único vídeo.
        Args:
            video_info (tuple):
        """
        i, video, total = video_info
        id_video = video.get("id_vd")
        titulo = video.get('titulo', '')

        print(f"[{i + 1}/{total}] Iniciando: {titulo}")

        try:
            # Consome dict, retorna dict + transcricao
            resultado = obter_transcricao(video)
            if resultado:
                # Sobe para relacional
                sobe_transcricoes_para_mysql([resultado])
                # Sobe para não-relacional
                sobe_transcricoes_para_db([resultado])
                # Adicona a transcrição à seleção
                adiciona_a_selecao_de_transcricoes([resultado])

                return f"✅ Sucesso: {titulo}"
            else:
                return f"❌ Falha: {id_video} (Sem transcrição)"
        except Exception as e:
            return f"⚠️ Erro no vídeo {id_video}: {str(e)}"

    # --- Configuração do Multi-threading ---

    # 3.2 Preparamos os dados para passar para as threads
    total_videos = len(lista_videos_sem_transcricao)
    dados_para_processar = [(i, v, total_videos) for i, v in enumerate(lista_videos_sem_transcricao)]

    # O limite de 50 conexões simultâneas é definido em max_workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(processar_video, d): d for d in dados_para_processar}

        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                print(res)  # Você vê o sucesso/erro em tempo real
            except Exception as e:
                print(f"Erro crítico na thread: {e}")

    print("\n--- Processo concluído ---")
    voltar_ao_menu_principal()