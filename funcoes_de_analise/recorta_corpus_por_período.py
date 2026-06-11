from datetime import datetime

def recorta_corpus_por_periodo(data_inicial, data_final, corpus, campo_data = "data_criacao"):
    corpus_recortado = []
    data_inicial = datetime.fromisoformat(data_inicial)
    data_final = datetime.fromisoformat(data_final)

    for video in corpus:
        data_do_video = video[campo_data]
        data_do_video = datetime.fromisoformat(data_do_video)
        if data_do_video >= data_inicial and data_do_video <= data_final:
            corpus_recortado.append(video)
    return corpus_recortado