import pickle, os
from dotenv import load_dotenv
from googleapiclient.discovery import build


# Carrega as variáveis do arquivo .env para o sistema
load_dotenv()

# Dicionário com as chaves de API
dict_api_keys = {
    "key_joao": os.getenv("KEY_JOAO"),
    "key_camisa": os.getenv("KEY_CAMISA"),
    "key_bertolo": os.getenv("KEY_BERTOLO"),
    "key_nathalia": os.getenv("KEY_NATHALIA"),
    "key_sandra": os.getenv("KEY_SANDRA"),
    "key_sandra2": os.getenv("KEY_SANDRA2"),
    "key_xaxin": os.getenv("KEY_XAXIN"),
    "key_iaemail961": os.getenv("KEY_IAEMAIL961"),
    "key_amadeus": os.getenv("KEY_AMADEUS"),
    "key_esther": os.getenv("KEY_ESTHER")
}


# Ordem de verificação das chaves
lista_indices_keys = list(dict_api_keys.keys())

CAMINHO_ARQUIVO = "memoria_das_chaves.pk1"


def resgata_chave_da_vez():
    """Resgata o índice da chave da vez a partir de um arquivo pickle"""
    if os.path.exists(CAMINHO_ARQUIVO):
        try:
            with open(CAMINHO_ARQUIVO, "rb") as arquivo:
                return pickle.load(arquivo)
        except Exception as e:
            print(f"Erro ao ler arquivo de memória: {e}")
    return 0


def pula_para_proxima_chave():
    """Avança para a próxima chave na lista circular"""
    print("Pulando para a próxima chave API...")
    chave_atual = resgata_chave_da_vez()
    chave_atual = (chave_atual + 1) % len(lista_indices_keys)

    try:
        with open(CAMINHO_ARQUIVO, "wb") as arquivo:
            pickle.dump(chave_atual, arquivo)
    except Exception as e:
        print(f"Erro ao salvar índice da chave: {e}")


def chave_api():
    """
    Retorna um objeto da API do YouTube com uma chave funcional.
    Tenta criar diretamente o serviço com a chave atual. Em caso de falha,
    avança para a próxima chave e repete o processo.

    Returns:
        googleapiclient.discovery.Resource: Instância da API do YouTube.

    Raises:
        RuntimeError: Se nenhuma chave conseguir criar o serviço da API.
    """
    tentativas = len(lista_indices_keys)
    for _ in range(tentativas):
        indice = resgata_chave_da_vez()
        nome_chave = lista_indices_keys[indice]
        chave = dict_api_keys[nome_chave]

        try:
            youtube = build("youtube", "v3", developerKey=chave)
            print(f"Sucesso com a chave: {nome_chave}")

            pula_para_proxima_chave()
            return youtube
        except Exception as e:
            print(f"❌ Erro ao usar chave {nome_chave}: {e}")
            pula_para_proxima_chave()

    raise RuntimeError("❗ Nenhuma chave API conseguiu criar o serviço da API do YouTube.")


teste = chave_api()

print(teste)

