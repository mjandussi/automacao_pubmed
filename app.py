import streamlit as st
import os
import time
import json
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.edge.service import Service
from selenium.webdriver.support.ui import Select
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from datetime import datetime

# --- Configurações da Página do Streamlit ---
st.set_page_config(page_title="Analisador de Artigos PubMed", layout="wide")
st.title("Robô de Análise de Artigos do PubMed com IA")
st.markdown("### Etapa 1: Buscar e Preparar o CSV | Etapa 2: Analisar com IA")

# --- Inicializa variáveis de estado do Streamlit ---
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.total_artigos = 0
    st.session_state.pmids_selecionados_finais = []
    st.session_state.current_batch_start_index = 0
    st.session_state.raw_csv_path = None
    st.session_state.new_filename_base = None

# --- Configurações na Barra Lateral (Sidebar) ---
st.sidebar.header("Configurações da Análise")

openai_api_key = st.sidebar.text_input("Chave da API da OpenAI", type="password")

model_options = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo-0125"]
selected_model = st.sidebar.selectbox("Escolha o Modelo da OpenAI", options=model_options)
temperature = st.sidebar.slider("Temperatura da Análise", 0.0, 1.0, 0.4)

keywords_input = st.sidebar.text_input(
    "Palavras-chave de busca (separadas por vírgula)",
    value="older adults, geriatrics, aging health, elderly care"
)
palavras_chave_busca = [keyword.strip() for keyword in keywords_input.split(',')]

# --- Etapa 1: Botão de Busca ---
st.header("1. Busca e Preparação do CSV")
st.info("Clique no botão abaixo para iniciar a busca no PubMed e preparar o arquivo CSV completo.")
if st.button("1: Buscar e Preparar CSV"):
    if not openai_api_key:
        st.error("Por favor, insira sua chave da API da OpenAI na barra lateral.")
        st.stop()
    
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Reseta o estado para uma nova busca
    st.session_state.df = None
    st.session_state.pmids_selecionados_finais = []
    st.session_state.current_batch_start_index = 0
    st.session_state.raw_csv_path = None
    st.session_state.new_filename_base = None

    with st.spinner("Iniciando a automação do Selenium..."):
        PATH_DO_DRIVER = "msedgedriver.exe"
        url_do_pubmed = "https://pubmed.ncbi.nlm.nih.gov/"
        service = Service(executable_path=PATH_DO_DRIVER)
        
        try:
            driver = webdriver.Edge(service=service)
            driver.get(url_do_pubmed)
            wait = WebDriverWait(driver, 20)
            
            advanced_link = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "Advanced")))
            advanced_link.click()
            time.sleep(2)
            
            for palavra in palavras_chave_busca:
                search_box = wait.until(EC.presence_of_element_located((By.ID, "id_term")))
                search_box.clear()
                search_box.send_keys(palavra)
                time.sleep(1)
                wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "add-button"))).click()
                time.sleep(1)

            search_button = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "search-btn")))
            search_button.click()
            time.sleep(5)
            
            save_button = wait.until(EC.element_to_be_clickable((By.ID, "save-results-panel-trigger")))
            save_button.click()
            time.sleep(2)

            select_element_pages = wait.until(EC.presence_of_element_located((By.ID, "save-action-selection")))
            select_obj_pages = Select(select_element_pages)
            select_obj_pages.select_by_value("all-results")
            time.sleep(2)

            format_element = wait.until(EC.presence_of_element_located((By.ID, "save-action-format")))
            format_select = Select(format_element)
            try:
                format_select.select_by_visible_text("CSV")
            except:
                format_select.select_by_visible_text("csv")
            time.sleep(2)
            
            create_file_button = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "action-panel-submit")))
            create_file_button.click()
            st.info("Botão 'Create file' clicado. Aguardando download...")
            time.sleep(30)
            driver.quit()

            download_dir = 'C:\\Users\\mjwa\\Downloads\\'
            list_of_csv_files = [download_dir + f for f in os.listdir(download_dir) if f.startswith('csv-') and f.endswith('.csv')]
            
            if not list_of_csv_files:
                st.error("Não foi possível encontrar o arquivo CSV baixado.")
                st.stop()
            
            latest_file = max(list_of_csv_files, key=os.path.getctime)
            
            keywords_filename = '-'.join([keyword.strip().replace(' ', '_') for keyword in palavras_chave_busca])
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            st.session_state.new_filename_base = f"{keywords_filename}-{timestamp}"
            new_filename_raw_csv = os.path.join(download_dir, f"{st.session_state.new_filename_base}_original.csv")
            
            os.rename(latest_file, new_filename_raw_csv)
            
            df_full = pd.read_csv(new_filename_raw_csv)
            st.session_state.df = df_full
            st.session_state.total_artigos = len(df_full)
            st.session_state.raw_csv_path = new_filename_raw_csv
            
            st.success("Busca e preparação do CSV concluídas!")
            st.write(f"Total de artigos encontrados: **{st.session_state.total_artigos}**")

        except Exception as e:
            st.error(f"Erro na Etapa 1: {e}")
            try:
                driver.quit()
            except:
                pass
            st.stop()

# --- Etapa 2: Análise com IA ---
if st.session_state.df is not None:
    st.header("2. Análise com IA")
    st.info("Agora você pode configurar os parâmetros da análise e iniciar o processamento em lotes.")
    st.write(f"CSV preparado com **{st.session_state.total_artigos}** artigos.")
    
    artigos_para_analisar = st.number_input(
        "Quantidade TOTAL de artigos a serem analisados",
        min_value=1,
        max_value=st.session_state.total_artigos,
        value=min(200, st.session_state.total_artigos)
    )
    artigos_para_selecionar_por_lote = st.number_input(
        "Quantos artigos principais selecionar por lote?",
        min_value=1,
        max_value=50,
        value=5
    )
    tamanho_do_lote = st.number_input(
        "Tamanho do lote para a análise da IA",
        min_value=5,
        max_value=1000,
        value=200
    )

    # --- REMOÇÃO DAS COLUNAS AQUI ---
    if st.button("2: Iniciar/Continuar Análise da IA"):
        
        with st.spinner(f"Processando lotes a partir do índice {st.session_state.current_batch_start_index}..."):
            llm = ChatOpenAI(temperature=temperature, model_name=selected_model) 
            
            total_artigos_para_analisar = min(artigos_para_analisar, len(st.session_state.df))
            
            progress_bar = st.progress(st.session_state.current_batch_start_index / total_artigos_para_analisar)
            
            for i in range(st.session_state.current_batch_start_index, total_artigos_para_analisar, tamanho_do_lote):
                lote_de_artigos = st.session_state.df.iloc[i:min(i + tamanho_do_lote, total_artigos_para_analisar)]
                
                dados_para_llm_list = []
                for index, row in lote_de_artigos.iterrows(): 
                    title = row.get('Title', 'Título não encontrado')
                    abstract = row.get('Abstract', 'Abstract não encontrado')
                    pmid = row.get('PMID', 'PMID não encontrado')
                    dados_para_llm_list.append(f"Título: {title}\nPMID: {pmid}\nAbstract: {abstract}\n")
                
                dados_para_llm_string = "\n---\n".join(dados_para_llm_list)

                prompt_template = f"""
                Você é um especialista em saúde do idoso e está ajudando a selecionar os artigos mais relevantes para uma revisão bibliográfica.
                Analise os seguintes abstracts de artigos científicos do PubMed sobre saúde do idoso.
                
                Sua tarefa é identificar e ranquear os {artigos_para_selecionar_por_lote} artigos mais relevantes deste lote de {len(lote_de_artigos)} artigos para uma revisão bibliográfica.
                Selecione os artigos que parecem ter a maior profundidade de análise, metodologia robusta ou resultados significativos.
                
                A saída deve ser SOMENTE a lista JSON, sem nenhum texto introdutório, explicativo ou adicional.
                É de extrema importância que o formato de saída seja um JSON válido e completo.
                
                Formate sua resposta como uma lista JSON, onde cada item da lista é um objeto com as seguintes chaves:
                - "rank": a posição do artigo na sua classificação (1 ao {artigos_para_selecionar_por_lote})
                - "title": o título do artigo
                - "pmid": o PMID do artigo
                - "justificativa": uma breve explicação (no máximo 2 frases) do porquê o artigo foi selecionado
                
                Conteúdo dos Abstracts:
                {{abstracts}}
                """
                
                prompt = PromptTemplate(input_variables=["abstracts"], template=prompt_template)
                llm_chain = LLMChain(llm=llm, prompt=prompt)
                
                st.info(f"Analisando lote {i // tamanho_do_lote + 1}...")
                
                resposta_str = llm_chain.run(abstracts=dados_para_llm_string)
                
                try:
                    resposta_limpa = resposta_str.strip().replace('```json', '').replace('```', '')
                    lista_de_artigos_ia = json.loads(resposta_limpa)
                    pmids_do_lote = [str(item['pmid']) for item in lista_de_artigos_ia]
                    st.session_state.pmids_selecionados_finais.extend(pmids_do_lote)
                except json.JSONDecodeError as e:
                    st.error(f"Erro ao processar o JSON do lote {i // tamanho_do_lote + 1}. Erro: {e}")
                    st.code(resposta_str, language='json')
                
                st.session_state.current_batch_start_index = min(i + tamanho_do_lote, total_artigos_para_analisar)
                progresso = st.session_state.current_batch_start_index / total_artigos_para_analisar
                progress_bar.progress(progresso)
                
                if st.session_state.current_batch_start_index >= total_artigos_para_analisar:
                    break
        
        if st.session_state.current_batch_start_index >= total_artigos_para_analisar:
            st.success("Análise de todos os lotes concluída!")
        else:
            st.info(f"Análise pausada. Processados {st.session_state.current_batch_start_index} de {total_artigos_para_analisar} artigos.")

    if st.button("Exportar CSV da Análise"):
        if not st.session_state.pmids_selecionados_finais:
            st.warning("Nenhum artigo foi selecionado ainda. Inicie a análise primeiro.")
        else:
            st.success("Gerando CSV final...")
            df_filtrado = st.session_state.df[st.session_state.df['PMID'].astype(str).isin(st.session_state.pmids_selecionados_finais)]
            
            final_filename = f"{len(st.session_state.pmids_selecionados_finais)}_principais_{st.session_state.new_filename_base}.csv"
            df_filtrado.to_csv(final_filename, index=False)
            
            st.subheader("Resultado Final")
            st.dataframe(df_filtrado)
            
            with open(final_filename, "rb") as file:
                st.download_button(
                    label=f"Baixar CSV com {len(st.session_state.pmids_selecionados_finais)} artigos",
                    data=file,
                    file_name=final_filename,
                    mime="text/csv"
                )
    
    st.subheader("Informações de Estado")
    st.write(f"Artigos já processados: **{st.session_state.current_batch_start_index}** de **{artigos_para_analisar}**")
    st.write(f"Total de PMIDs selecionados até o momento: **{len(st.session_state.pmids_selecionados_finais)}**")

    if st.session_state.raw_csv_path:
        with open(st.session_state.raw_csv_path, "rb") as file:
            st.download_button(
                label="Baixar CSV Completo da Busca (Raw)",
                data=file,
                file_name=os.path.basename(st.session_state.raw_csv_path),
                mime="text/csv"
            )