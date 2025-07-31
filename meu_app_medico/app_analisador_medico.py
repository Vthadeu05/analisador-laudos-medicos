import streamlit as st
import PyPDF2
import spacy
import re

# --- Carregar o modelo spaCy para processamento de linguagem ---
# Preferimos o modelo em português, mas temos um fallback para inglês.
try:
    nlp = spacy.load("pt_core_news_sm")
except OSError:
    st.warning("Modelo spaCy para português não encontrado. Tentando carregar modelo em inglês.")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.error("Nenhum modelo spaCy encontrado. Por favor, execute 'python -m spacy download pt_core_news_sm' ou 'en_core_web_sm' no seu terminal.")
        st.stop() # Interrompe a execução se nenhum modelo puder ser carregado


# --- Configurações da Página do Streamlit ---
st.set_page_config(
    page_title="Analisador Inteligente de Laudos",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Função para extrair texto de PDF ---
@st.cache_data # Cache para não reprocessar o mesmo PDF
def extract_text_from_pdf(pdf_file):
    """Extrai texto de um arquivo PDF carregado."""
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n" # Adiciona quebra de linha entre páginas
    except Exception as e:
        st.error(f"Erro ao extrair texto do PDF: {e}. Verifique se o PDF está legível.")
        return ""
    return text

# --- Função para processar o texto e extrair informações ---
def process_medical_text(text):
    """
    Processa o texto extraído do PDF para identificar informações médicas chave.
    A lógica é baseada em padrões de texto e palavras-chave.
    """
    extracted_info = {
        "Palavras-chave de Reconhecimento": set(), # Usamos set para garantir unicidade
        "Diagnóstico Possível": "Não identificado claramente",
        "Exames Padrão Ouro": set(),
        "Exames Complementares": set(),
        "Tratamento Sugerido": "Não identificado claramente",
        "Diagnóstico Diferencial": set()
    }

    doc = nlp(text.lower()) # Processa o texto em minúsculas para facilitar a correspondência

    # --- 1. Palavras-chave de Reconhecimento ---
    # Termos comuns em laudos que indicam achados ou queixas.
    keywords_recognition_list = [
        "sintoma", "sintomas", "achado", "achados", "clínico", "clínica",
        "história", "quadro", "paciente", "queixa", "queixas", "dor", "febre",
        "inflamação", "infecção", "alteração", "lesão", "presença de", "evidência de",
        "exame físico", "anamnese", "resultado de"
    ]
    for keyword in keywords_recognition_list:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
            extracted_info["Palavras-chave de Reconhecimento"].add(keyword)


    # --- 2. Diagnóstico Possível ---
    # Tentativa de capturar o diagnóstico principal.
    # Padrões comuns: "diagnóstico de", "compatível com", "sugestivo de", "hipótese diagnóstica"
    match_diag = re.search(r'(?:diagnóstico de|compatível com|sugestivo de|hipótese diagnóstica)[:\s]*([\w\s,-]+?)(?:\.|\n|e\s|\bpara\b|em\s|\bcom\b|\bsem\b|$)', text, re.IGNORECASE)
    if match_diag:
        # Pega o grupo capturado e limpa espaços extras
        diagnosis = match_diag.group(1).strip()
        # Remove caracteres indesejados no final
        diagnosis = re.sub(r'[,.;:\s]+$', '', diagnosis)
        # Limita o tamanho da string do diagnóstico para evitar capturas muito longas
        if len(diagnosis) > 100:
            diagnosis = diagnosis[:100] + "..."
        extracted_info["Diagnóstico Possível"] = diagnosis.capitalize()
    else:
        # Fallback: tentar identificar entidades médicas gerais se o padrão não for encontrado
        # Isso é limitado, mas pode pegar nomes de doenças se o modelo do spaCy as reconhecer.
        # Estamos procurando por "entidades nomeadas" que podem ser doenças ou problemas.
        potential_diagnoses = [ent.text for ent in doc.ents if ent.label_ in ["DISEASE", "MEDICAL_CONDITION", "SYMPTOM", "ORG"]]
        if potential_diagnoses:
            # Pega os 2-3 primeiros termos mais prováveis ou frequentes como um diagnóstico possível
            # Poderíamos adicionar contagem de frequência aqui para maior relevância
            extracted_info["Diagnóstico Possível"] = ", ".join(list(set(potential_diagnoses[:3]))).capitalize()


    # --- 3. Exames Padrão Ouro e 4. Exames Complementares ---
    # Uma lista mais abrangente de termos de exames.
    exam_keywords_list = [
        "ressonância magnética", "tomografia computadorizada", "raio-x", "ultrassonografia",
        "exame de sangue", "hemograma", "urina", "cultura", "biópsia", "endoscopia",
        "colonoscopia", "eletrocardiograma", "teste ergométrico", "sorologia", "PCR",
        "anatomopatológico", "imunohistoquímica", "cultura de urina", "teste de glicemia",
        "colesterol", "triglicerídeos", "creatinina", "ureia", "ecocardiograma",
        "teste de função pulmonar", "espirometria", "tomografia por emissão de pósitrons", "PET-CT"
    ]

    for exam in exam_keywords_list:
        # Verifica se o exame é mencionado no texto
        if re.search(r'\b' + re.escape(exam) + r'\b', text, re.IGNORECASE):
            # Tenta inferir "padrão ouro" se a frase estiver próxima
            # Esta é uma heuristicia e pode não ser 100% precisa.
            context_around_exam = text[max(0, text.lower().find(exam.lower()) - 50):min(len(text), text.lower().find(exam.lower()) + len(exam) + 50)]
            if re.search(r'padrão ouro|gold standard', context_around_exam, re.IGNORECASE):
                extracted_info["Exames Padrão Ouro"].add(exam.capitalize())
            else:
                extracted_info["Exames Complementares"].add(exam.capitalize())


    # --- 5. Tratamento Sugerido ---
    # Termos comuns que indicam tratamento.
    treatment_keywords_list = [
        "tratamento", "terapia", "medicação", "medicamento", "cirurgia", "intervenção",
        "aconselhamento", "reabilitação", "dose", "prescrição", "conduta", "indicado",
        "administrar", "uso de", "cirúrgico", "farmacológico", "fisioterapia", "quimioterapia",
        "radioterapia", "dieta", "repouso"
    ]
    found_treatments = []
    # Procurar sentenças que contenham termos de tratamento e tentar extrair a sentença completa
    for sent in doc.sents:
        if any(re.search(r'\b' + re.escape(keyword) + r'\b', sent.text, re.IGNORECASE) for keyword in treatment_keywords_list):
            found_treatments.append(sent.text.strip())
            if len(found_treatments) >= 2: # Pegar no máximo 2 sentenças como exemplo
                break
    if found_treatments:
        extracted_info["Tratamento Sugerido"] = " ".join(found_treatments).capitalize()
    else:
        # Fallback: tentar encontrar termos de tratamento isolados
        for keyword in treatment_keywords_list:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                extracted_info["Tratamento Sugerido"] = keyword.capitalize() + " (mencionado)"
                break # Pega o primeiro encontrado


    # --- 6. Diagnóstico Diferencial ---
    # Termos que indicam outras condições a serem consideradas.
    differential_keywords_list = ["diagnóstico diferencial", "DD", "descartar", "excluir", "considerar a possibilidade de"]
    found_diff_diag = []
    for keyword in differential_keywords_list:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
            # Tenta capturar a frase após a palavra-chave do DD
            match_dd = re.search(r'(' + re.escape(keyword) + r'[:\s]*(.*?)(?:\.|\n|e\s|\bcom\b|$))', text, re.IGNORECASE)
            if match_dd:
                diff_diag = match_dd.group(2).strip()
                diff_diag = re.sub(r'[,.;:\s]+$', '', diff_diag)
                if len(diff_diag) > 100:
                    diff_diag = diff_diag[:100] + "..."
                found_diff_diag.append(diff_diag.capitalize())
            else:
                # Se não encontrar um padrão específico, adiciona a própria palavra-chave
                found_diff_diag.append(keyword.capitalize())

    if found_diff_diag:
        extracted_info["Diagnóstico Diferencial"] = set(found_diff_diag)
    else:
        extracted_info["Diagnóstico Diferencial"].add("Não identificado claramente (requer análise manual)")


    # Converter sets para listas para exibição
    extracted_info["Palavras-chave de Reconhecimento"] = list(extracted_info["Palavras-chave de Reconhecimento"])
    extracted_info["Exames Padrão Ouro"] = list(extracted_info["Exames Padrão Ouro"])
    extracted_info["Exames Complementares"] = list(extracted_info["Exames Complementares"])
    extracted_info["Diagnóstico Diferencial"] = list(extracted_info["Diagnóstico Diferencial"])

    return extracted_info

# --- Título e Descrição da Interface ---
st.title("📄 Analisador Inteligente de Laudos Médicos")
st.markdown("""
    Este aplicativo ajuda a extrair e organizar informações chave de documentos PDF,
    como possíveis diagnósticos, exames e tratamentos.
    **Importante:** Esta ferramenta é um **auxílio** para análise textual e **não substitui**
    a avaliação e o diagnóstico de um profissional de saúde qualificado.
""")

st.markdown("---")

# --- Seção de Upload de Arquivo ---
st.subheader("1. Carregue seu Laudo em PDF")
uploaded_file = st.file_uploader("Arraste e solte ou clique para escolher um arquivo PDF", type="pdf")

pdf_text = ""
if uploaded_file is not None:
    st.success("✅ Arquivo PDF carregado com sucesso!")
    with st.spinner("Extraindo texto do PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)

    if pdf_text:
        st.expander("Prévia do Texto Extraído (clique para expandir)").text(pdf_text[:2000] + "..." if len(pdf_text) > 2000 else pdf_text)
        st.markdown("---")
        st.subheader("2. Analisar Laudo")
        if st.button("🚀 Iniciar Análise"):
            with st.spinner("Analisando o texto do laudo..."):
                analysis_results = process_medical_text(pdf_text)

            st.markdown("---")
            st.subheader("3. Resultados da Análise")

            # --- Exibição dos Resultados ---
            st.markdown("### 🔍 Palavras-chave de Reconhecimento")
            if analysis_results["Palavras-chave de Reconhecimento"]:
                st.info(", ".join(analysis_results["Palavras-chave de Reconhecimento"]))
            else:
                st.info("Nenhuma palavra-chave de reconhecimento específica encontrada.")

            st.markdown("### 💡 Diagnóstico Possível")
            st.success(analysis_results["Diagnóstico Possível"])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### ✅ Exames Padrão Ouro")
                if analysis_results["Exames Padrão Ouro"]:
                    st.write(", ".join(analysis_results["Exames Padrão Ouro"]))
                else:
                    st.write("Nenhum exame padrão ouro identificado.")
            with col2:
                st.markdown("### ➕ Exames Complementares")
                if analysis_results["Exames Complementares"]:
                    st.write(", ".join(analysis_results["Exames Complementares"]))
                else:
                    st.write("Nenhum exame complementar identificado.")

            st.markdown("### 💊 Tratamento Sugerido")
            st.warning(analysis_results["Tratamento Sugerido"])

            st.markdown("### ↔️ Diagnóstico Diferencial")
            if analysis_results["Diagnóstico Diferencial"]:
                st.error(", ".join(analysis_results["Diagnóstico Diferencial"]))
            else:
                st.error("Nenhum diagnóstico diferencial identificado claramente.")

            st.markdown("---")
            st.info("Lembre-se: Este é um protótipo! A precisão depende muito da clareza do documento e da complexidade da terminologia.")
        else:
            st.info("Clique no botão 'Iniciar Análise' para processar o laudo.")
    else:
        st.error("Não foi possível extrair texto do PDF. Por favor, tente com outro arquivo ou verifique se o PDF não é uma imagem.")
else:
    st.info("Aguardando o carregamento de um arquivo PDF...")
