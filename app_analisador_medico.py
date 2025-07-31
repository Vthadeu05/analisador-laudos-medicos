
import streamlit as st
import PyPDF2
import spacy
import re
import os

# --- Verificação e Download do Modelo spaCy ---
# Esta é a parte crucial que corrige o erro de inicialização.
# O aplicativo irá baixar o modelo pt_core_news_sm apenas se ele não estiver presente.
@st.cache_resource
def load_spacy_model():
    model_name = "pt_core_news_sm"
    try:
        # Tenta carregar o modelo. Se não existir, a exceção é capturada.
        nlp = spacy.load(model_name)
    except OSError:
        with st.spinner(f"Baixando modelo de linguagem '{model_name}' (pode levar alguns minutos)..."):
            spacy.cli.download(model_name)
            nlp = spacy.load(model_name)
    return nlp

nlp = load_spacy_model()


# --- Configurações da Página do Streamlit ---
st.set_page_config(
    page_title="Analisador Inteligente de Laudos",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Função para extrair texto de PDF ---
@st.cache_data
def extract_text_from_pdf(pdf_file):
    """Extrai texto de um arquivo PDF carregado."""
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.error(f"Erro ao extrair texto do PDF: {e}. Verifique se o PDF está legível e não é uma imagem escaneada.")
        return ""
    return text

# --- Função para processar o texto e extrair informações ---
def process_medical_text(text):
    """
    Processa o texto extraído do PDF para identificar informações médicas chave.
    A lógica é baseada em padrões de texto e palavras-chave.
    """
    extracted_info = {
        "Palavras-chave de Reconhecimento": set(),
        "Diagnóstico Possível": "Não identificado claramente",
        "Exames Padrão Ouro": set(),
        "Exames Complementares": set(),
        "Tratamento Sugerido": "Não identificado claramente",
        "Diagnóstico Diferencial": set()
    }

    doc = nlp(text.lower())

    # --- 1. Palavras-chave de Reconhecimento ---
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
    match_diag = re.search(r'(?:diagnóstico de|compatível com|sugestivo de|hipótese diagnóstica)[:\s]*([\w\s,-]+?)(?:\.|\n|e\s|\bpara\b|em\s|\bcom\b|\bsem\b|$)', text, re.IGNORECASE)
    if match_diag:
        diagnosis = match_diag.group(1).strip()
        diagnosis = re.sub(r'[,.;:\s]+$', '', diagnosis)
        if len(diagnosis) > 100: diagnosis = diagnosis[:100] + "..."
        extracted_info["Diagnóstico Possível"] = diagnosis.capitalize()
    else:
        potential_diagnoses = [ent.text for ent in doc.ents if ent.label_ in ["DISEASE", "MEDICAL_CONDITION", "SYMPTOM", "ORG"]]
        if potential_diagnoses:
            extracted_info["Diagnóstico Possível"] = ", ".join(list(set(potential_diagnoses[:3]))).capitalize()

    # --- 3. Exames Padrão Ouro e 4. Exames Complementares ---
    exam_keywords_list = [
        "ressonância magnética", "tomografia computadorizada", "raio-x", "ultrassonografia",
        "exame de sangue", "hemograma", "urina", "cultura", "biópsia", "endoscopia",
        "colonoscopia", "eletrocardiograma", "teste ergométrico", "sorologia", "PCR",
        "anatomopatológico", "imunohistoquímica", "cultura de urina", "teste de glicemia",
        "colesterol", "triglicerídeos", "creatinina", "ureia", "ecocardiograma",
        "teste de função pulmonar", "espirometria", "tomografia por emissão de pósitrons", "PET-CT"
    ]
    for exam in exam_keywords_list:
        if re.search(r'\b' + re.escape(exam) + r'\b', text, re.IGNORECASE):
            context_around_exam = text[max(0, text.lower().find(exam.lower()) - 50):min(len(text), text.lower().find(exam.lower()) + len(exam) + 50)]
            if re.search(r'padrão ouro|gold standard', context_around_exam, re.IGNORECASE):
                extracted_info["Exames Padrão Ouro"].add(exam.capitalize())
            else:
                extracted_info["Exames Complementares"].add(exam.capitalize())

    # --- 5. Tratamento Sugerido ---
    treatment_keywords_list = [
        "tratamento", "terapia", "medicação", "medicamento", "cirurgia", "intervenção",
        "aconselhamento", "reabilitação", "dose", "prescrição", "conduta", "indicado",
        "administrar", "uso de", "cirúrgico", "farmacológico", "fisioterapia", "quimioterapia",
        "radioterapia", "dieta", "repouso"
    ]
    found_treatments = []
    for sent in doc.sents:
        if any(re.search(r'\b' + re.escape(keyword) + r'\b', sent.text, re.IGNORECASE) for keyword in treatment_keywords_list):
            found_treatments.append(sent.text.strip())
            if len(found_treatments) >= 2: break
    if found_treatments:
        extracted_info["Tratamento Sugerido"] = " ".join(found_treatments).capitalize()
    else:
        for keyword in treatment_keywords_list:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                extracted_info["Tratamento Sugerido"] = keyword.capitalize() + " (mencionado)"
                break

    # --- 6. Diagnóstico Diferencial ---
    differential_keywords_list = ["diagnóstico diferencial", "DD", "descartar", "excluir", "considerar a possibilidade de"]
    found_diff_diag = []
    for keyword in differential_keywords_list:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
            match_dd = re.search(r'(' + re.escape(keyword) + r'[:\s]*(.*?)(?:\.|\n|e\s|\bcom\b|$))', text, re.IGNORECASE)
            if match_dd:
                diff_diag = match_dd.group(2).strip()
                diff_diag = re.sub(r'[,.;:\s]+$', '', diff_diag)
                if len(diff_diag) > 100: diff_diag = diff_diag[:100] + "..."
                found_diff_diag.append(diff_diag.capitalize())
            else:
                found_diff_diag.append(keyword.capitalize())

    if found_diff_diag:
        extracted_info["Diagnóstico Diferencial"] = set(found_diff_diag)
    else:
        extracted_info["Diagnóstico Diferencial"].add("Não identificado claramente (requer análise manual)")

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
