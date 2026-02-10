
import os
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from openai import OpenAI

APP_TITLE = "Aprendendo Algoritmos"
DATA_DIR = "data"
STATE_PATH = os.path.join(DATA_DIR, "state.json")
SUBMISSIONS_PATH = os.path.join(DATA_DIR, "submissions.jsonl")
GRADES_PATH = os.path.join(DATA_DIR, "grades.json")

DEFAULT_QUESTION = (
    "Aguarde a questão para enviar sua resposta."
    "Escreva um passo a passo dessas instruções."
)

# -------------------------
# Persistência (arquivos)
# -------------------------
def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def load_state() -> Dict[str, Any]:
    ensure_data_dir()
    if not os.path.exists(STATE_PATH):
        state = {"question": DEFAULT_QUESTION, "accepting": True, "updated_at": now_iso()}
        save_state(state)
        return state
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(state: Dict[str, Any]) -> None:
    ensure_data_dir()
    state["updated_at"] = now_iso()
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def append_submission(entry: Dict[str, Any]) -> None:
    ensure_data_dir()
    with open(SUBMISSIONS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def load_submissions() -> List[Dict[str, Any]]:
    ensure_data_dir()
    if not os.path.exists(SUBMISSIONS_PATH):
        return []
    rows: List[Dict[str, Any]] = []
    with open(SUBMISSIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def clear_submissions(mode: str, current_question: Optional[str] = None) -> None:
    """
    mode:
      - "all": apaga tudo
      - "current": apaga apenas a questão atual
    """
    ensure_data_dir()
    if mode == "all":
        if os.path.exists(SUBMISSIONS_PATH):
            os.remove(SUBMISSIONS_PATH)
        return

    if mode == "current" and current_question is not None:
        subs = load_submissions()
        kept = [s for s in subs if s.get("question") != current_question]
        with open(SUBMISSIONS_PATH, "w", encoding="utf-8") as f:
            for s in kept:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        return

def load_grades() -> Dict[str, Any]:
    ensure_data_dir()
    if not os.path.exists(GRADES_PATH):
        return {}
    with open(GRADES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_grades(grades: Dict[str, Any]) -> None:
    ensure_data_dir()
    with open(GRADES_PATH, "w", encoding="utf-8") as f:
        json.dump(grades, f, ensure_ascii=False, indent=2)

def clear_grades() -> None:
    ensure_data_dir()
    if os.path.exists(GRADES_PATH):
        os.remove(GRADES_PATH)

# -------------------------
# Admin auth simples
# -------------------------
def is_admin_logged_in() -> bool:
    return bool(st.session_state.get("admin_authed", False))

def admin_login_ui() -> None:
    st.subheader("🔐 Área Admin")
    user = st.text_input("Usuário", value="", key="admin_user")
    pwd = st.text_input("Senha", value="", type="password", key="admin_pass")

    secrets_user = st.secrets.get("ADMIN_USER", "")
    secrets_pass = st.secrets.get("ADMIN_PASS", "")

    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("Entrar", type="primary"):
            if user == secrets_user and pwd == secrets_pass:
                st.session_state["admin_authed"] = True
                st.success("Admin autenticado.")
            else:
                st.error("Usuário ou senha inválidos.")
    with c2:
        if is_admin_logged_in() and st.button("Sair"):
            st.session_state["admin_authed"] = False
            st.info("Saiu da área admin.")

# -------------------------
# OpenAI / Avaliação
# -------------------------
def build_rubric_prompt(question: str, submissions: List[Dict[str, Any]]) -> str:
    payload = []
    for s in submissions:
        payload.append(
            {
                "submission_id": s.get("submission_id", ""),
                "student": (s.get("student_name") or "").strip(),
                "answer": (s.get("answer") or "").strip(),
                "submitted_at": s.get("submitted_at", ""),
            }
        )

    return f"""
Você é um professor da disciplina de ALGORITMOS E PROGRAMAÇÃO e precisaa avaliar as respostas sobre ALGORITMOS (passo a passo).
Avalie cada resposta de forma objetiva e respeitosa.

PERGUNTA:
{question}

CRITÉRIOS (0 a 10):
- Sequência lógica (passos em ordem)
- Clareza e objetividade (instruções simples, sem ambiguidade)
- Uso de decisões/condições quando necessário (ex.: "se a letra for antes/depois...")
- Completude (início, processo e término)
- Adequação ao público (criança alfabetizada)

REGRAS IMPORTANTES:
- Não invente nomes, respostas, nem IDs.
- Se a resposta estiver vazia ou muito curta, dê nota baixa e explique.
- Evite jargão excessivo. Seja didático.

TAREFA EXTRA:
- Selecione o TOP 3 MELHORES RESPOSTAS (mais “acadêmicas”): melhor estrutura algorítmica,
  clareza, completude e bom uso de condições.
- Se houver empate, prefira a mais fácil de seguir.
- Preencha o bloco "top3" com exatamente 3 itens (ou menos, se houver menos de 3 submissões).

FORMATO DE SAÍDA (OBRIGATÓRIO):
Retorne APENAS um JSON válido, sem texto extra, exatamente com esta estrutura:

{{
  "results": [
    {{
      "submission_id": "string",
      "student": "string",
      "score": 0,
      "strengths": ["..."],
      "improvements": ["..."],
      "one_suggestion": "..."
    }}
  ],
  "top3": [
    {{
      "rank": 1,
      "submission_id": "string",
      "student": "string",
      "why_it_wins": ["..."],
      "highlight_excerpt": "..."
    }}
  ],
  "summary": {{
    "common_strengths": ["..."],
    "common_gaps": ["..."],
    "teacher_tip": "..."
  }}
}}

SUBMISSÕES (JSON):
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()

def parse_json_safely(text: str) -> Dict[str, Any]:
    # 1) tenta direto
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) tenta extrair primeiro bloco JSON
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("Não foi possível interpretar JSON do retorno do modelo.")

def run_openai_evaluation(api_key: str, model: str, prompt: str) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)
    resp = client.responses.create(
        model=model,
        input=prompt,
    )
    return parse_json_safely(resp.output_text)

def normalize_results(grades: Dict[str, Any]) -> Dict[str, Any]:
    # Garantir campos mínimos (sem quebrar UI)
    grades = grades or {}
    grades.setdefault("results", [])
    grades.setdefault("top3", [])
    grades.setdefault("summary", {"common_strengths": [], "common_gaps": [], "teacher_tip": ""})
    return grades

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")
st.title("🧠 Aprendendo Algoritmos 💻")

state = load_state()
current_q = state.get("question") or DEFAULT_QUESTION

tabs = st.tabs(["👩‍🎓 Aluno", "🛠️ Admin"])

# -------------------------
# ALUNO
# -------------------------
with tabs[0]:
    st.subheader("Pergunta do dia")
    st.info(current_q)

    if not state.get("accepting", True):
        st.warning("⛔ O envio de respostas está fechado no momento.")
    else:
        with st.form("student_form", clear_on_submit=True):
            student_name = st.text_input("Seu nome (obrigatório)")
            answer = st.text_area(
                "Escreva aqui o seu algoritmo :)",
                height=220,
                placeholder="Exemplo: 1) Abra o estojo, 2) Procure a caneta azul, 3) Pegue a caneta azul ....",
            )
            submitted = st.form_submit_button("Enviar resposta", type="primary")

        if submitted:
            student_name = (student_name or "").strip()
            answer = (answer or "").strip()

            if len(student_name) < 2:
                st.error("Digite seu nome.")
            elif len(answer) < 30:
                st.error("Escreva uma resposta um pouco mais completa (mínimo ~30 caracteres).")
            else:
                entry = {
                    "submission_id": f"{int(time.time()*1000)}",
                    "student_name": student_name,
                    "answer": answer,
                    "question": current_q,
                    "submitted_at": now_iso(),
                }
                append_submission(entry)
                st.success("✅ Resposta registrada! Obrigado 🙂")

    st.divider()
    st.caption("Dica: passos curtos, ordem clara e use condições do tipo “se... então...” quando fizer sentido.")

# -------------------------
# ADMIN
# -------------------------
with tabs[1]:
    admin_login_ui()

    if not is_admin_logged_in():
        st.info("Entre para gerenciar a questão e avaliar respostas.")
    else:
        st.subheader("⚙️ Configurações da Questão")

        colA, colB = st.columns([2, 1])
        with colA:
            new_q = st.text_area("Texto da questão", value=current_q, height=140)
            if st.button("Salvar questão", type="primary"):
                state["question"] = (new_q.strip() or DEFAULT_QUESTION)
                save_state(state)
                st.success("Questão atualizada.")
                st.rerun()

        with colB:
            accepting = st.toggle("Aceitar novas respostas", value=bool(state.get("accepting", True)))
            if accepting != bool(state.get("accepting", True)):
                state["accepting"] = accepting
                save_state(state)
                st.success("Status de recebimento atualizado.")
                st.rerun()
            st.caption(f"Última atualização: {state.get('updated_at','-')}")

        st.divider()
        st.subheader("📥 Submissões")

        subs_all = load_submissions()
        subs_current = [s for s in subs_all if s.get("question") == (state.get("question") or DEFAULT_QUESTION)]

        m1, m2, m3 = st.columns(3)
        m1.metric("Total (todas)", len(subs_all))
        m2.metric("Da questão atual", len(subs_current))
        m3.metric("Recebendo agora?", "Sim" if state.get("accepting", True) else "Não")

        st.markdown("### 🧹 Limpeza")
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("Limpar questão atual", disabled=(len(subs_current) == 0)):
                clear_submissions(mode="current", current_question=state.get("question"))
                st.success("Submissões da questão atual apagadas.")
                st.rerun()
        with c2:
            if st.button("Limpar TUDO"):
                clear_submissions(mode="all")
                st.success("Todas as submissões foram apagadas.")
                st.rerun()
        with c3:
            st.caption("Isso apaga respostas do arquivo local. Use com cuidado 🙂")

        df = pd.DataFrame(subs_current) if subs_current else pd.DataFrame(columns=["submission_id","student_name","answer","submitted_at"])
        if not df.empty:
            st.dataframe(df[["submitted_at", "student_name", "answer"]], use_container_width=True, height=320)
            csv = df[["submission_id","student_name","answer","submitted_at","question"]].to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Baixar CSV (questão atual)", data=csv, file_name="submissoes.csv", mime="text/csv")
        else:
            st.warning("Ainda não há submissões para a questão atual.")

        st.divider()
        st.subheader("🤖 Avaliar respostas")

        st.caption(" ")
        api_key = st.text_input("API Key", type="password", placeholder="sk-...", key="openai_api_key")

        colm1, colm2, colm3 = st.columns([1, 1, 2])
        with colm1:
            model = st.selectbox("Modelo", options=["gpt-5.2", "gpt-5-mini", "gpt-4.1"], index=0)
        with colm2:
            st.write("")
            st.write("")
            only_current = st.toggle("Avaliar só questão atual", value=True)
        with colm3:
            st.caption("Depois de inserir a chave é só ir :)")

        eval_list = subs_current if only_current else subs_all

        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            if st.button("Avaliar agora", type="primary", disabled=(len(eval_list) == 0)):
                if not api_key or len(api_key) < 10:
                    st.error("Informe uma API Key válida.")
                else:
                    q_for_eval = (state.get("question") or DEFAULT_QUESTION) if only_current else "MÚLTIPLAS QUESTÕES (misturadas)"
                    prompt = build_rubric_prompt(q_for_eval if only_current else "Avalie as respostas para a questão abaixo (cada submissão já traz a pergunta).", eval_list)

                    # Se for avaliar tudo, melhor manter a pergunta por submissão no payload.
                    # Aqui, como simplificação, usamos o mesmo prompt; o payload já tem ID/aluno/resposta.
                    # Para múltiplas questões, você pode adaptar o payload para incluir "question" também.

                    try:
                        with st.spinner("Avaliando..."):
                            result = run_openai_evaluation(api_key=api_key, model=model, prompt=prompt)
                        result = normalize_results(result)
                        result["evaluated_at"] = now_iso()
                        result["evaluated_count"] = len(eval_list)
                        result["question_mode"] = "current_only" if only_current else "all"
                        save_grades(result)
                        st.success("✅ Avaliação concluída e salva em data/grades.json.")
                    except Exception as e:
                        st.error(f"Falha ao avaliar: {e}")

        with btn_col2:
            if st.button("Apagar resultados (grades.json)"):
                clear_grades()
                st.success("Resultados apagados.")
                st.rerun()

        st.divider()
        st.subheader("📊 Resultados salvos (inclui TOP 3)")

        grades = normalize_results(load_grades())
        if grades and grades.get("results"):
            # TOP 3
            top3 = grades.get("top3", [])
            if top3:
                st.markdown("### 🏆 TOP 3 (melhores respostas)")
                for item in top3:
                    rank = item.get("rank", "?")
                    student = item.get("student", "-")
                    sid = item.get("submission_id", "-")
                    st.markdown(f"**#{rank} — {student}**  \nSubmission ID: `{sid}`")

                    why = item.get("why_it_wins", [])
                    if why:
                        st.write("**Por que entrou no TOP 3:**")
                        for w in why:
                            st.write(f"- {w}")

                    excerpt = item.get("highlight_excerpt")
                    if excerpt:
                        st.write("**Trecho destaque:**")
                        st.code(excerpt)

                    st.divider()

            # Tabela de resultados
            st.markdown("### 🧾 Notas e feedback (todas as submissões avaliadas)")
            df_g = pd.DataFrame(grades.get("results", []))
            if not df_g.empty:
                st.dataframe(df_g, use_container_width=True, height=320)
                csv_g = df_g.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Baixar CSV (notas/feedbacks)", data=csv_g, file_name="avaliacao.csv", mime="text/csv")

            # Resumo para o professor
            summary = grades.get("summary", {})
            if summary:
                st.markdown("### 🧑‍🏫 Resumo para o professor")
                common_strengths = summary.get("common_strengths", [])
                common_gaps = summary.get("common_gaps", [])
                teacher_tip = summary.get("teacher_tip", "")

                if common_strengths:
                    st.write("**Forças comuns:**")
                    for x in common_strengths:
                        st.write(f"- {x}")

                if common_gaps:
                    st.write("**Lacunas comuns:**")
                    for x in common_gaps:
                        st.write(f"- {x}")

                if teacher_tip:
                    st.write("**Dica de condução da aula:**")
                    st.write(teacher_tip)

            st.caption(f"Avaliado em: {grades.get('evaluated_at','-')} | Total: {grades.get('evaluated_count','-')}")
        else:
            st.info("Nenhuma avaliação salva ainda. Clique em **Avaliar agora** para gerar notas e TOP 3.")
