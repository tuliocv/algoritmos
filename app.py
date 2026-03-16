# app.py

import os
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from openai import OpenAI
from streamlit_autorefresh import st_autorefresh

APP_TITLE = "Aprendendo Algoritmos"
DATA_DIR = "data"
STATE_PATH = os.path.join(DATA_DIR, "state.json")
SUBMISSIONS_PATH = os.path.join(DATA_DIR, "submissions.jsonl")
GRADES_DIR = os.path.join(DATA_DIR, "grades")

DEFAULT_QUESTION = (
    "Aguarde a questão para enviar sua resposta.\n\n"
    "Escreva um passo a passo dessas instruções."
)


# =========================
# Persistência
# =========================
def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(GRADES_DIR, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_question_id() -> str:
    return f"q_{int(time.time())}"


def load_state() -> Dict[str, Any]:
    ensure_data_dir()

    if not os.path.exists(STATE_PATH):
        state = {
            "question": DEFAULT_QUESTION,
            "active_question_id": new_question_id(),
            "accepting": False,
            "updated_at": now_iso(),
            "deadline_iso": None,
            "show_top3_to_students": False,
            "live_mode": True,
            "evaluation_mode": "natural",  # natural | pseudocode
        }
        save_state(state)
        return state

    with open(STATE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)

    state.setdefault("active_question_id", new_question_id())
    state.setdefault("deadline_iso", None)
    state.setdefault("show_top3_to_students", False)
    state.setdefault("live_mode", True)
    state.setdefault("evaluation_mode", "natural")
    state.setdefault("accepting", False)
    state.setdefault("question", DEFAULT_QUESTION)

    return state


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
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def clear_submissions(mode: str, question_id: Optional[str] = None) -> None:
    """
    mode:
      - "all": apaga tudo
      - "qid": apaga apenas a rodada
    """
    ensure_data_dir()

    if mode == "all":
        if os.path.exists(SUBMISSIONS_PATH):
            os.remove(SUBMISSIONS_PATH)
        return

    if mode == "qid" and question_id:
        subs = load_submissions()
        kept = [s for s in subs if s.get("question_id") != question_id]
        with open(SUBMISSIONS_PATH, "w", encoding="utf-8") as f:
            for s in kept:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")


def grade_path_for(question_id: str) -> str:
    ensure_data_dir()
    return os.path.join(GRADES_DIR, f"{question_id}.json")


def load_grades(question_id: str) -> Dict[str, Any]:
    p = grade_path_for(question_id)
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_grades(question_id: str, grades: Dict[str, Any]) -> None:
    p = grade_path_for(question_id)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(grades, f, ensure_ascii=False, indent=2)


def clear_grades(question_id: str) -> None:
    p = grade_path_for(question_id)
    if os.path.exists(p):
        os.remove(p)


# =========================
# Auth Admin
# =========================
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
                st.rerun()
            else:
                st.error("Usuário ou senha inválidos.")

    with c2:
        if is_admin_logged_in() and st.button("Sair"):
            st.session_state["admin_authed"] = False
            st.info("Saiu da área admin.")
            st.rerun()


# =========================
# OpenAI / Avaliação
# =========================
def build_rubric_prompt(question: str, submissions: List[Dict[str, Any]], mode: str) -> str:
    payload = []
    for s in submissions:
        payload.append(
            {
                "submission_id": s.get("submission_id", ""),
                "student": (s.get("student_name") or "").strip(),
                "answer": (s.get("answer") or "").strip(),
            }
        )

    if mode == "pseudocode":
        mode_block = """
MODO DE AVALIAÇÃO: PSEUDOCÓDIGO

Avalie considerando:
- Uso adequado de estruturas (SE, SENÃO, ENQUANTO, PARA)
- Clareza da lógica
- Organização da solução
- Completude
- Coerência do algoritmo

Não exija sintaxe perfeita de linguagem específica.
""".strip()
    else:
        mode_block = """
MODO DE AVALIAÇÃO: LINGUAGEM NATURAL

Avalie considerando:
- Clareza do passo a passo
- Ordem lógica
- Uso de decisões em linguagem natural
- Completude
- Facilidade de entendimento
""".strip()

    return f"""
Você é um professor da disciplina de ALGORITMOS E PROGRAMAÇÃO.

{mode_block}

PERGUNTA:
{question}

TAREFA:
Avalie cada resposta com uma nota inteira de 0 a 100.

REGRAS:
- 0 = resposta totalmente inadequada
- 100 = resposta excelente
- Use somente números inteiros
- Avalie todas as submissões

FORMATO DE SAÍDA:
Retorne SOMENTE um JSON válido, sem markdown, sem comentários e sem qualquer texto adicional.

Formato obrigatório:
{{
  "results": [
    {{
      "submission_id": "string",
      "student": "string",
      "score": 0
    }}
  ]
}}

SUBMISSÕES:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def parse_json_safely(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    if not text:
        raise ValueError("Resposta vazia do modelo.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        return json.loads(candidate)

    raise ValueError(f"Não foi possível interpretar JSON. Resposta recebida: {text[:1000]}")


def extract_text_from_response(resp: Any) -> str:
    text = getattr(resp, "output_text", None)
    if text:
        return text

    parts: List[str] = []

    try:
        for item in getattr(resp, "output", []):
            for c in getattr(item, "content", []):
                c_type = getattr(c, "type", "")
                if c_type in ("output_text", "text"):
                    t = getattr(c, "text", "")
                    if t:
                        parts.append(t)
    except Exception:
        pass

    final_text = "\n".join(parts).strip()
    if not final_text:
        raise ValueError("A resposta da OpenAI veio vazia.")
    return final_text


def run_openai_evaluation(api_key: str, model: str, prompt: str) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "Você é um avaliador de respostas de alunos. "
                    "Retorne somente JSON válido. "
                    "Não use markdown. "
                    "Não escreva explicações."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    raw_text = extract_text_from_response(response)
    return parse_json_safely(raw_text)


def normalize_results(grades: Dict[str, Any]) -> Dict[str, Any]:
    grades = grades or {}
    grades.setdefault("results", [])

    normalized = []

    for item in grades.get("results", []):
        try:
            score = int(item.get("score", 0))
        except Exception:
            score = 0

        score = max(0, min(100, score))

        normalized.append(
            {
                "submission_id": item.get("submission_id", ""),
                "student": item.get("student", ""),
                "score": score,
            }
        )

    normalized = sorted(normalized, key=lambda x: x["score"], reverse=True)

    top3 = []
    for i, item in enumerate(normalized[:3], start=1):
        top3.append(
            {
                "rank": i,
                "submission_id": item["submission_id"],
                "student": item["student"],
                "score": item["score"],
            }
        )

    grades["results"] = normalized
    grades["top3"] = top3
    return grades


# =========================
# Prazo / tempo
# =========================
def maybe_auto_close(state: Dict[str, Any]) -> Dict[str, Any]:
    deadline = state.get("deadline_iso")
    if not deadline:
        return state

    try:
        dl = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
        if datetime.now(timezone.utc) >= dl and state.get("accepting", False):
            state["accepting"] = False
            save_state(state)
    except Exception:
        pass

    return state


def remaining_seconds(deadline_iso: Optional[str]) -> Optional[int]:
    if not deadline_iso:
        return None

    try:
        dl = datetime.fromisoformat(deadline_iso.replace("Z", "+00:00"))
        secs = int((dl - datetime.now(timezone.utc)).total_seconds())
        return max(secs, 0)
    except Exception:
        return None


# =========================
# UI
# =========================
st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 1.1rem; }
h1, h2, h3 { letter-spacing: -0.2px; }
div[data-testid="stMetric"] {
  background: rgba(15, 23, 42, 0.06);
  border: 1px solid rgba(15, 23, 42, 0.08);
  padding: 12px;
  border-radius: 14px;
}
.stButton>button {
  border-radius: 14px !important;
  padding: 0.65rem 1.05rem !important;
}
div[data-testid="stAlert"] { border-radius: 14px; }
.badge {
  display:inline-block; padding:6px 10px; border-radius:999px;
  background: rgba(2, 132, 199, 0.12); border: 1px solid rgba(2, 132, 199, 0.25);
  font-weight: 700;
}
.badge-red {
  background: rgba(220, 38, 38, 0.10); border: 1px solid rgba(220, 38, 38, 0.25);
}
.badge-green {
  background: rgba(34, 197, 94, 0.10); border: 1px solid rgba(34, 197, 94, 0.25);
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🧠 Aprendendo Algoritmos 💻")

state = load_state()
state = maybe_auto_close(state)

current_q = state.get("question") or DEFAULT_QUESTION
qid = state.get("active_question_id") or new_question_id()
eval_mode = state.get("evaluation_mode", "natural")
mode_label = "🗣️ Linguagem natural" if eval_mode == "natural" else "🧩 Pseudocódigo"

tabs = st.tabs(["👩‍🎓 Aluno", "🛠️ Admin"])


# =========================
# ALUNO
# =========================
with tabs[0]:
    if state.get("live_mode", True):
        st_autorefresh(interval=5000, key="live_refresh")

    st.subheader("Pergunta do dia")
    st.info(current_q)
    st.caption(f"🔍 Esta atividade será avaliada como: **{mode_label}**")

    subs_all = load_submissions()
    subs_current = [s for s in subs_all if s.get("question_id") == qid]
    rem = remaining_seconds(state.get("deadline_iso"))

    c1, c2, c3 = st.columns(3)
    c1.metric("Envios da turma", len(subs_current))
    c2.metric("Status", "Aberto ✅" if state.get("accepting", False) else "Fechado ⛔")

    if rem is None:
        c3.metric("Tempo restante", "—")
    else:
        mins = rem // 60
        secs = rem % 60
        c3.metric("Tempo restante", f"{mins:02d}:{secs:02d}")

    st.markdown(
        f"""
<div style="margin-top:6px; margin-bottom:14px;">
  <span class="badge {'badge-green' if state.get('accepting', False) else 'badge-red'}">
    {'Coleta aberta — envie sua resposta' if state.get('accepting', False) else 'Coleta fechada — aguarde o professor'}
  </span>
</div>
""",
        unsafe_allow_html=True,
    )

    if not state.get("accepting", False):
        st.warning("⛔ O envio de respostas está fechado no momento.")
    else:
        with st.form("student_form", clear_on_submit=True):
            student_name = st.text_input("Seu nome (obrigatório)")
            answer = st.text_area(
                "Escreva aqui o seu algoritmo :)",
                height=220,
                placeholder="Ex.: 1) Faça isso... 2) Depois faça aquilo... 3) Se acontecer X, então faça Y...",
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
                    "submission_id": f"{int(time.time() * 1000)}",
                    "question_id": qid,
                    "question": current_q,
                    "student_name": student_name,
                    "answer": answer,
                    "submitted_at": now_iso(),
                }
                append_submission(entry)
                st.success("✅ Resposta registrada! Obrigado 🙂")
                st.rerun()

    st.divider()
    st.caption("Dica: passos curtos, ordem clara e use condições do tipo “se... então...” quando fizer sentido.")

    if state.get("show_top3_to_students", False):
        grades = normalize_results(load_grades(qid))
        top3 = grades.get("top3", [])

        if top3:
            st.subheader("🏆 TOP 3 da turma")
            for item in top3:
                st.markdown(f"**#{item['rank']} — {item['student']}**  \nNota: **{item['score']}**")
        else:
            st.info("TOP 3 ainda não disponível.")


# =========================
# ADMIN
# =========================
with tabs[1]:
    admin_login_ui()

    if not is_admin_logged_in():
        st.info("Entre para gerenciar a questão e avaliar respostas.")
    else:
        st.subheader("⚙️ Configurações da Questão")

        colA, colB = st.columns([2, 1])

        with colA:
            new_q = st.text_area("Texto da questão", value=current_q, height=140)
            if st.button("Salvar questão (nova rodada)", type="primary"):
                state["question"] = (new_q.strip() or DEFAULT_QUESTION)
                state["active_question_id"] = new_question_id()
                state["show_top3_to_students"] = False
                state["deadline_iso"] = None
                state["accepting"] = False
                save_state(state)
                st.success("Questão atualizada e nova rodada criada.")
                st.rerun()

        with colB:
            accepting = st.toggle("Aceitar novas respostas", value=bool(state.get("accepting", False)))
            if accepting != bool(state.get("accepting", False)):
                state["accepting"] = accepting
                if accepting and remaining_seconds(state.get("deadline_iso")) == 0:
                    state["deadline_iso"] = None
                save_state(state)
                st.success("Status de recebimento atualizado.")
                st.rerun()

            show_top3 = st.toggle("Revelar TOP 3 para alunos", value=bool(state.get("show_top3_to_students", False)))
            if show_top3 != bool(state.get("show_top3_to_students", False)):
                state["show_top3_to_students"] = show_top3
                save_state(state)
                st.success("Configuração de revelação atualizada.")
                st.rerun()

            live_mode = st.toggle("Modo Live (auto-refresh no aluno)", value=bool(state.get("live_mode", True)))
            if live_mode != bool(state.get("live_mode", True)):
                state["live_mode"] = live_mode
                save_state(state)
                st.success("Modo Live atualizado.")
                st.rerun()

            st.caption(f"Última atualização: {state.get('updated_at', '-')}")

        st.markdown("### 🧠 Tipo de avaliação do algoritmo")

        mode_options = {
            "natural": "🗣️ Linguagem natural",
            "pseudocode": "🧩 Pseudocódigo",
        }

        selected_mode = st.radio(
            "Avaliar respostas como:",
            options=list(mode_options.keys()),
            format_func=lambda x: mode_options[x],
            index=0 if state.get("evaluation_mode", "natural") == "natural" else 1,
        )

        if selected_mode != state.get("evaluation_mode", "natural"):
            state["evaluation_mode"] = selected_mode
            save_state(state)
            st.success("Modo de avaliação atualizado.")
            st.rerun()

        st.markdown("### ⏱️ Tempo de coleta")
        cc1, cc2, cc3 = st.columns([1, 1, 2])

        with cc1:
            minutes = st.number_input("Minutos", min_value=0, max_value=180, value=0, step=1)

        with cc2:
            if st.button("Iniciar contagem", type="primary"):
                if minutes == 0:
                    state["deadline_iso"] = None
                    st.success("Sem tempo definido.")
                else:
                    dl = datetime.now(timezone.utc) + timedelta(minutes=int(minutes))
                    state["deadline_iso"] = dl.isoformat()
                    state["accepting"] = True
                    st.success(f"Coleta aberta por {minutes} min.")
                save_state(state)
                st.rerun()

        with cc3:
            rem = remaining_seconds(state.get("deadline_iso"))
            if rem is None:
                st.info("Sem contagem ativa.")
            else:
                mins = rem // 60
                secs = rem % 60
                st.info(f"Tempo restante atual: **{mins:02d}:{secs:02d}**")

        st.divider()
        st.subheader("📥 Submissões (rodada atual)")

        qid = state.get("active_question_id") or new_question_id()
        subs_all = load_submissions()
        subs_current = [s for s in subs_all if s.get("question_id") == qid]

        m1, m2, m3 = st.columns(3)
        m1.metric("Total (todas as rodadas)", len(subs_all))
        m2.metric("Da rodada atual", len(subs_current))
        m3.metric("Recebendo agora?", "Sim" if state.get("accepting", False) else "Não")

        st.markdown("### 🧹 Limpeza")
        lc1, lc2, lc3 = st.columns([1, 1, 2])

        with lc1:
            if st.button("Limpar rodada atual", disabled=(len(subs_current) == 0)):
                clear_submissions(mode="qid", question_id=qid)
                clear_grades(qid)
                st.success("Submissões e resultados da rodada atual apagados.")
                st.rerun()

        with lc2:
            if st.button("Limpar TUDO"):
                clear_submissions(mode="all")
                st.success("Todas as submissões foram apagadas.")
                st.rerun()

        with lc3:
            st.caption("Isso apaga respostas do arquivo local. Use com cuidado.")

        df = pd.DataFrame(subs_current) if subs_current else pd.DataFrame(
            columns=["submission_id", "student_name", "answer", "submitted_at"]
        )

        if not df.empty:
            st.dataframe(df[["submitted_at", "student_name", "answer"]], use_container_width=True, height=320)
            csv = df[["submission_id", "question_id", "student_name", "answer", "submitted_at", "question"]].to_csv(
                index=False
            ).encode("utf-8")
            st.download_button(
                "⬇️ Baixar CSV (rodada atual)",
                data=csv,
                file_name="submissoes_rodada_atual.csv",
                mime="text/csv",
            )
        else:
            st.warning("Ainda não há submissões para a rodada atual.")

        st.divider()
        st.subheader("🤖 Avaliar respostas (rodada atual)")

        default_api_key = st.secrets.get("OPENAI_API_KEY", "")
        api_key = st.text_input(
            "API Key",
            type="password",
            value=default_api_key,
            placeholder="sk-...",
            key="openai_api_key",
        )

        colm1, colm2, colm3 = st.columns([1, 1, 2])

        with colm1:
            model = st.selectbox(
                "Modelo",
                options=["gpt-5-mini", "gpt-4.1-mini", "gpt-4.1"],
                index=0,
            )

        with colm2:
            st.caption(" ")
            st.caption(f"Modo: **{mode_label}**")

        with colm3:
            st.caption("Clique em avaliar para gerar o ranking.")

        btn_col1, btn_col2 = st.columns([1, 1])

        with btn_col1:
            if st.button("Avaliar agora", type="primary", disabled=(len(subs_current) == 0)):
                if not api_key or len(api_key) < 10:
                    st.error("Informe uma API Key válida.")
                else:
                    prompt = build_rubric_prompt(
                        question=current_q,
                        submissions=subs_current,
                        mode=state.get("evaluation_mode", "natural"),
                    )

                    try:
                        with st.spinner("Avaliando..."):
                            result = run_openai_evaluation(
                                api_key=api_key,
                                model=model,
                                prompt=prompt,
                            )

                        result = normalize_results(result)
                        result["evaluated_at"] = now_iso()
                        result["evaluated_count"] = len(subs_current)
                        result["question_id"] = qid
                        result["question_text"] = current_q
                        result["evaluation_mode"] = state.get("evaluation_mode", "natural")

                        save_grades(qid, result)
                        st.success("✅ Avaliação concluída com sucesso.")
                        st.rerun()

                    except Exception as e:
                        st.exception(e)

        with btn_col2:
            if st.button("Apagar resultados (rodada atual)"):
                clear_grades(qid)
                st.success("Resultados apagados.")
                st.rerun()

        st.divider()
        st.subheader("📊 Resultados salvos (rodada atual)")

        grades = normalize_results(load_grades(qid))

        if grades and grades.get("results"):
            df_g = pd.DataFrame(grades["results"])

            if not df_g.empty:
                df_g = df_g[["student", "score"]].copy()
                df_g.columns = ["Aluno", "Nota"]
                df_g = df_g.sort_values(by="Nota", ascending=False).reset_index(drop=True)

                st.markdown("### 🧾 Ranking da turma")
                st.dataframe(df_g, use_container_width=True, height=400)

                csv_g = df_g.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Baixar CSV (ranking da turma)",
                    data=csv_g,
                    file_name="ranking_turma.csv",
                    mime="text/csv",
                )

            top3 = grades.get("top3", [])
            if top3:
                st.markdown("### 🏆 TOP 3")
                for item in top3:
                    st.markdown(
                        f"**#{item['rank']} — {item['student']}**  \nNota: **{item['score']}**"
                    )

            st.caption(
                f"Avaliado em: {grades.get('evaluated_at', '-')} | "
                f"Total: {grades.get('evaluated_count', '-')} | "
                f"Modo: {grades.get('evaluation_mode', '-')} | Rodada: {qid}"
            )
        else:
            st.info("Nenhuma avaliação salva ainda. Clique em **Avaliar agora** para gerar as notas.")
