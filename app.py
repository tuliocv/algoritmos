# app.py (com diagnóstico forte da API + fallback)
import os
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

import openai  # <-- pra pegar versão
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

# -------------------------
# Persistência
# -------------------------
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
            "evaluation_mode": "natural",
            "debug_mode": True,  # deixa ligado pra você ver tudo
        }
        save_state(state)
        return state

    with open(STATE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)

    state.setdefault("question", DEFAULT_QUESTION)
    state.setdefault("active_question_id", new_question_id())
    state.setdefault("accepting", False)
    state.setdefault("deadline_iso", None)
    state.setdefault("show_top3_to_students", False)
    state.setdefault("live_mode", True)
    state.setdefault("evaluation_mode", "natural")
    state.setdefault("debug_mode", True)
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
                rows.append(json.loads(line))
    return rows

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

# -------------------------
# Admin auth
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
# Prompt / Avaliação
# -------------------------
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
Avalie o uso de estruturas (SE/SENÃO/ENQUANTO/PARA), blocos, clareza e lógica.
Não exija sintaxe de linguagem real.
""".strip()
    else:
        mode_block = """
MODO DE AVALIAÇÃO: LINGUAGEM NATURAL
Avalie clareza, sequência, completude e condições em texto (“se… então…”).
""".strip()

    return f"""
Você é um professor de ALGORITMOS.

{mode_block}

PERGUNTA:
{question}

Retorne APENAS JSON com:
- results[] (score 0-10 + feedback)
- top3[] (3 melhores)
- summary (pontos comuns)

SUBMISSÕES:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()

def normalize_results(grades: Dict[str, Any]) -> Dict[str, Any]:
    grades = grades or {}
    grades.setdefault("results", [])
    grades.setdefault("top3", [])
    grades.setdefault("summary", {"common_strengths": [], "common_gaps": [], "teacher_tip": ""})
    return grades

def run_openai_evaluation(api_key: str, model: str, prompt: str) -> Dict[str, Any]:
    """
    1) Tenta Structured Outputs (json_schema)
    2) Se falhar (SDK antigo ou modelo sem suporte), faz fallback pedindo JSON "no prompt" e parseia.
    """
    client = OpenAI(api_key=api_key)

    schema = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "submission_id": {"type": "string"},
                        "student": {"type": "string"},
                        "score": {"type": "number"},
                        "strengths": {"type": "array", "items": {"type": "string"}},
                        "improvements": {"type": "array", "items": {"type": "string"}},
                        "one_suggestion": {"type": "string"},
                    },
                    "required": ["submission_id", "student", "score", "strengths", "improvements", "one_suggestion"],
                    "additionalProperties": False,
                },
            },
            "top3": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rank": {"type": "number"},
                        "submission_id": {"type": "string"},
                        "student": {"type": "string"},
                        "why_it_wins": {"type": "array", "items": {"type": "string"}},
                        "highlight_excerpt": {"type": "string"},
                    },
                    "required": ["rank", "submission_id", "student", "why_it_wins", "highlight_excerpt"],
                    "additionalProperties": False,
                },
            },
            "summary": {
                "type": "object",
                "properties": {
                    "common_strengths": {"type": "array", "items": {"type": "string"}},
                    "common_gaps": {"type": "array", "items": {"type": "string"}},
                    "teacher_tip": {"type": "string"},
                },
                "required": ["common_strengths", "common_gaps", "teacher_tip"],
                "additionalProperties": False,
            },
        },
        "required": ["results", "top3", "summary"],
        "additionalProperties": False,
    }

    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "grading_result",
                    "schema": schema,
                    "strict": True,
                }
            },
        )
        out = getattr(resp, "output_text", "") or ""
        if not out.strip():
            raise RuntimeError("Resposta vazia do modelo.")
        return json.loads(out)
    except Exception:
        # fallback: pede JSON no prompt e tenta extrair
        resp = client.responses.create(
            model=model,
            input=prompt + "\n\nIMPORTANTE: Retorne APENAS um JSON válido, sem texto extra.",
        )
        out = getattr(resp, "output_text", "") or ""
        if not out.strip():
            raise RuntimeError("Resposta vazia do modelo (fallback).")
        # parse tolerante
        start = out.find("{")
        end = out.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(out[start : end + 1])
        return json.loads(out)

def openai_ping(api_key: str, model: str) -> str:
    client = OpenAI(api_key=api_key)
    resp = client.responses.create(model=model, input="Responda apenas: OK")
    return (getattr(resp, "output_text", "") or "").strip()

# -------------------------
# Coleta: prazo
# -------------------------
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

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")
st.title("🧠 Aprendendo Algoritmos 💻")

state = load_state()
state = maybe_auto_close(state)

current_q = state.get("question") or DEFAULT_QUESTION
qid = state.get("active_question_id") or new_question_id()

tabs = st.tabs(["👩‍🎓 Aluno", "🛠️ Admin"])

# --- ALUNO ---
with tabs[0]:
    if state.get("live_mode", True):
        st_autorefresh(interval=5000, key="live_refresh")

    eval_mode = state.get("evaluation_mode", "natural")
    mode_label = "🗣️ Linguagem natural" if eval_mode == "natural" else "🧩 Pseudocódigo"

    st.subheader("Pergunta do dia")
    st.info(current_q)
    st.caption(f"🔍 Avaliação: **{mode_label}**")

    subs_all = load_submissions()
    subs_current = [s for s in subs_all if s.get("question_id") == qid]
    rem = remaining_seconds(state.get("deadline_iso"))

    c1, c2, c3 = st.columns(3)
    c1.metric("Envios da turma", len(subs_current))
    c2.metric("Status", "Aberto ✅" if state.get("accepting", False) else "Fechado ⛔")
    c3.metric("Tempo restante", "—" if rem is None else f"{rem//60:02d}:{rem%60:02d}")

    if not state.get("accepting", False):
        st.warning("⛔ O envio de respostas está fechado no momento.")
    else:
        with st.form("student_form", clear_on_submit=True):
            student_name = st.text_input("Seu nome (obrigatório)")
            answer = st.text_area("Escreva aqui o seu algoritmo :)", height=220)
            submitted = st.form_submit_button("Enviar resposta", type="primary")

        if submitted:
            student_name = (student_name or "").strip()
            answer = (answer or "").strip()
            if len(student_name) < 2:
                st.error("Digite seu nome.")
            elif len(answer) < 30:
                st.error("Responda com mais detalhes (mín. ~30 caracteres).")
            else:
                append_submission(
                    {
                        "submission_id": f"{int(time.time()*1000)}",
                        "question_id": qid,
                        "question": current_q,
                        "student_name": student_name,
                        "answer": answer,
                        "submitted_at": now_iso(),
                    }
                )
                st.success("✅ Enviado!")

    st.divider()

    if state.get("show_top3_to_students", False):
        grades = load_grades(qid)
        top3 = (grades or {}).get("top3", [])
        if top3:
            st.subheader("🏆 TOP 3 (revelado pelo professor)")
            for item in top3:
                st.markdown(f"**#{item.get('rank','?')} — {item.get('student','-')}**")
                if item.get("highlight_excerpt"):
                    st.code(item["highlight_excerpt"])
        else:
            st.info("TOP 3 ainda não disponível.")

# --- ADMIN ---
with tabs[1]:
    admin_login_ui()
    if not is_admin_logged_in():
        st.info("Entre para gerenciar e avaliar.")
    else:
        subs_all = load_submissions()

        st.subheader("🧭 Rodadas (question_id)")
        if subs_all:
            df_all = pd.DataFrame(subs_all)
            if "question_id" in df_all.columns:
                rounds = (
                    df_all.groupby("question_id")
                    .agg(qtd=("submission_id", "count"), ultima=("submitted_at", "max"))
                    .reset_index()
                    .sort_values("ultima", ascending=False)
                )
                st.dataframe(rounds, use_container_width=True, height=220)

                ids = rounds["question_id"].tolist()
                cur = state.get("active_question_id")
                idx = ids.index(cur) if cur in ids else 0
                selected = st.selectbox("Rodada ativa", options=ids, index=idx)
                if selected != state.get("active_question_id"):
                    state["active_question_id"] = selected
                    save_state(state)
                    st.success("Rodada ativa atualizada.")
                    st.rerun()
        else:
            st.warning("Sem submissões ainda.")

        # refresh qid
        qid = state.get("active_question_id") or new_question_id()
        subs_current = [s for s in subs_all if s.get("question_id") == qid]

        st.divider()
        st.subheader("⚙️ Configurações")

        colA, colB = st.columns([2, 1])
        with colA:
            new_q = st.text_area("Texto da questão", value=state.get("question") or DEFAULT_QUESTION, height=120)
            if st.button("Salvar questão (nova rodada)", type="primary"):
                state["question"] = (new_q.strip() or DEFAULT_QUESTION)
                state["active_question_id"] = new_question_id()
                state["deadline_iso"] = None
                state["accepting"] = False
                state["show_top3_to_students"] = False
                save_state(state)
                st.success("Nova rodada criada.")
                st.rerun()

        with colB:
            state["accepting"] = st.toggle("Aceitar respostas", value=bool(state.get("accepting", False)))
            state["show_top3_to_students"] = st.toggle("Revelar TOP 3 aos alunos", value=bool(state.get("show_top3_to_students", False)))
            state["live_mode"] = st.toggle("Modo Live (auto-refresh aluno)", value=bool(state.get("live_mode", True)))
            state["debug_mode"] = st.toggle("Debug", value=bool(state.get("debug_mode", True)))
            save_state(state)

        st.markdown("### 🧠 Tipo de avaliação")
        mode = st.radio(
            "Avaliar como:",
            options=[("natural","🗣️ Linguagem natural"), ("pseudocode","🧩 Pseudocódigo")],
            format_func=lambda x: x[1],
            index=0 if state.get("evaluation_mode","natural")=="natural" else 1
        )[0]
        if mode != state.get("evaluation_mode"):
            state["evaluation_mode"] = mode
            save_state(state)
            st.success("Modo atualizado.")
            st.rerun()

        st.markdown("### ⏱️ Coleta (fecha automático)")
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            minutes = st.number_input("Minutos", 0, 180, 0, 1)
        with c2:
            if st.button("Iniciar contagem", type="primary"):
                if minutes == 0:
                    state["deadline_iso"] = None
                else:
                    state["deadline_iso"] = (datetime.now(timezone.utc) + timedelta(minutes=int(minutes))).isoformat()
                    state["accepting"] = True
                save_state(state)
                st.rerun()
        with c3:
            rem = remaining_seconds(state.get("deadline_iso"))
            st.info("Sem contagem ativa." if rem is None else f"Tempo restante: **{rem//60:02d}:{rem%60:02d}**")

        st.divider()
        st.subheader("📥 Submissões da rodada ativa")
        st.write(f"Rodada ativa: `{qid}` — Submissões: **{len(subs_current)}**")
        if subs_current:
            df = pd.DataFrame(subs_current)
            st.dataframe(df[["submitted_at","student_name","answer"]], use_container_width=True, height=260)
            st.download_button(
                "⬇️ Baixar CSV (rodada)",
                data=df[["submission_id","question_id","student_name","answer","submitted_at","question"]].to_csv(index=False).encode("utf-8"),
                file_name=f"submissoes_{qid}.csv",
                mime="text/csv",
            )

        st.divider()
        st.subheader("🧪 Diagnóstico da API (faça isso antes de avaliar)")
        api_key = st.text_input("API Key", type="password", placeholder="sk-...", key="openai_api_key")
        model = st.selectbox("Modelo", options=["gpt-5.2", "gpt-5-mini", "gpt-4.1"], index=0)

        if st.button("🧪 Diagnóstico da API", type="primary"):
            st.write("Versão da lib openai:", getattr(openai, "__version__", "desconhecida"))
            st.write("Tamanho da chave:", len(api_key) if api_key else 0)
            try:
                with st.spinner("Pingando o modelo..."):
                    out = openai_ping(api_key=api_key, model=model)
                st.success(f"Ping OK. Resposta: {out}")
            except Exception as e:
                st.error("Falha no ping (isso explica porque não salva). Erro completo:")
                st.exception(e)

        st.divider()
        st.subheader("🤖 Avaliar (gera grades/<qid>.json)")
        if st.button("Avaliar agora", type="primary", disabled=(len(subs_current) == 0)):
            if not api_key or len(api_key) < 10:
                st.error("Informe uma API Key válida.")
            else:
                prompt = build_rubric_prompt(
                    question=state.get("question") or DEFAULT_QUESTION,
                    submissions=subs_current,
                    mode=state.get("evaluation_mode", "natural"),
                )
                try:
                    with st.spinner("Avaliando..."):
                        result = run_openai_evaluation(api_key=api_key, model=model, prompt=prompt)

                    result = normalize_results(result)
                    result["evaluated_at"] = now_iso()
                    result["evaluated_count"] = len(subs_current)
                    result["question_id"] = qid
                    result["evaluation_mode"] = state.get("evaluation_mode", "natural")
                    result["model_used"] = model

                    save_grades(qid, result)
                    st.success(f"✅ Salvo em {grade_path_for(qid)}")
                    st.rerun()
                except Exception as e:
                    st.error("Falha ao avaliar (erro completo abaixo):")
                    st.exception(e)

        st.divider()
        st.subheader("📊 Resultados salvos (rodada ativa)")
        grades = normalize_results(load_grades(qid))
        if grades.get("results"):
            st.markdown("### 🏆 TOP 3")
            for item in grades.get("top3", []):
                st.markdown(f"**#{item.get('rank','?')} — {item.get('student','-')}**")
                if item.get("highlight_excerpt"):
                    st.code(item["highlight_excerpt"])

            st.markdown("### 🧾 Notas")
            df_g = pd.DataFrame(grades.get("results", []))
            st.dataframe(df_g, use_container_width=True, height=260)
            st.download_button(
                "⬇️ Baixar CSV (avaliação)",
                data=df_g.to_csv(index=False).encode("utf-8"),
                file_name=f"avaliacao_{qid}.csv",
                mime="text/csv",
            )
        else:
            st.info("Nenhuma avaliação salva ainda.")

        # DEBUG de arquivos
        if state.get("debug_mode", False):
            st.divider()
            st.subheader("🧪 DEBUG — arquivos e conteúdo salvo")
            expected = grade_path_for(qid)
            st.write("Arquivo esperado:", expected)
            st.write("Existe?", os.path.exists(expected))

            st.write("Listagem data/:")
            for root, _, files in os.walk(DATA_DIR):
                st.write(f"📁 {root}")
                for f in files:
                    st.write(f"  📄 {f}")

            if os.path.exists(expected):
                st.write("Conteúdo JSON salvo:")
                with open(expected, "r", encoding="utf-8") as f:
                    st.json(json.load(f))
