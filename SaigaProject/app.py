import gradio as gr
import os
import time
from typing import List, Tuple

# Легковесный кеш для модели
MODEL = None


# Модули для работы с датасетом и метриками
try:
    from ctransformers import AutoModelForCausalLM
except Exception as e:
    AutoModelForCausalLM = None

from datasets import load_dataset
import evaluate


# =========================
# Загрузка модели
# =========================
def load_model(repo_id: str = "IlyaGusev/saiga_mistral_7b_gguf"):
    global MODEL
    if MODEL is not None:
        return MODEL
    if AutoModelForCausalLM is None:
        raise RuntimeError("ctransformers не установлен. Добавьте 'ctransformers' в requirements.txt")

    MODEL = AutoModelForCausalLM.from_pretrained(repo_id)
    return MODEL


# =========================
# Генерация шуток
# =========================
def generate_jokes(num_jokes: int = 3, max_new_tokens: int = 120, temperature: float = 0.8) -> List[str]:
    model = load_model()
    jokes = []
    prompt_template = (
        "Напиши короткую смешную шутку на русском языке. Короткая шутка 1-2 предложения."
    )

    for i in range(max(1, num_jokes)):
        out = model(prompt_template, max_new_tokens=max_new_tokens, temperature=temperature)
        text = "".join(out) if isinstance(out, (list, tuple)) else str(out)
        jokes.append(text.strip())

    return jokes


# =========================
# Вспомогательная функция
# =========================
def extract_last_turn(messages: list) -> tuple[str, str]:
    """
    Из списка сообщений возвращает пару (последний user, следующий bot).
    Если следующего bot нет — возвращает пустую строку.
    """
    last_user_idx = None
    for i, m in enumerate(messages):
        role = (m.get("role") or "").lower()
        if role == "user":
            last_user_idx = i

    if last_user_idx is None:
        # fallback: первый user и первый bot
        user_txt = messages[0].get("content", "") if messages else ""
        assistant_txt = ""
        for m in messages:
            role = (m.get("role") or "").lower()
            if role == "bot":
                assistant_txt = m.get("content", "")
                break
        return user_txt, assistant_txt

    user_txt = messages[last_user_idx].get("content", "")
    assistant_txt = ""
    for j in range(last_user_idx + 1, len(messages)):
        role = (messages[j].get("role") or "").lower()
        if role == "bot":
            assistant_txt = messages[j].get("content", "")
            break

    return user_txt, assistant_txt


# =========================
# Оценка модели
# =========================
def evaluate_on_dataset_semantic(sample_size: int = 50, max_new_tokens: int = 120, temperature: float = 0.0):
    """
    Оценка модели с использованием BERTScore (семантическая близость).
    """
    model = load_model()

    if not os.path.exists("ru_turbo_saiga.jsonl"):
        raise FileNotFoundError("Файл ru_turbo_saiga.jsonl не найден в корне Space!")

    ds = load_dataset("json", data_files="ru_turbo_saiga.jsonl", split="train")
    sample_size = min(sample_size, len(ds))

    bertscore = evaluate.load("bertscore", module_type="metric")  # семантическая оценка
    preds, refs, examples = [], [], []

    for i in range(sample_size):
        item = ds[i]
        messages = item.get("messages", [])
        user_txt, assistant_txt = extract_last_turn(messages)

        if not user_txt:
            continue

        prompt = f"Пользователь: {user_txt}\nАссистент:"
        gen = model(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        gen_text = "".join(gen) if isinstance(gen, (list, tuple)) else str(gen)
        gen_text = gen_text.strip()

        if not gen_text:
            continue

        # нормализация текста
        preds.append(gen_text.lower().strip())
        refs.append((assistant_txt or "").lower().strip())

        if len(examples) < 3:
            examples.append({"user": user_txt, "pred": gen_text, "ref": assistant_txt})

    if len(preds) == 0:
        return {"score": None, "n": 0, "examples": examples}

    # вычисляем BERTScore
    res = bertscore.compute(predictions=preds, references=refs, lang="ru")
    # берем среднее F1
    avg_f1 = sum(res["f1"]) / len(res["f1"]) if res["f1"] else 0.0

    return {
        "score": float(avg_f1) * 100.0,
        "n": len(preds),
        "examples": examples
    }

# =========================
# Обёртки для кнопок
# =========================
def run_generate_only(num_jokes, max_new_tokens, temperature):
    jokes = generate_jokes(num_jokes, max_new_tokens, temperature)
    jokes_text = "\n\n".join([f"{i+1}. {j}" for i, j in enumerate(jokes)])
    return jokes_text


def run_eval_only(max_new_tokens, temperature, eval_samples):
    start = time.time()
    eval_res = evaluate_on_dataset_semantic(eval_samples, max_new_tokens, temperature)
    elapsed = time.time() - start

    if eval_res["score"] is None:
        summary = "Оценка не выполнена."
    else:
        summary = f"ROUGE-L F1: {eval_res['score']:.2f}% on {eval_res['n']} examples"

    examples = []
    for ex in eval_res.get("examples", []):
        examples.append("USER:\n" + ex["user"] + "\n\nPRED:\n" + ex["pred"] + "\n\nREF:\n" + (ex["ref"] or ""))
    examples_text = "\n\n---\n\n".join(examples)

    summary += f"\n\n(Время выполнения: {elapsed:.1f}s)"

    return summary, examples_text


# =========================
# Gradio UI
# =========================
with gr.Blocks(title="Saiga jokes + eval (saiga_mistral_7b_gguf)") as demo:
    gr.Markdown("# Saiga (saiga_mistral_7b_gguf) — генерация шуток и оценка")

    with gr.Row():
        with gr.Column(scale=1):
            num_jokes = gr.Slider(1, 10, value=3, step=1, label="Количество шуток")
            max_tokens = gr.Slider(8, 512, value=120, step=8, label="Макс токенов (max_new_tokens)")
            temperature = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Temperature")
            eval_samples = gr.Slider(1, 200, value=50, step=1, label="Количество примеров для оценки")

            btn_gen = gr.Button("Сгенерировать шутки")
            btn_eval = gr.Button("Оценить модель")

        with gr.Column(scale=1):
            jokes_out = gr.Textbox(label="Сгенерированные шутки", lines=8)
            eval_out = gr.Textbox(label="Результат оценки", lines=3)
            examples_out = gr.Textbox(label="Примеры (user / pred / ref)", lines=12)

    btn_gen.click(
        run_generate_only,
        inputs=[num_jokes, max_tokens, temperature],
        outputs=[jokes_out]
    )

    btn_eval.click(
        run_eval_only,
        inputs=[max_tokens, temperature, eval_samples],
        outputs=[eval_out, examples_out]
    )


if __name__ == "__main__":
    demo.launch(share=True)
