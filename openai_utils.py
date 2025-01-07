import openai

def answer_question_with_chat_gpt(question, context, openai_api_key, model="gpt-3.5-turbo"):
    openai.api_key = openai_api_key
    system_message = {
        "role": "system",
        "content": (
            "You are a helpful assistant. Use the provided context to answer the question. "
            "If the answer isn't in the context, say: 'I'm not sure based on the document.'"
        )
    }
    user_message = {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}"
    }
    response = openai.ChatCompletion.create(
        model=model,
        messages=[system_message, user_message],
        max_tokens=300,
        temperature=0.0
    )
    return response["choices"][0]["message"]["content"].strip()
