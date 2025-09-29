import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.fsm.storage.memory import MemoryStorage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from rag import get_or_create_vectorstore
from db import init_db, get_context, set_context, clear_context
from config import BOT_TOKEN, OPENROUTER_API_KEY, SYSTEM_PROMPT

# Настройка логгера
logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

# Клавиатура
keyboard = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="🧹 Очистить контекст")]],
    resize_keyboard=True,
    one_time_keyboard=False
)

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Привет! Я — ваш  консультант по снам.\n\n"
        "Нажмите кнопку ниже, чтобы очистить историю диалога.",
        reply_markup=keyboard
    )

@dp.message(lambda msg: msg.text == "🧹 Очистить контекст")
async def clear_ctx(message: types.Message):
    await clear_context(message.chat.id)
    await message.answer("🧹 Контекст очищен!", reply_markup=keyboard)

@dp.message()
async def handle_message(message: types.Message):
    if not message.text or not message.text.strip():
        await message.answer("Пожалуйста, отправьте текстовое сообщение.", reply_markup=keyboard)
        return

    user_query = message.text.strip()
    if user_query == "🧹 Очистить контекст":
        return  # уже обработано

    try:
        # Получаем контекст из БД
        chat_context = await get_context(message.chat.id)

        # Получаем актуальную векторную БД (с RAG)
        vectorstore = get_or_create_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # LLM через OpenRouter (только для генерации)
        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            model="deepseek/deepseek-chat-v3.1:free",
            temperature=0.3,
            max_tokens=1000
        )

        # ⚠️ ВАЖНО: промт ДОЛЖЕН содержать переменную {context}
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT + "\n\nИспользуй ТОЛЬКО следующую информацию из документов:\n{context}"),
            ("human", "История диалога (если есть):\n{chat_history}\n\nВопрос пользователя:\n{input}")
        ])

        # Создаём цепочку RAG
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

        # Выполняем запрос
        response = await retrieval_chain.ainvoke({
            "input": user_query,
            "chat_history": chat_context
        })

        answer = response["answer"]

        # Обновляем контекст диалога
        new_context = f"{chat_context}\nПользователь: {user_query}\nАссистент: {answer}".strip()
        await set_context(message.chat.id, new_context)

        await message.answer(answer, reply_markup=keyboard)

    except Exception as e:
        logging.exception("Ошибка при обработке сообщения")
        await message.answer(
            "Произошла ошибка при обработке запроса. Попробуйте позже.",
            reply_markup=keyboard
        )

async def main():
    await init_db()
    # Инициализируем векторную БД при старте
    get_or_create_vectorstore()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())