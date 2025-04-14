import os
import sqlite3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("API ключ не знайдено! Перевірте .env файл.")
    exit(1)

def setup_sample_db():
    conn = sqlite3.connect("sample.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            country TEXT,
            age INTEGER
        );
    """)
    customers = [
        (1, "Іван", "Україна", 30),
        (2, "Анна", "Польща", 25),
        (3, "Джон", "США", 35),
        (4, "Олена", "Україна", 28),
    ]
    cursor.executemany("INSERT OR IGNORE INTO customers VALUES (?, ?, ?, ?);", customers)
    conn.commit()
    conn.close()

setup_sample_db()

db = SQLDatabase.from_uri("sqlite:///sample.db")

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    model_name="openai/gpt-4o-mini",
    temperature=0,
)

db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

while True:
    try:
        user_input = input("\nПитай про базу: ")
        if user_input.lower() in ["вихід", "exit", "quit"]:
            break
        result = db_chain.invoke(user_input)

        print("Лог результату:", result)

        if isinstance(result, dict) and 'result' in result:
            result_text = result['result']

            if isinstance(result_text, list) or isinstance(result_text, tuple):
                if len(result_text) > 0 and isinstance(result_text[0], tuple):
                    print(f"Відповідь: {result_text[0][0]}")
                else:
                    print("Запит не повернув результату.")
            else:
                print(f"Відповідь: {result_text}")

        else:
            print("Не вдалося отримати відповідь.")

    except Exception as e:
        print("Сталася помилка:", e)
