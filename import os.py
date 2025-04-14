import os
from dotenv import load_dotenv
from typing import Dict, Optional
from llama_index.core.llms import ChatMessage
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.memory import ChatMemoryBuffer

load_dotenv()

class ProgrammingChatBot:
    def __init__(self):
        self.llm = OpenRouter(
            model="anthropic/claude-3-haiku",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3
        )
        
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        
        self.qa_pairs: Dict[str, str] = {
            "що таке python": "Python - це інтерпретована, об'єктно-орієнтована мова програмування...",
            "як оголосити змінну": "У Python: x = 5, у JavaScript: let x = 5, у C++: int x = 5;",
        }
        
        self.system_prompt = """Ти експертний помічник з програмування. Надавай точні, технічні відповіді.
Якщо питання стосується базових тем (синтаксис, ООП, алгоритми), дай стислу відповідь з прикладом коду.
У незрозумілих випадках уточни деталі."""

    def get_predefined_answer(self, question: str) -> Optional[str]:
        q_lower = question.lower()
        for q, a in self.qa_pairs.items():
            if q in q_lower:
                return a
        return None

    def chat(self):
        print("Бот: Привіт! Я чат-бот з програмування. Задавайте питання (або 'вихід' для завершення)")
        
        while True:
            user_input = input("Ви: ").strip()
            
            if user_input.lower() in ['вихід', 'exit', 'quit']:
                print("Бот: Гарного кодування! Звертайтеся ще.")
                break
                
            self._process_message(user_input)

    def _process_message(self, user_input: str):
        try:
            predefined_answer = self.get_predefined_answer(user_input)
            if predefined_answer:
                print(f"Бот: {predefined_answer}")
                self.memory.put(ChatMessage(role="assistant", content=predefined_answer))
                return
                
            self.memory.put(ChatMessage(role="user", content=user_input))
            
            messages = [
                ChatMessage(role="system", content=self.system_prompt),
                *self.memory.get()
            ]
            
            response = self.llm.chat(messages)
            bot_response = response.message.content
            
            print(f"Бот: {bot_response}")
            self.memory.put(ChatMessage(role="assistant", content=bot_response))
            
        except Exception as e:
            print(f"Помилка: {str(e)}")
            self.memory.reset()

if __name__ == "__main__":
    bot = ProgrammingChatBot()
    bot.chat()