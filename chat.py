import logging
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    Application,
    MessageHandler,
    CommandHandler,
    filters,
    ContextTypes,
    ConversationHandler,
)
import translator
import utils
import datetime



# Настройка логирования
logging.basicConfig(filename='chat_bot.log', encoding='utf-8',
    format="%(asctime)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

translat=translator.Translator()
roberta=utils.RobertaModel()

SELECT_MODE, FIRST_PHRASE, SECOND_PHRASE, TEXT_ANALYSIS = range(4)

# Клавиатура для выбора режима
mode_keyboard = [["🔍 Сравнить две фразы", "📝 Анализировать текст"]]

class UserData:
    def __init__(self):
        self.mode = None
        self.first_phrase = None
        self.second_phrase = None
        self.text = None

# Хранилище данных пользователей
users = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.message.from_user.id
    users[user_id] = UserData()
    
    await update.message.reply_text(
        "Выберите режим работы:",
        reply_markup=ReplyKeyboardMarkup(mode_keyboard, one_time_keyboard=True),
    )
    return SELECT_MODE

async def select_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.message.from_user.id
    user_data = users.get(user_id)
    
    if update.message.text == "🔍 Сравнить две фразы":
        user_data.mode = "phrases"
        await update.message.reply_text(
            "Включен режим сравнения фраз. Пришлите первую фразу:",
            reply_markup=None,
        )
        return FIRST_PHRASE
    
    elif update.message.text == "📝 Анализировать текст":
        user_data.mode = "text"
        await update.message.reply_text(
            "Включен режим анализа текста. Пришлите текст для проверки:",
            reply_markup=None,
        )
        return TEXT_ANALYSIS
    
    else:
        await update.message.reply_text("Пожалуйста, выберите режим из предложенных.")
        return SELECT_MODE

async def first_phrase(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.message.from_user.id
    users[user_id].first_phrase = update.message.text
    
    await update.message.reply_text("Теперь пришлите вторую фразу:")
    return SECOND_PHRASE

async def second_phrase(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.message.from_user.id
    user_data = users[user_id]
    user_data.second_phrase = update.message.text
    
    first_translated=translat.get_eng(user_data.first_phrase)
    second_translated=translat.get_eng(user_data.second_phrase)

    result=roberta.predict(first_translated, second_translated)

    result = "🔹 Результат проверки двух фраз:\n" \
             f"Первое сообщение: {user_data.first_phrase}\n"\
             f"Второе сообщение: {user_data.second_phrase}\n\n"\
             f"Первое переведенное сообщение: {first_translated}\n"\
             f"Второе переведенное сообщение: {second_translated}\n\n"\
             f"{result}"
    
    await update.message.reply_text(result, parse_mode="Markdown")

    logger.info(user_id.__str__() +" > " + datetime.datetime.now().__str__() +" > "+result)
    return SELECT_MODE

async def text_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.message.from_user.id
    text = update.message.text
    
    sentences = text.split('.')  
    translated_sentences = []
    for sent in sentences:
        striped = sent.strip()
        if striped != '':
            translated=translat.get_eng(striped)
            translated_sentences.append(translated)
    

    results = roberta.find_contradiction_text(translated_sentences)
    analysis = "🔹 Анализ текста:\n\n" 
    analysis += "Переведенные предложения:\n"
    for item in translated_sentences:
        analysis += f"{item}\n" 

    analysis += "Найденные противоречия:\n"
    for item in results:
        analysis += f"{item[0]} <--> {item[1]}\n\n"
    
    await update.message.reply_text(analysis, parse_mode="Markdown")

    logger.info(user_id.__str__() +">" + datetime.datetime.now().__str__() +">"+analysis)
    return SELECT_MODE

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.message.from_user.id
    if user_id in users:
        del users[user_id]
    
    await update.message.reply_text(
        "Диалог прерван. Нажмите /start для начала.",
        reply_markup=None,
    )
    return ConversationHandler.END

def main() -> None:
    application = Application.builder().token("SecretKey").build()
    
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={       
            SELECT_MODE: [MessageHandler(filters.TEXT & ~filters.COMMAND, select_mode)],
            FIRST_PHRASE: [MessageHandler(filters.TEXT & ~filters.COMMAND, first_phrase)],
            SECOND_PHRASE: [MessageHandler(filters.TEXT & ~filters.COMMAND, second_phrase)],
            TEXT_ANALYSIS: [MessageHandler(filters.TEXT & ~filters.COMMAND, text_analysis)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    
    application.add_handler(conv_handler)
    application.run_polling()

if __name__ == "__main__":
    main()


