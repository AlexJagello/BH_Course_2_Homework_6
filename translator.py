from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langdetect import detect, DetectorFactory

class Translator:
    def __init__(self):
        self.model_name = 'facebook/nllb-200-distilled-600M'
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)



    def detect_language(self, text):
        try:
            DetectorFactory.seed = 0
            detected_language = detect(text)
            return detected_language
        except:
            return "ru"
       


    def get_eng(self, text):
        source_lang = self.detect_language(text)
        if source_lang == "eng":
            return text
        target = 'eng_Latn'
        translator = pipeline('translation', model=self.model, tokenizer=self.tokenizer, src_lang=source_lang, tgt_lang=target, device="cpu")
        output = translator(text, max_length=512)
        translated_text = output[0]['translation_text']
        return translated_text
