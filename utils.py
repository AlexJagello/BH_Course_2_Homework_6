from transformers import RobertaForSequenceClassification, RobertaTokenizer


class RobertaModel():
    def __init__(self):
        self.model = RobertaForSequenceClassification.from_pretrained("./roberta_snli_model")
        self.tokenizer = RobertaTokenizer.from_pretrained("./roberta_snli_model")

    def predict(self, premise, hypothesis):
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        predicted_class = outputs.logits.argmax().item()
        labels = ["entailment", "neutral", "contradiction"]
        return labels[predicted_class]

    def find_contradiction_text(self, splited_text):
        contradictions = []
        for i in range(len(splited_text)):
            for j in range(i + 1, len(splited_text)):
                predicted = self.predict(splited_text[i], splited_text[j])
                if(predicted == "contradiction"):
                    contradictions.append((splited_text[i], splited_text[j]))
        return contradictions

            
