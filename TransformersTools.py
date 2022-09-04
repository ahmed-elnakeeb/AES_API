from transformers import AutoTokenizer, AutoModelForSequenceClassification,pipeline
from sentence_transformers import CrossEncoder
import os
class TransformersTools:
    def __init__(self):
        self.cola_model=None
        self.sts_modle=None
        self.zero_shot_model=None
        self.sentiment_model=None
    
    def sts(self,sent1:str,sent2:str,model_name="cross-encoder/stsb-roberta-large"):
        if not self.sts_modle: 
            self.sts_modle = CrossEncoder('cross-encoder/stsb-roberta-large')
            sts_tok=AutoTokenizer.from_pretrained(model_name)

            #save model to disk if not exist
            if not os.path.exists(model_name):
                self.sts_modle.model.save_pretrained(model_name)
                sts_tok.save_pretrained(model_name)
        return self.sts_modle.predict([(sent1,sent2 )])[0]

    def cola(self,sentece,model_name="howey/roberta-large-cola"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.cola_model: 
            self.cola_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            cola_tok=AutoTokenizer.from_pretrained(model_name)
            #save model to disk if not exist
            if not os.path.exists(model_name):
                self.cola_model.save_pretrained(model_name)
                cola_tok.save_pretrained(model_name)

        inputs = tokenizer(sentece, return_tensors="pt")
        x=float(self.cola_model(**inputs)[0][0][1]).__round__(3)+4
        y=float(self.cola_model(**inputs)[0][0][0]).__round__(3)+4
        return x/8,y/8  
    
    def zero_shot_classification(self,sentence:str,tags_comma_sprated:str ,model_name:str="roberta-large-mnli"):
        if not self.zero_shot_model: 
            self.zero_shot_model = pipeline("zero-shot-classification",model=model_name)
            zero_shot_tok=AutoTokenizer.from_pretrained(model_name)
            #save model to disk if not exist
            if not os.path.exists(model_name):
                self.zero_shot_model.model.save_pretrained(model_name)
                zero_shot_tok.save_pretrained(model_name)
        return self.zero_shot_model([sentence],tags_comma_sprated.split(","))


    def sentiment_analysis(self,text:str,text_pair:str=None,model_name:str="roberta-large-mnli"):
        if not self.sentiment_model: 
            self.sentiment_model = pipeline("sentiment-analysis",model=model_name)
            sentiment_tok=AutoTokenizer.from_pretrained(model_name)
            #save model to disk if not exist
            if not os.path.exists(model_name):
                self.sentiment_model.model.save_pretrained(model_name)
                sentiment_tok.save_pretrained(model_name)
        d={"text": text,"text_pair":text_pair }
        return self.sentiment_model(d)
    

        


    def clear(self):
        self.cola_model=None
        self.sts_modle=None
        self.zero_shot_model=None
        self.sentiment_model=None
        
