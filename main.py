from fastapi import FastAPI
from TransformersTools import TransformersTools


t=TransformersTools()

app=FastAPI()
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/api/cola/{sentece}")
def cola(sentece):
    result=t.cola(sentece)
    return float(result[0])
    
@app.get("/api/sts/{sent1}/{sent2}")
def sts(sent1,sent2):

    return min(5.0,(float(t.sts(sent1,sent2))*5).__round__(3))

@app.get("/api/zero_shot_classification/{sentence}/{tags_comma_sprated}")
def zero_shot_classification(sentence,tags_comma_sprated):
    return t.zero_shot_classification(sentence,tags_comma_sprated)

@app.get("/api/sentiment_analysis/{text}")
def sentiment_analysis(text,text_pair:str=None):
    
    result=t.sentiment_analysis(text,text_pair)
    if result["label"]=="ENTAILMENT":
        return 'good'
    elif result["label"]=="NEUTRAL":
        return 'normal'
    else:
        return 'bad'
@app.get("/api/paraphrase_detection/{sent1}/{sent2}")
def paraprase_detection(sent1,sent2):
    result=t.paraprase_detection(sent1,sent2)
    if result>.6:
        return 'good'
    elif result>.3:
        return 'normal'
    else:
        return "bad"
    
@app.get("/api/clear")
def clear():
    t.clear()
    return {"status":"ok"}

def main():
    import uvicorn

    debug=True
    uvicorn.run(app, host="localhost", port=8087)



    
if __name__ == "__main__":
    main()
