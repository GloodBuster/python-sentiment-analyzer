from fastapi import FastAPI
from pydantic import BaseModel
from pysentimiento import create_analyzer
from typing import Literal

app = FastAPI()
analyzer = create_analyzer(task="sentiment", lang="es")


class ArrayTextModel(BaseModel):
    texts: list[str]


class Probas(BaseModel):
    NEG: float
    NEU: float
    POS: float


class AnalyzerOutput(BaseModel):
    sentence: str
    probas: Probas
    context: None
    is_multilabel: bool
    output: Literal["NEG", "NEU", "POS"]


@app.post("/analyzer")
def analyze_texts(input_texts: ArrayTextModel) -> list[AnalyzerOutput]:
    predictions = []
    texts = input_texts.texts
    for text in texts:
        result = analyzer.predict(text)
        predictions.append(result)

    return predictions
