import uvicorn
import logging
import time
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List

import constant
from model import RetrieverModule


logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = RetrieverModule(useRerank=False, K=5, top_return=3)

class Request_Item(BaseModel):
    role: str # phd master udergraduate
    question: str


class Response_Item(BaseModel):
    question: str
    context: List
    paragraph_id: List
    isFAQ: bool
    # execution_time: float


@app.post("/bkheart/api/retrieve")
def retrieve(Request: Request_Item):
    role = Request.role
    question = Request.question

    start_time = time.time()
    context, paragraph_id, isFAQ = model(role, question)

    logging.info('Question: %s', question)
    logging.info('Context: %s', context)
    logging.info('isFAQ: %s', isFAQ)

    return Response_Item(
        question=question,
        context=context,
        paragraph_id=paragraph_id,
        isFAQ=isFAQ,
        # execution_time=round(time.time()-start_time, 4),
    )


if __name__ == '__main__':
    uvicorn.run("app:app", host='0.0.0.0', port=8005)
