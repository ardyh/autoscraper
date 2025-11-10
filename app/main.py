from fastapi import FastAPI, HTTPException
from app.schemas import ParseRequest, ParseResponse
from app.service.parser import parse_with_llm


app = FastAPI(title="MrScraper HTML to JSON Parser", version="0.1.0")


@app.post("/parse", response_model=ParseResponse)
def parse(req: ParseRequest):
    if not req.html or not req.query:
        raise HTTPException(status_code=400, detail="html and query are required")

    data, meta = parse_with_llm(req.html, req.query)
    if data is None:
        raise HTTPException(status_code=422, detail="Could not extract structured data")

    return ParseResponse(data=data, meta=meta)


