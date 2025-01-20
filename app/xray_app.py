from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from xray_report_gen import report_annotation
from xray_report_gen import utils

utils.set_api_keys()

app = FastAPI()

# Request schema
class ReportsRequest(BaseModel):
    reports: list[str]

@app.post("/report_separation/")
async def categorize_reports(request: ReportsRequest):

    reports = request.reports

    results = report_separation.separate_reports(reports)

    return {"results": results}
