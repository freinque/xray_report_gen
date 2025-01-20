from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from xray_report_gen import report_annotation
from xray_report_gen import utils

utils.set_api_keys()

app = FastAPI()

# Request schema
class ReportsRequest(BaseModel):
    reports: list[str]

@app.post("/report_annotation/")
async def annotate_reports(request: ReportsRequest):

    reports = request.reports

    results = report_annotation.annotate_reports(reports)

    return {"results": results}
