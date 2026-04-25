from fastapi import APIRouter
from main_testcode import data_structure


router = APIRouter()


@router.get("/detections")
async def get_detections():
    return {"success": True, "count": len(data_structure), "result": data_structure}
