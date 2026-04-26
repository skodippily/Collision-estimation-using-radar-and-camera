from fastapi import APIRouter
import shared_state


router = APIRouter()


@router.get("/detections")
async def get_detections():
    return {"success": True, "count": len(shared_state.data_structure), "result": shared_state.data_structure}
