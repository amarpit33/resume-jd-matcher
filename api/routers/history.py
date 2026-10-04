"""GET /history: past analyses, to track how a resume's score moves as it's tailored."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_storage
from api.services.storage import Storage

router = APIRouter(tags=["history"])


@router.get("/history")
def list_history(limit: int = Query(100, ge=1, le=1000), storage: Storage = Depends(get_storage)) -> list[dict[str, Any]]:
    """Newest-first summaries: id, created_at, company, role, resume_name and the four scores."""
    return storage.list_analyses(limit)


@router.get("/history/{analysis_id}")
def get_history_item(analysis_id: int, storage: Storage = Depends(get_storage)) -> dict[str, Any]:
    """One stored analysis with its full result."""
    item = storage.get_analysis(analysis_id)
    if item is None:
        raise HTTPException(404, detail={"code": "not_found", "message": "No such analysis."})
    return item
