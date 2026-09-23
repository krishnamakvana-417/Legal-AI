# AI Legal Platform — Build Tasks

## Root / Config
- [ ] `.env.example`
- [ ] `requirements.txt`
- [ ] `backend/config.py`
- [ ] `backend/models.py`

## Ingestion Layer
- [ ] `ingestion/__init__.py`
- [ ] `ingestion/drive_fetcher.py`
- [ ] `ingestion/parser.py`

## Vector Store
- [ ] `vector_store/__init__.py`
- [ ] `vector_store/indexer.py`

## Citation Graph
- [ ] `graph/__init__.py`
- [ ] `graph/citation_network.py`

## API Connectors
- [ ] `api_connectors/__init__.py`
- [ ] `api_connectors/kanoon_client.py`
- [ ] `api_connectors/anrak_client.py`

## RAG Summarizer
- [ ] `rag/__init__.py`
- [ ] `rag/summarizer.py`

## FastAPI Backend
- [ ] `backend/main.py`

## Next.js Frontend
- [ ] Initialize Next.js project
- [ ] `frontend/app/layout.tsx`
- [ ] `frontend/app/page.tsx` (Dashboard)
- [ ] `frontend/app/search/page.tsx`
- [ ] `frontend/app/document/[id]/page.tsx`
- [ ] `frontend/app/graph/page.tsx`
- [ ] `frontend/app/external/page.tsx`
- [ ] `frontend/components/Navbar.tsx`
- [ ] `frontend/components/SearchBar.tsx`
- [ ] `frontend/components/ResultCard.tsx`
- [ ] `frontend/components/SummaryViewer.tsx`
- [ ] `frontend/components/CitationGraph.tsx`
- [ ] `frontend/components/IngestionPanel.tsx`
- [ ] `frontend/lib/api.ts`
