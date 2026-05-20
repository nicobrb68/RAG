from pydantic import BaseModel

class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class ChunkStorage(BaseModel):
    """permet de lier les sources et le texte en un seule classe"""
    source: MinimalSource
    text_content: str
