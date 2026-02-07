from pathlib import Path
import hashlib,json,re
from datetime import date
from pypdf import PdfReader
#degine the path for read
RAW_dir = Path('data/raw')
TEXT_dir = Path('data/texts')
META_path = Path('data/metadata.jsonl')

TEXT_dir.mkdir(parents=True, exist_ok=True)
RAW_dir.mkdir(parents=True, exist_ok=True)
META_path.parent.mkdir(parents=True, exist_ok=True)

def make_doc_id(file_path: Path) -> str:
    #i give u data will give me back an hash id 
    h = hashlib.sha256()
    # filepath.resolve will transfer the relative path into absolute path
    #avoid use the script at different location get different hash
    # transfer the path to string and encode to bytes
    h.update(str(file_path.resolve()).encode("utf-8"))
    #if the document been change we think they ar new document
    #this wes add timestep into this
    h.update(str(file_path.stat().st_mtime_ns).encode("utf-8"))
    return h.hexdigest()[:16]

#this will match mutiple space or tab
_ws = re.compile(r"[ \t]+")
#this wil compress empty line more than 3 
_nl = re.compile(r"\n{3,}")

# def clean_text(text: str) -> str:
# s = s.replace("\r\n", "\n").replace("\r", "\n")

