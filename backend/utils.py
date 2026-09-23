"""
utils.py — multipart/form-data parsing without the deprecated `cgi` module,
plus small JSON helpers. Pure stdlib.
"""
import json
import uuid
import os


def parse_multipart(body: bytes, content_type: str):
    """
    Minimal but correct multipart/form-data parser.
    Returns (fields: dict[str,str], files: dict[str, {"filename","content_type","data"}])
    """
    fields, files = {}, {}
    if "boundary=" not in content_type:
        return fields, files
    boundary = content_type.split("boundary=")[-1].strip()
    if boundary.startswith('"') and boundary.endswith('"'):
        boundary = boundary[1:-1]
    boundary_bytes = ("--" + boundary).encode("utf-8")

    parts = body.split(boundary_bytes)
    for part in parts:
        if not part or part in (b"--\r\n", b"--"):
            continue
        part = part.strip(b"\r\n")
        if not part:
            continue
        if b"\r\n\r\n" not in part:
            continue
        header_blob, content = part.split(b"\r\n\r\n", 1)
        # strip trailing CRLF before next boundary
        if content.endswith(b"\r\n"):
            content = content[:-2]
        headers_text = header_blob.decode("utf-8", errors="replace")
        name, filename, ctype = None, None, None
        for line in headers_text.split("\r\n"):
            if line.lower().startswith("content-disposition"):
                for chunk in line.split(";"):
                    chunk = chunk.strip()
                    if chunk.startswith("name="):
                        name = chunk[5:].strip('"')
                    elif chunk.startswith("filename="):
                        filename = chunk[9:].strip('"')
            elif line.lower().startswith("content-type"):
                ctype = line.split(":", 1)[1].strip()
        if name is None:
            continue
        if filename is not None and filename != "":
            files[name] = {"filename": filename, "content_type": ctype, "data": content}
        else:
            fields[name] = content.decode("utf-8", errors="replace")
    return fields, files


def save_upload(file_dict, dest_dir, prefix=""):
    os.makedirs(dest_dir, exist_ok=True)
    ext = os.path.splitext(file_dict["filename"] or "")[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"
    fname = f"{prefix}{uuid.uuid4().hex}{ext}"
    path = os.path.join(dest_dir, fname)
    with open(path, "wb") as f:
        f.write(file_dict["data"])
    return path, fname


def json_dumps(obj):
    return json.dumps(obj, default=str)
