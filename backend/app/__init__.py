"""Obscura backend — face detection + redaction pipeline.

No face recognition, no identity matching, no embeddings. Frames are held in
memory only for the duration of a job; inputs and outputs are deleted after
``settings.result_ttl_seconds``.
"""

__version__ = "0.1.0"
