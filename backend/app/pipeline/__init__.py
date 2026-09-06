"""End-to-end redaction pipelines.

Deliverable 5:
    image_pipeline.py - decode -> detect -> redact -> encode
    video_pipeline.py - demux -> (detect every Nth frame + ByteTrack) -> redact
                        -> re-encode -> mux original audio back

Both emit the structured stage logs described in app/logging.py and never
persist raw frames.
"""
