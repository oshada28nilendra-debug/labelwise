from pathlib import Path
from flask import Flask,jsonify,render_template,request
from werkzeug.exceptions import HTTPException
from .core import parse_csv,benchmark,train_and_queue

SAMPLE=Path(__file__).parent/"sample"
def create_app():
    app=Flask(__name__);app.config["MAX_CONTENT_LENGTH"]=16*1024*1024
    @app.before_request
    def guard():
        if request.host.split(":")[0] not in {"localhost","127.0.0.1"}:return jsonify(error="Use localhost or 127.0.0.1."),403
        if request.method=="POST" and (request.headers.get("X-LabelWise")!="local" or (request.headers.get("Origin") and request.headers["Origin"] != request.host_url.rstrip("/"))):return jsonify(error="Local requests only."),403
    @app.after_request
    def headers(response):
        response.headers["Content-Security-Policy"]="default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; frame-ancestors 'none'"
        response.headers["Cache-Control"]="no-store";response.headers["X-Content-Type-Options"]="nosniff";return response
    @app.errorhandler(ValueError)
    def invalid(exc):return jsonify(error=str(exc)),400
    @app.errorhandler(HTTPException)
    def http(exc):return jsonify(error=exc.description),exc.code
    @app.get("/")
    def home():return render_template("index.html")
    def raw(name,sample_name):
        if request.form.get("sample")=="1":return (SAMPLE/sample_name).read_bytes()
        f=request.files.get(name)
        if not f:raise ValueError(f"Upload {name} CSV.")
        return f.read()
    @app.post("/api/queue")
    def queue():
        try:size=int(request.form.get("size",10))
        except ValueError as exc:raise ValueError("Queue size must be an integer.") from exc
        result=train_and_queue(parse_csv(raw("seed","seed.csv")),parse_csv(raw("pool","pool-unlabelled.csv"),False),size)
        result["dataset"]="Synthetic demo" if request.form.get("sample")=="1" else "User supplied";return jsonify(result)
    @app.post("/api/benchmark")
    def run():
        result=benchmark(parse_csv(raw("seed","seed.csv")),parse_csv(raw("pool","pool-labelled.csv")),parse_csv(raw("test","test.csv")),[0,5,10,20,40],20,42)
        result["dataset"]="Synthetic seeded demo" if request.form.get("sample")=="1" else "User supplied";return jsonify(result)
    return app
