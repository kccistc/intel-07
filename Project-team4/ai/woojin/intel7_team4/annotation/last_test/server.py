# server.py
from flask import Flask, Response, render_template_string, jsonify
from state import AppState

INDEX_HTML = """
<!doctype html>
<html>
<head><meta charset="utf-8"/><title>Processed Stream & Status</title></head>
<body>
  <h2>Processed Stream</h2>
  <img id="stream" src="{{route}}" width="800">
  <pre id="status"></pre>
<script>
async function fetchStatus(){
  try{
    let r = await fetch('/status');
    if(!r.ok) return;
    let j = await r.json();
    document.getElementById('status').textContent = JSON.stringify(j,null,2);
  }catch(e){ console.log(e); }
}
setInterval(fetchStatus, 1000); fetchStatus();
</script>
</body>
</html>
"""

def create_app(app_state: AppState):
    app = Flask(__name__)
    PROC_ROUTE = "/processed"
    @app.route("/")
    def index():
        return render_template_string(INDEX_HTML, route=PROC_ROUTE)
    @app.route(PROC_ROUTE)
    def proc_stream():
        def generator():
            while not app_state.stop_event.is_set():
                if not app_state.frame_event.wait(timeout=1.0):
                    continue
                data = app_state.latest_jpeg
                if not data:
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(data)).encode() + b'\r\n\r\n' + data + b'\r\n')
        return Response(generator(), mimetype='multipart/x-mixed-replace; boundary=--frame')

    @app.route("/status")
    def status():
        with app_state.status_lock:
            return jsonify({
                "face": app_state.current_face_state,
                "slp": app_state.current_slp_state,
                "neck": None if app_state.current_neck_angle is None else float(app_state.current_neck_angle),
                "eye": app_state.current_eye_status,
                "fps": app_state.current_fps
            })
    return app
