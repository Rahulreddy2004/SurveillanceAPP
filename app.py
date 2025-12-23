import eventlet
eventlet.monkey_patch() # MUST be first
import threading
from flask import Flask, render_template, Response, request, redirect, url_for, flash, session, jsonify
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from twilio.rest import Client
import cv2
from ultralytics import YOLO
import time
import os
import datetime
import torch
import traceback
from sqlalchemy import func, cast, Date
from datetime import timedelta
from collections import deque
import json # For loading/saving settings
import re # --- NEW --- Import regex for CSS parsing

# Flask-WTF Imports
from flask_wtf import FlaskForm
from wtforms import FloatField, IntegerField, StringField, SubmitField, BooleanField, PasswordField
from wtforms.validators import DataRequired, NumberRange, Email, Optional, Length
from functools import wraps # For login_required decorator

app = Flask(__name__)

# --- Configurations ---

# --- Initialize Extensions ---
socketio = SocketIO(app, cors_allowed_origins="*")
db = SQLAlchemy(app)
mail = Mail(app)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# --- Global ROI variable ---
current_roi = None

# --- Dummy User (Used for basic login) ---
DUMMY_USER = {"user": "admin", "pass": "12345"}

# --- Settings Loading/Saving ---
SETTINGS_FILE = os.path.join(app.root_path, 'settings.json')
app_settings = {} # Global dict to hold settings

def load_settings():
    global app_settings
    defaults = {
        "confidence_threshold": 0.80,
        "alert_cooldown_seconds": 3,
        "recipient_email": "",
        "recipient_phone": "",
        "enable_email_alerts": True,
        "enable_sms_alerts": True,
        "enable_beep_sound": True
    }
    try:
        with open(SETTINGS_FILE, 'r') as f: app_settings = json.load(f)
        updated = False
        for key, value in defaults.items():
            if key not in app_settings: app_settings[key] = value; updated = True
        if updated: save_settings()
        print("--- Settings loaded successfully ---")
    except FileNotFoundError:
        print("--- WARNING: settings.json not found. Using default values. ---"); app_settings = defaults.copy(); save_settings()
    except json.JSONDecodeError:
        print("!!! ERROR: Could not decode settings.json. Using defaults. !!!"); app_settings = defaults.copy()
    except Exception as e:
        print(f"!!! ERROR loading settings: {e}. Using defaults. !!!"); app_settings = defaults.copy()

def save_settings():
    try:
        with open(SETTINGS_FILE, 'w') as f: json.dump(app_settings, f, indent=4)
        print("--- Settings saved successfully ---")
    except Exception as e: print(f"!!! ERROR saving settings: {e} !!!")

load_settings() # Load settings on app start
# -----------------------------

# --- NEW: Load CSS Variables for Charts ---
css_variables = {}
def load_css_variables():
    global css_variables
    try:
        css_path = os.path.join(app.root_path, 'static', 'css', 'style.css')
        with open(css_path, 'r') as f:
            content = f.read()
        
        # Regex to find all CSS variables in the :root section
        matches = re.findall(r'(--[\w-]+):\s*([^;]+);', content)
        css_variables = {name: value.strip() for name, value in matches}
        if css_variables:
            print(f"--- Loaded {len(css_variables)} CSS variables for charts ---")
        else:
             print("--- WARNING: Could not find CSS variables in style.css ---")
    except Exception as e:
        print(f"!!! ERROR loading CSS variables: {e} !!!")

load_css_variables() # --- NEW --- Load CSS variables on app start
# ----------------------------------------


# --- Load YOLO Models ---
try:
    MODELS = { "violence": YOLO("violence_model.pt"), "fall": YOLO("fall_model.pt") }
    print("Successfully loaded 'violence' and 'fall' models.")
except Exception as e:
    print(f"Error loading YOLO models: {e}"); MODELS = {}

# --- Database Model ---
class Alert(db.Model):
    __tablename__ = 'alerts'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.datetime.utcnow)
    detection_type = db.Column(db.String(50), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    image_filename = db.Column(db.String(255), nullable=False)
    # --- NEW --- Added video_filename column
    video_filename = db.Column(db.String(255), nullable=True)


# --- Notification Functions ---
def send_email_alert(alert_type, confidence, image_filename):
    if not app_settings.get('enable_email_alerts', False): print("--- Email alert skipped: Disabled in settings. ---"); return
    recipient = app_settings.get('recipient_email')
    if not recipient: print("--- Email alert skipped: No recipient email configured. ---"); return
    try:
        subject = f"Security Alert: {alert_type} Detected!"; body = (f"Event: {alert_type} ({confidence * 100:.0f}% conf) at {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC. Image: {image_filename}")
        msg = Message(subject, recipients=[recipient], body=body); mail.send(msg); print(f"--- Email alert sent to {recipient} ---")
    except Exception as e: print(f"!!! ERROR sending email: {e} !!!")

def send_sms_alert(alert_type, confidence):
    if not app_settings.get('enable_sms_alerts', False): print("--- SMS alert skipped: Disabled in settings. ---"); return
    recipient = app_settings.get('recipient_phone')
    if not recipient: print("--- SMS alert skipped: No recipient phone configured. ---"); return
    try:
        body = (f"ALERT: {alert_type} ({confidence * 100:.0f}%) detected at {datetime.datetime.utcnow().strftime('%H:%M:%S')} UTC.")
        message = twilio_client.messages.create(body=body, from_=TWILIO_PHONE_NUMBER, to=recipient)
        print(f"--- SMS alert sent to {recipient} (SID: {message.sid}) ---")
    except Exception as e: print(f"!!! ERROR sending SMS: {e} !!!")

# --- DB Logging & Notification Function ---
# --- MODIFIED --- This function is now just for saving (notifications happen in background)
def log_alert_to_db(detection_type, confidence, frame_to_save):
    filename = None
    try:
        now = datetime.datetime.now()
        filename = f"{detection_type}_snapshot_{now.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(image_path, frame_to_save)
        
        # --- MODIFIED --- Only create the alert object, don't commit yet
        new_alert = Alert(
            detection_type=detection_type, 
            confidence_score=round(confidence, 2), 
            image_filename=filename,
            video_filename=None # Will be updated later if video is saved
        )
        db.session.add(new_alert)
        db.session.flush() # Flush to get the ID for the video
        alert_id = new_alert.id
        db.session.commit() # Commit the initial alert
        
        print(f"--- Successfully logged alert to DB (ID: {alert_id}): {filename} ---")
        return alert_id # Return the new alert's ID
        
    except Exception as e:
        db.session.rollback(); print(f"!!! ERROR logging alert to DB: {e} !!!"); traceback.print_exc()
        return None

# --- Background Task Wrapper ---
# --- MODIFIED --- This now handles notifications AND video saving
# --- Background Task Wrapper ---
# --- MODIFIED: This function now handles everything ---
def run_alert_tasks_in_background(app_context, detection_type, confidence, snapshot_frame, frame_buffer):
    with app_context: # This context is CRITICAL
        print(f"--- Background thread started for {detection_type} ---")
        
        # --- STEP 1: Log to DB (NOW INSIDE CONTEXT) ---
        alert_id = log_alert_to_db(
            detection_type, 
            confidence, 
            snapshot_frame
        )
        
        # If logging fails, stop.
        if not alert_id:
            print("!!! Background thread stopping: Failed to log alert. !!!")
            return

        # --- STEP 2: Send Notifications ---
        # We can get the real filename from the alert object if we want, but for now this works
        image_filename = f"snapshot_for_alert_{alert_id}.jpg" 
        send_email_alert(detection_type, confidence, image_filename)
        send_sms_alert(detection_type, confidence)
        
        # --- STEP 3: Save Video Clip ---
        video_filename = f"alert_video_{alert_id}.mp4"
        video_path = os.path.join(VIDEO_UPLOAD_FOLDER, video_filename)
        video_saved_successfully = False
        try:
            frame_height, frame_width, _ = frame_buffer[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'mp4v' for .mp4
            out = cv2.VideoWriter(video_path, fourcc, 10.0, (frame_width, frame_height)) # 10 FPS
            
            for frame in frame_buffer:
                out.write(frame)
            out.release()
            video_saved_successfully = True
            print(f"--- Successfully saved video clip: {video_filename} ---")
        except Exception as e:
            print(f"!!! ERROR saving video clip: {e} !!!")
            video_filename = f"error_saving_video_{alert_id}" # Log error
        
        # --- STEP 4: Update DB with video filename ---
        try:
            alert = db.session.get(Alert, alert_id) # Use db.session.get()
            if alert:
                alert.video_filename = video_filename if video_saved_successfully else "failed_to_save"
                db.session.commit()
                print(f"--- Updated DB with video info for Alert ID: {alert_id} ---")
        except Exception as e:
            db.session.rollback()
            print(f"!!! ERROR updating alert with video filename: {e} !!!")

        print(f"--- Background thread finished for Alert ID: {alert_id} ---")

# --- Stream Generator (Reads directly from camera) ---
db_cooldown = False
FRAME_BUFFER_SIZE = 50 # 5 seconds at 10 FPS
frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)

def gen_frames(detect_type="all"):
    global db_cooldown, current_roi, frame_buffer
    confidence_thresh = app_settings.get('confidence_threshold', 0.8)
    device_to_use = "cuda" if torch.cuda.is_available() else "cpu"
    model_to_use = MODELS.get(detect_type)
    if not model_to_use: return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("!!! ERROR: Could not open webcam. ---"); return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"--- Camera Properties: {width}x{height} ---")
    print(f"--- Starting video stream for {detect_type} using {device_to_use} ---")

    while True:
        try:
            socketio.sleep(0.01)
            success, frame = cap.read()
            if not success: print("--- Webcam frame read failed. ---"); break
            
            if len(frame_buffer) == FRAME_BUFFER_SIZE:
                 frame_buffer.popleft()
            frame_buffer.append(frame.copy())

            results = model_to_use.track(frame, persist=True, device=device_to_use, tracker="bytetrack.yaml")

            annotated_frame = frame
            first_alert_to_log = None

            if results and len(results) > 0:
                r = results[0]
                annotated_frame = r.plot() # Get the frame with boxes drawn on it
                frame_h, frame_w = height, width

                if r.boxes is not None:
                    boxes = r.boxes.xyxy.cpu()
                    confs = r.boxes.conf.cpu()
                    clss = r.boxes.cls.cpu()
                    ids = r.boxes.id.cpu() if r.boxes.id is not None else [None] * len(boxes)

                    for i in range(len(boxes)):
                        # Get data for the i-th box
                        x1, y1, x2, y2 = boxes[i].tolist()
                        confidence = float(confs[i])
                        class_id = int(clss[i])
                        track_id = int(ids[i]) if ids[i] is not None else (i + 1)
                
                        if confidence > confidence_thresh:
                            class_name_exact = model_to_use.names[class_id]
                            class_name_lower = class_name_exact.lower()

                            is_inside_roi = True
                            if current_roi:
                                box_center_x=(x1+x2)/2; box_center_y=(y1+y2)/2
                                roi_x1=current_roi[0]*frame_w; roi_y1=current_roi[1]*frame_h
                                roi_x2=current_roi[2]*frame_w; roi_y2=current_roi[3]*frame_h
                                if not (roi_x1 <= box_center_x <= roi_x2 and roi_y1 <= box_center_y <= roi_y2): is_inside_roi = False

                            alert_triggered = False; alert_class_name_display = ""
                            if is_inside_roi:
                                if detect_type=='violence' and class_name_lower in ['gun','knife','violence']:
                                    alert_triggered=True; alert_class_name_display=class_name_exact.capitalize()
                                elif detect_type=='fall' and class_name_lower=="fall detected":
                                    alert_triggered=True; alert_class_name_display="Fall Detected"

                            if alert_triggered:
                                alert_message = f"!!! ALERT: {alert_class_name_display} (ID: {track_id}) @ {confidence * 100:.0f}% detected !!!"
                                if app_settings.get('enable_beep_sound', False):
                                    socketio.emit('detection_alert', {'message': alert_message})
                                else:
                                    print(f"--- SERVER DETECTED (Web Alert Disabled): {alert_message} ---")

                                if not db_cooldown and first_alert_to_log is None:
                                    first_alert_to_log = {
                                        "type": alert_class_name_display, "conf": confidence,
                                        "snapshot_frame": annotated_frame.copy(), "track_id": track_id }

            if not db_cooldown and first_alert_to_log:
                db_cooldown = True
                
                # --- THIS IS THE FIX ---
                # We no longer call log_alert_to_db() here.
                # We just pass the raw data to the background thread.
                thread = threading.Thread(
                    target=run_alert_tasks_in_background, 
                    args=(
                        app.app_context(), # Pass the context
                        first_alert_to_log["type"], 
                        first_alert_to_log["conf"],
                        first_alert_to_log["snapshot_frame"], # Pass the frame
                        list(frame_buffer) # Pass the buffer
                    )
                )
                thread.start()
                # --- END OF FIX ---
                
                socketio.start_background_task(target=reset_cooldown_timer)

            if 'annotated_frame' not in locals(): annotated_frame = frame
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret: continue
            frame_bytes = buffer.tobytes(); yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e: 
            print(f"!!! CRITICAL ERROR in gen_frames loop: {e} !!!"); traceback.print_exc()
            break # Stop the loop on error

    cap.release(); print(f"--- Webcam released for stream {detect_type} ---")


# --- Cooldown Timer (Use loaded settings) ---
def reset_cooldown_timer():
    cooldown_sec = app_settings.get('alert_cooldown_seconds', 3)
    socketio.sleep(cooldown_sec)
    global db_cooldown
    db_cooldown = False
    print(f"--- DB Cooldown reset ({cooldown_sec}s). ---")

# --- SocketIO Handlers for ROI ---
@socketio.on('set_roi')
def handle_set_roi(data):
    global current_roi
    if data and 'roi' in data and len(data['roi']) == 4:
        roi = [max(0.0, min(1.0, float(c))) for c in data['roi']]; x1, y1, x2, y2 = roi
        current_roi = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]; print(f"--- ROI Set: {current_roi} ---")
    else: print("--- Invalid ROI data received ---")

@socketio.on('clear_roi')
def handle_clear_roi():
    global current_roi; current_roi = None; print("--- ROI Cleared ---")

# --- WTForms Definitions ---
class SettingsForm(FlaskForm):
    confidence_threshold = FloatField('Confidence Threshold', validators=[DataRequired(), NumberRange(min=0.1, max=0.99)], render_kw={"step": "0.05"})
    alert_cooldown_seconds = IntegerField('Alert Cooldown (seconds)', validators=[DataRequired(), NumberRange(min=1, max=300)])
    recipient_email = StringField('Recipient Email', validators=[Optional(), Email()])
    recipient_phone = StringField('Recipient Phone (e.g., +91987... )', validators=[Optional(), Length(min=10, max=15)])
    enable_email_alerts = BooleanField('Enable Email Alerts', default=True)
    enable_sms_alerts = BooleanField('Enable SMS Alerts', default=True)
    enable_beep_sound = BooleanField('Enable Beep Sound Alert (Web Page)', default=True)
    submit = SubmitField('Save Settings')

# --- Login Form (for CSRF Token) ---
class LoginForm(FlaskForm):
    user_id = StringField('User ID')
    password = PasswordField('Password')
# ----------------------------------

# --- Login Required Decorator ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# --- NEW: Context Processor for CSS Variables ---
@app.context_processor
def utility_processor():
    def get_css_variable(var_name):
        return css_variables.get(var_name)
    return dict(get_css_variable=get_css_variable)
# ---------------------------------------------

# --- Web Routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('index'))

    # Handle standard POST from the form
    if request.method == 'POST':
        # Get data from the standard form submission (request.form)
        user_id = request.form.get('user_id')
        password = request.form.get('password')

        if user_id == DUMMY_USER['user'] and password == DUMMY_USER['pass']:
            session['logged_in'] = True
            flash('Login successful!', 'success') # Flash message on success
            return redirect(url_for('index'))
        else:
            flash('Invalid User ID or Password', 'danger') # Flash message on failure
            return redirect(url_for('login')) # Redirect back to login page
    
    # Handle GET request
    form = LoginForm() # Create form to pass for CSRF token
    return render_template('login.html', cache_v=int(time.time()), form=form)

@app.route('/logout')
def logout():
    session.pop('logged_in', None); flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html', cache_v=int(time.time()))

@app.route('/detect/<string:detect_type>')
@login_required
def detect_page(detect_type):
    if detect_type not in ['violence', 'fall']: return "Invalid detection type", 404
    # --- NEW --- Clear buffer when starting a new stream
    global frame_buffer
    frame_buffer.clear()
    print("--- Frame buffer cleared for new stream ---")
    return render_template('detect.html', detect_type=detect_type, cache_v=int(time.time()))


@app.route('/video_feed/<string:detect_type>')
@login_required
def video_feed(detect_type):
    return Response(gen_frames(detect_type), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
@login_required
def logs():
    try: all_alerts = Alert.query.order_by(Alert.timestamp.desc()).all()
    except Exception as e: print(f"Error querying logs: {e}"); all_alerts = []
    return render_template('logs.html', alerts=all_alerts, cache_v=int(time.time()))

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        alert_type_data=db.session.query(Alert.detection_type, func.count(Alert.id)).group_by(Alert.detection_type).all(); alert_type_counts={atype: count for atype,count in alert_type_data}
        seven_days_ago = datetime.datetime.utcnow() - timedelta(days=6); alerts_per_day_data = db.session.query(cast(Alert.timestamp, Date).label('alert_date'), func.count(Alert.id).label('count')).filter(Alert.timestamp >= seven_days_ago).group_by('alert_date').order_by('alert_date').all()
        alerts_per_day = { (seven_days_ago.date() + timedelta(days=i)).strftime('%Y-%m-%d'): 0 for i in range(7) }; [alerts_per_day.update({date_obj.strftime('%Y-%m-%d'): count}) for date_obj, count in alerts_per_day_data]
        total_alerts = Alert.query.count()
    except Exception as e: print(f"Error querying dashboard data: {e}"); alert_type_counts={}; alerts_per_day={}; total_alerts=0
    return render_template('dashboard.html', alert_type_counts=alert_type_counts, alerts_per_day=alerts_per_day, total_alerts=total_alerts, cache_v=int(time.time()))

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    form = SettingsForm()
    if form.validate_on_submit():
        app_settings['confidence_threshold'] = form.confidence_threshold.data
        app_settings['alert_cooldown_seconds'] = form.alert_cooldown_seconds.data
        app_settings['recipient_email'] = form.recipient_email.data or None
        app_settings['recipient_phone'] = form.recipient_phone.data or None
        app_settings['enable_email_alerts'] = form.enable_email_alerts.data
        app_settings['enable_sms_alerts'] = form.enable_sms_alerts.data
        app_settings['enable_beep_sound'] = form.enable_beep_sound.data
        save_settings()
        flash('Settings updated successfully!', 'success')
        return redirect(url_for('settings'))
    elif request.method == 'GET':
        form.confidence_threshold.data = app_settings.get('confidence_threshold', 0.8)
        form.alert_cooldown_seconds.data = app_settings.get('alert_cooldown_seconds', 3)
        form.recipient_email.data = app_settings.get('recipient_email', '')
        form.recipient_phone.data = app_settings.get('recipient_phone', '')
        form.enable_email_alerts.data = app_settings.get('enable_email_alerts', True)
        form.enable_sms_alerts.data = app_settings.get('enable_sms_alerts', True)
        form.enable_beep_sound.data = app_settings.get('enable_beep_sound', True)
    return render_template('settings.html', form=form, cache_v=int(time.time()))

# --- Run the App ---
if __name__ == '__main__':
    with app.app_context(): db.create_all()
    print("Starting Flask app with SocketIO...")
    socketio.run(app, use_reloader=False) # use_reloader=False is important for threaded apps
    print("--- Flask app shutting down. ---")