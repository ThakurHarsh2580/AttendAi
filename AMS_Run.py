#!/usr/bin/env python3
"""
Face Recognition Attendance System with Advanced Features.
- Teacher & Admin Portals
- Student & Class Management
- Attendance Reporting
"""

import os
import time
import datetime
import csv
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

# Optional DB
try:
    import pymysql
except Exception:
    pymysql = None

# -----------------------------------------------------------
# Core Configuration: Paths and File System Setup
# -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define directories for storing data
TRAINING_DIR = os.path.join(BASE_DIR, "TrainingImage")
TRAINING_LABEL_DIR = os.path.join(BASE_DIR, "TrainingImageLabel")
STUDENT_DIR = os.path.join(BASE_DIR, "StudentDetails")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "Attendance")

# Define key file paths
STUDENT_CSV = os.path.join(STUDENT_DIR, "StudentDetails.csv")
HAAR_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
TRAINER_PATH = os.path.join(TRAINING_LABEL_DIR, "Trainner.yml")
SUBJECT_CSV = os.path.join(STUDENT_DIR, "subjects.csv")
TEACHER_CSV = os.path.join(BASE_DIR, "teachers.csv")

# Ensure all necessary directories exist
for d in (TRAINING_DIR, TRAINING_LABEL_DIR, STUDENT_DIR, ATTENDANCE_DIR):
    os.makedirs(d, exist_ok=True)

# Create initial data files if they don't exist
if not os.path.exists(SUBJECT_CSV):
    with open(SUBJECT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Subject"])
        writer.writerow(["Math"])
        writer.writerow(["Science"])

if not os.path.exists(TEACHER_CSV):
    with open(TEACHER_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Password"])
        writer.writerow(["teacher", "teacher123"])

# UI configuration constants
APP_TITLE = "Smart Attend Ai-(Face Recognition Attendance System)"
BG_COLOR = "#F9F9F9"
ACCENT_COLOR = "#4A90E2"
FONT = "Helvetica Neue"

# -----------------------------------------------------------
# App Window and Styling
# -----------------------------------------------------------
# Initialize the main Tkinter window
window = tk.Tk()
window.title(APP_TITLE)
window.geometry("1280x720")
window.configure(bg=BG_COLOR)

# Configure modern UI styling using Tkinter's themed widgets (Ttk)
style = ttk.Style(window)
try:
    style.theme_use("clam")
except Exception:
    pass

style.configure("TFrame", background=BG_COLOR)
style.configure("TButton",
                font=(FONT, 14, "bold"),
                padding=12,
                relief="flat",
                background=ACCENT_COLOR,
                foreground="white",
                borderwidth=0,
                focusthickness=0)
style.map("TButton",
          background=[("active", "#3A7BD5"), ("pressed", "#2A64B2")])

style.configure("TLabel",
                font=(FONT, 14),
                background=BG_COLOR,
                foreground="#333333")

style.configure("TEntry",
                font=(FONT, 14),
                padding=8,
                relief="flat",
                borderwidth=1,
                fieldbackground="white",
                foreground="#333333")
style.map("TEntry",
          bordercolor=[("focus", ACCENT_COLOR)])

style.configure("Treeview",
                font=(FONT, 13),
                rowheight=28,
                background="white",
                fieldbackground="white",
                bordercolor="#DDDDDD",
                borderwidth=1)
style.configure("Treeview.Heading",
                font=(FONT, 14, "bold"),
                background="#EFEFEF",
                foreground="#333333")
style.map("Treeview",
          background=[("selected", "#CDE1F4")],
          foreground=[("selected", "black")])

# -----------------------------------------------------------
# Page Management System
# -----------------------------------------------------------
class App(tk.Frame):
    """
    Main application class that manages different pages (frames).
    Uses a history stack for back navigation.
    """
    def __init__(self, master=None):
        super().__init__(master, bg=BG_COLOR)
        self.master = master
        self.pack(fill="both", expand=True)
        self.history = ["HomePage"]  # Navigation history stack
        self.create_widgets()

    def create_widgets(self):
        """Initializes the main UI structure and all app pages."""
        header_frame = tk.Frame(self, bg=ACCENT_COLOR, height=70)
        header_frame.pack(fill="x", side="top")
        tk.Label(header_frame,
                 text="Smart Attend Ai - Face Recognition Attendance System",
                 bg=ACCENT_COLOR,
                 fg="white",
                 font=(FONT, 28, "bold")).pack(pady=15)

        self.main_frame = ttk.Frame(self, padding=25)
        self.main_frame.pack(fill="both", expand=True)

        self.pages = {}
        # Instantiate each page and store it in a dictionary
        for F in (HomePage, RegisterPage, TeacherLoginPage, TeacherPage, AdminLoginPage, AdminPage, ManageTeachersPage):
            page_name = F.__name__
            frame = F(parent=self.main_frame, controller=self)
            self.pages[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        self.show_frame("HomePage")

        # Status bar at the bottom of the window
        self.status = tk.Label(self.master, text="", anchor="e", font=(FONT, 11),
                               bg="#E5E5EA", fg="black", pady=6)
        self.status.pack(side="bottom", fill="x")
        self.set_status("Ready")
        self.tick_clock()

    def show_frame(self, page_name):
        """Brings the specified page to the front."""
        if self.history and self.history[-1] != page_name:
            self.history.append(page_name)
        frame = self.pages[page_name]
        frame.tkraise()

    def go_back(self):
        """Navigates to the previous page in the history stack."""
        if len(self.history) > 1:
            self.history.pop()
            previous_page = self.history[-1]
            self.show_frame(previous_page)

    def set_status(self, msg: str):
        """Updates the text in the status bar."""
        text = self.status.cget("text")
        if "    " in text:
            left = text.split("    ")[0]
        else:
            left = "Ready"
        self.status.config(text=f"{msg}    {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def tick_clock(self):
        """Updates the clock in the status bar every second."""
        text = self.status.cget("text")
        if "    " in text:
            left = text.split("    ")[0]
        else:
            left = "Ready"
        self.status.config(text=f"{left}    {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.master.after(1000, self.tick_clock)

# -----------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------
def ensure_haar():
    """Checks if the Haar Cascade file exists."""
    if not os.path.exists(HAAR_PATH):
        messagebox.showerror("Missing file", f"Missing Haar cascade: {HAAR_PATH}")
        return False
    return True

def ensure_cv2_face():
    """Checks if the `cv2.face` module is available (requires opencv-contrib-python)."""
    if not hasattr(cv2, "face"):
        messagebox.showerror("OpenCV Missing Module",
                             "Your OpenCV build doesn't include 'cv2.face'.\nInstall opencv-contrib-python:\n    pip install opencv-contrib-python")
        return False
    return True

def get_images_and_labels(path, detector):
    """
    Processes images from a directory to get face samples and their corresponding IDs.
    This is used for the training process.
    """
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []
    for image_path in image_paths:
        try:
            pil_img = Image.open(image_path).convert('L') # Convert to grayscale
        except Exception:
            continue
        img_np = np.array(pil_img, 'uint8') # Convert to a NumPy array
        try:
            # Extract student ID from the filename (e.g., 'Name.ID.SampleNum.jpg')
            parts = os.path.basename(image_path).split(".")
            Id = int(parts[1])
        except Exception:
            continue
        faces = detector.detectMultiScale(img_np)
        for (x, y, w, h) in faces:
            face_samples.append(img_np[y:y + h, x:x + w])
            ids.append(Id)
    return face_samples, ids
    
def load_subjects():
    """Loads a list of subjects from the subjects.csv file."""
    if not os.path.exists(SUBJECT_CSV):
        return []
    try:
        df = pd.read_csv(SUBJECT_CSV)
        return df['Subject'].tolist()
    except Exception:
        return []

# -----------------------------------------------------------
# Page Definitions
# -----------------------------------------------------------

class HomePage(ttk.Frame):
    """The main entry point of the application."""
    def __init__(self, parent, controller):
        super().__init__(parent, padding=50)
        self.controller = controller
        
        tk.Label(self, text="Welcome! What would you like to do?",
                 font=(FONT, 22, "bold"), bg=BG_COLOR, fg="#1C1C1E").pack(pady=(0, 30))

        button_frame = ttk.Frame(self)
        button_frame.pack(pady=20)
        
        buttons = [
            ("Teacher Login", lambda: self.controller.show_frame("TeacherLoginPage")),
            ("Admin Login", lambda: self.controller.show_frame("AdminLoginPage")),
            ("Register New Student", lambda: self.controller.show_frame("RegisterPage")),
            ("Train Recognition Model", self.train_images),
            ("Fill Automatic Attendance", self.fill_automatic),
        ]
        
        for i, (text, command) in enumerate(buttons):
            ttk.Button(button_frame, text=text, command=command, width=30).grid(row=i, column=0, pady=10)

    def train_images(self):
        """
        Trains the face recognition model (LBPH).
        It checks for existing images and handles the training process with a progress bar.
        """
        if not ensure_haar() or not ensure_cv2_face():
            return
        if not os.listdir(TRAINING_DIR):
            messagebox.showerror("No Data", 'Please add images to "TrainingImage" first.')
            return

        # Create a progress window
        top = tk.Toplevel(self.controller.master)
        top.title("Training Model")
        top.geometry("440x160")
        top.configure(bg=BG_COLOR)
        ttk.Label(top, text="Training LBPH model, please wait...", font=(FONT, 12, "bold"), background=BG_COLOR).pack(pady=12)
        pb = ttk.Progressbar(top, mode="indeterminate", length=380)
        pb.pack(pady=8)
        pb.start(10)
        self.controller.master.update_idletasks() # Update the UI to show the progress bar

        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            detector = cv2.CascadeClassifier(HAAR_PATH)
            faces, ids = get_images_and_labels(TRAINING_DIR, detector)
            if not faces:
                raise RuntimeError("No faces found in the TrainingImage folder.")
            recognizer.train(faces, np.array(ids))
            recognizer.save(TRAINER_PATH)
            messagebox.showinfo("Success", "Model trained successfully.")
            self.controller.set_status("Model trained successfully.")
        except Exception as e:
            messagebox.showerror("Training Error", str(e))
            self.controller.set_status("Training failed.")
        finally:
            pb.stop()
            top.destroy()

    def fill_automatic(self):
        """Opens a new window to select a subject for automatic attendance."""
        sub_win = tk.Toplevel(self.controller.master)
        sub_win.title("Select Subject")
        sub_win.geometry("420x240")
        sub_win.configure(bg=BG_COLOR)

        ttk.Label(sub_win, text="Select Subject:", font=(FONT, 15, "bold"), background=BG_COLOR).pack(pady=(18, 8))
        
        subjects = load_subjects()
        subject_var = tk.StringVar()
        subject_combo = ttk.Combobox(sub_win, textvariable=subject_var, values=subjects, font=(FONT, 13), width=28, state="readonly")
        subject_combo.pack(pady=6, padx=20)
        if subjects:
            subject_combo.current(0)

        ttk.Label(sub_win, text="Attendance will run for 20 seconds or press ESC.",
                  font=(FONT, 10), background=BG_COLOR).pack(pady=6, padx=18)

        def do_fill():
            """Starts the automatic attendance process."""
            subject = subject_var.get().strip()
            if not subject:
                messagebox.showerror("Missing Subject", "Please select a subject.")
                return
            
            self._run_automatic_attendance(subject)
            sub_win.destroy()

        ttk.Button(sub_win, text="Start Automatic Attendance", command=do_fill).pack(pady=12)
        
    def _run_automatic_attendance(self, subject):
        """
        Runs the automatic attendance process using the webcam.
        This function captures video, detects faces, and logs attendance.
        """
        if not ensure_haar() or not ensure_cv2_face(): return
        if not os.path.exists(TRAINER_PATH):
            messagebox.showerror("Model not found", "Please train the model first.")
            return
        
        try: 
            # Read student details, ensuring 'Enrollment' is treated as a string for comparison.
            df = pd.read_csv(STUDENT_CSV, dtype={'Enrollment': str})
        except Exception:
            messagebox.showerror("Missing Data", f"{STUDENT_CSV} not found or unreadable.")
            return

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        try: recognizer.read(TRAINER_PATH)
        except Exception:
            messagebox.showerror("Model Error", "Could not load trained model.")
            return

        face_cascade = cv2.CascadeClassifier(HAAR_PATH)
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            messagebox.showerror("Camera Error", "Could not open camera.")
            return

        self.controller.set_status(f"Attendance running for '{subject}'...")
        col_names = ['Enrollment', 'Name', 'Date', 'Time']
        attendance = pd.DataFrame(columns=col_names)
        start = time.time()
        duration = 20
        font = cv2.FONT_HERSHEY_SIMPLEX

        try:
            while True:
                ret, frame = cam.read()
                if not ret: break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.2, 5)

                for (x, y, w, h) in faces:
                    try: 
                        Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                    except Exception: 
                        Id, conf = "Unknown", 999
                    
                    if isinstance(Id, str) and Id == "Unknown" or conf >= 70:
                        label, color = "Unknown", (0, 0, 255)
                    else:
                        ts = time.time()
                        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                        
                        # Fix: Use .astype(str) for robust string-based comparison
                        aa = df.loc[df['Enrollment'].astype(str) == str(Id)]['Name'].values
                        name_val = str(aa[0]) if len(aa) > 0 else "Unknown"
                        attendance.loc[len(attendance)] = [str(Id), name_val, date, timeStamp]
                        label, color = f"{Id}-{name_val}", (0, 255, 0)
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 10), font, 0.8, (255, 255, 255), 2)
                    
                if len(attendance) > 0:
                    # Update status with recognized students
                    recognized_list = attendance.drop_duplicates('Enrollment')['Name'].tolist()
                    self.controller.set_status(f"Recognized: {', '.join(recognized_list)}")
                else:
                    self.controller.set_status("Detecting faces...")

                cv2.imshow("Automatic Attendance (ESC to stop)", frame)
                if cv2.waitKey(30) & 0xFF == 27 or time.time() - start > duration:
                    break
        finally:
            cam.release()
            cv2.destroyAllWindows()
            
        # Save attendance to a CSV file
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        Hour, Minute, Second = timeStamp.split(":")
        attendance = attendance.drop_duplicates(['Enrollment'], keep='first')
        fileName = os.path.join(ATTENDANCE_DIR, f"{subject}_{date}_{Hour}-{Minute}-{Second}.csv")
        attendance.to_csv(fileName, index=False)

        # Optional: Save to MySQL database if available
        if pymysql is not None:
            try:
                connection = pymysql.connect(host='localhost', user='root', password='', db='Face_reco_fill')
                cursor = connection.cursor()
                DB_Table_name = f"{subject}_{date.replace('-', '_')}_Time_{Hour}_{Minute}_{Second}"
                cursor.execute(f"CREATE TABLE IF NOT EXISTS `{DB_Table_name}`(ID INT NOT NULL AUTO_INCREMENT, ENROLLMENT VARCHAR(100) NOT NULL, NAME VARCHAR(50) NOT NULL, DATE VARCHAR(20) NOT NULL, TIME VARCHAR(20) NOT NULL, PRIMARY KEY (ID));")
                for _, row in attendance.iterrows():
                    cursor.execute(f"INSERT INTO `{DB_Table_name}` (ENROLLMENT, NAME, DATE, TIME) VALUES (%s,%s,%s,%s)", (str(row['Enrollment']), str(row['Name']), str(row['Date']), str(row['Time'])))
                connection.commit()
                cursor.close()
                connection.close()
            except Exception as e:
                messagebox.showwarning("DB Warning", f"Could not write to database:\n{e}")

        self.controller.set_status(f"Attendance saved: {os.path.basename(fileName)}")
        messagebox.showinfo("Done", f"Attendance saved:\n{fileName}")


class RegisterPage(ttk.Frame):
    """Page for registering new students and capturing their images."""
    def __init__(self, parent, controller):
        super().__init__(parent, padding=50)
        self.controller = controller

        tk.Label(self, text="Register New Student", font=(FONT, 20, "bold"), bg=BG_COLOR).pack(pady=10)
        
        form_frame = ttk.Frame(self)
        form_frame.pack(pady=20)
        
        ttk.Label(form_frame, text="Enrollment No:").grid(row=0, column=0, padx=8, pady=10, sticky="w")
        self.enroll_entry = ttk.Entry(form_frame, font=(FONT, 13), width=28)
        self.enroll_entry.grid(row=0, column=1, padx=8, pady=10, sticky="w")

        ttk.Label(form_frame, text="Student Name:").grid(row=1, column=0, padx=8, pady=10, sticky="w")
        self.name_entry = ttk.Entry(form_frame, font=(FONT, 13), width=28)
        self.name_entry.grid(row=1, column=1, padx=8, pady=10, sticky="w")
        
        ttk.Button(form_frame, text="📸 Take Images", command=self.take_images).grid(row=2, column=0, columnspan=2, pady=20)

        ttk.Button(self, text="⬅️ Back", command=self.controller.go_back).pack(pady=20)

    def validate_student_inputs(self):
        """Validates the enrollment number and name fields."""
        enr = self.enroll_entry.get().strip()
        nm = self.name_entry.get().strip()
        if not enr or not enr.isdigit():
            messagebox.showerror("Invalid Enrollment", "Please enter a numeric Enrollment number.")
            return None, None
        if not nm:
            messagebox.showerror("Invalid Name", "Please enter Student name.")
            return None, None
        return enr, nm

    def take_images(self):
        """
        Captures images of a student's face using the webcam.
        Images are saved to the 'TrainingImage' directory.
        """
        enr, nm = self.validate_student_inputs()
        if enr is None: return
        if not ensure_haar(): return

        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            messagebox.showerror("Camera Error", "Could not open the camera.")
            return

        self.controller.set_status("Camera opened. Press 'q' to stop capture early.")
        face_cascade = cv2.CascadeClassifier(HAAR_PATH)
        sample_num = 0
        max_samples = 70

        while True:
            ret, img = cam.read()
            if not ret: break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                sample_num += 1
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                img_path = os.path.join(TRAINING_DIR, f"{nm}.{enr}.{sample_num}.jpg")
                cv2.imwrite(img_path, gray[y:y + h, x:x + w])

            cv2.imshow("Capture (press q to stop)", img)
            if cv2.waitKey(1) & 0xFF == ord('q') or sample_num >= max_samples:
                break

        cam.release()
        cv2.destroyAllWindows()

        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        time_str = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        row = [enr, nm, date, time_str]

        # Save student details to StudentDetails.csv
        new_file = not os.path.exists(STUDENT_CSV)
        with open(STUDENT_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["Enrollment", "Name", "Date", "Time"])
            w.writerow(row)

        messagebox.showinfo("Saved", f"Saved {sample_num} images for {nm} ({enr}).")
        self.controller.set_status(f"Saved {sample_num} images for {nm} ({enr}).")


class TeacherLoginPage(ttk.Frame):
    """Login page for teachers."""
    def __init__(self, parent, controller):
        super().__init__(parent, padding=50)
        self.controller = controller

        tk.Label(self, text="Teacher Login", font=(FONT, 20, "bold"), bg="red").pack(pady=12)
        ttk.Label(self, text="Username:", font=(FONT, 12), background=BG_COLOR).pack(padx=28, anchor="w")
        self.un_entr = ttk.Entry(self, width=30)
        self.un_entr.pack(padx=28, pady=6)
        ttk.Label(self, text="Password:", font=(FONT, 12), background=BG_COLOR).pack(padx=28, anchor="w")
        self.pw_entr = ttk.Entry(self, width=30, show="•")
        self.pw_entr.pack(padx=28, pady=6)
        ttk.Button(self, text="Log In", command=self.do_login).pack(pady=14)
        
        # Back button now explicitly goes to the Home Page
        ttk.Button(self, text="⬅️ Back", command=lambda: self.controller.show_frame("HomePage")).pack(pady=25)



    def do_login(self):
        """Authenticates a teacher based on the teachers.csv file."""
        username = self.un_entr.get().strip()
        password = self.pw_entr.get().strip()
        self.pw_entr.delete(0, 'end') # Clear password field for security and UX
        
        if not os.path.exists(TEACHER_CSV):
            messagebox.showerror("Login Failed", "Teacher database not found.")
            return

        try:
            # Bug fix: Read ID and Password as strings for robust comparison
            df = pd.read_csv(TEACHER_CSV, dtype={'ID': str, 'Password': str})
            match = df[(df['ID'] == username) & (df['Password'] == password)]
            if not match.empty:
                self.controller.show_frame("TeacherPage")
            else:
                messagebox.showerror("Login Failed", "Incorrect ID or Password.")
        except Exception as e:
            messagebox.showerror("Login Failed", f"Could not read teacher database:\n{e}")


class TeacherPage(ttk.Frame):
    """The main portal for teachers to manage attendance and reports."""
    def __init__(self, parent, controller):
        super().__init__(parent, padding=25)
        self.controller = controller
        
        tk.Label(self, text="Teacher Portal", font=(FONT, 20, "bold"), bg=BG_COLOR).pack(pady=10)

        main_buttons_frame = ttk.Frame(self)
        main_buttons_frame.pack(pady=10)
        
        ttk.Button(main_buttons_frame, text="View Attendance", command=self.view_attendance).pack(side="left", padx=10)
        ttk.Button(main_buttons_frame, text="Generate Report", command=self.generate_report).pack(side="left", padx=10)
        ttk.Button(main_buttons_frame, text="Generate Semester Report", command=self.generate_semester_report).pack(side="left", padx=10)
        ttk.Button(main_buttons_frame, text="⬅️ Log Out", command=lambda: self.controller.show_frame("TeacherLoginPage")).pack(side="left", padx=10)

        # UI for manual attendance
        manual_frame = ttk.LabelFrame(self, text="Manual Attendance", padding=15)
        manual_frame.pack(fill="both", expand=True, pady=20)
        
        manual_frame.grid_columnconfigure(1, weight=1)

        subjects = load_subjects()
        ttk.Label(manual_frame, text="Subject:", font=(FONT, 12)).grid(row=0, column=0, padx=12, pady=8, sticky="w")
        self.subject_var = tk.StringVar()
        self.subject_combo = ttk.Combobox(manual_frame, textvariable=self.subject_var, values=subjects, font=(FONT, 12), width=28, state="readonly")
        self.subject_combo.grid(row=0, column=1, padx=12, pady=8, sticky="w")
        if subjects:
            self.subject_combo.current(0)
        
        ttk.Label(manual_frame, text="Enrollment:", font=(FONT, 12)).grid(row=1, column=0, padx=12, pady=8, sticky="w")
        self.enr_entry = ttk.Entry(manual_frame, font=(FONT, 12), width=30)
        self.enr_entry.grid(row=1, column=1, padx=12, pady=8, sticky="w")
        
        ttk.Label(manual_frame, text="Student Name:", font=(FONT, 12)).grid(row=2, column=0, padx=12, pady=8, sticky="w")
        self.nm_entry = ttk.Entry(manual_frame, font=(FONT, 12), width=30)
        self.nm_entry.grid(row=2, column=1, padx=12, pady=8, sticky="w")
        
        self.rows = [] # List to hold rows for manual attendance
        
        ttk.Button(manual_frame, text="Add Row", command=self.add_row).grid(row=3, column=0, padx=12, pady=10, sticky="w")
        ttk.Button(manual_frame, text="Save to CSV/DB", command=self.save_to_db_and_csv).grid(row=3, column=1, padx=12, pady=10, sticky="e")
        
        self.tv = ttk.Treeview(manual_frame, columns=["Enrollment", "Name", "Date", "Time"], show="headings", height=8)
        for c in ["Enrollment", "Name", "Date", "Time"]:
            self.tv.heading(c, text=c)
            self.tv.column(c, width=120 if c != "Name" else 180, anchor="center")
        self.tv.grid(row=4, column=0, columnspan=2, padx=12, pady=(6, 12), sticky="nsew")
        manual_frame.grid_rowconfigure(4, weight=1)

    def view_attendance(self):
        """Allows the user to view an attendance CSV file in a new window."""
        filename = filedialog.askopenfilename(initialdir=ATTENDANCE_DIR, title="Select Attendance File", filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")))
        if filename:
            try:
                df = pd.read_csv(filename)
                viewer = tk.Toplevel(self.controller.master)
                viewer.title(f"Attendance - {os.path.basename(filename)}")
                viewer.geometry("700x420")
                viewer.configure(bg=BG_COLOR)
                
                ttk.Label(viewer, text=f"Attendance: {os.path.basename(filename)}", font=(FONT, 16, "bold"), background=BG_COLOR).pack(pady=10)
                frame = ttk.Frame(viewer)
                frame.pack(fill="both", expand=True, padx=12, pady=8)

                tv_view = ttk.Treeview(frame, columns=list(df.columns), show="headings")
                for col in df.columns:
                    tv_view.heading(col, text=col)
                    tv_view.column(col, width=150, anchor="center")
                for _, row in df.iterrows():
                    tv_view.insert("", "end", values=list(row.values))
                tv_view.pack(fill="both", expand=True)

            except Exception as e:
                messagebox.showerror("File Error", f"Could not read the file:\n{e}")

    def generate_report(self):
        """Generates a combined attendance report for a subject over all available dates."""
        top = tk.Toplevel(self.controller.master)
        top.title("Generate Attendance Report")
        top.geometry("450x300")
        top.configure(bg=BG_COLOR)
        
        ttk.Label(top, text="Select Subject for Report:", font=(FONT, 14, "bold"), background=BG_COLOR).pack(pady=10)
        
        subjects = load_subjects()
        subject_var = tk.StringVar()
        subject_combo = ttk.Combobox(top, textvariable=subject_var, values=subjects, font=(FONT, 13), width=28, state="readonly")
        subject_combo.pack(pady=6)
        if subjects:
            subject_combo.current(0)
            
        def process_report():
            selected_subject = subject_var.get()
            if not selected_subject:
                messagebox.showerror("Error", "Please select a subject.")
                return

            try:
                all_attendance = []
                for f in os.listdir(ATTENDANCE_DIR):
                    if f.startswith(selected_subject) and f.endswith(".csv"):
                        filepath = os.path.join(ATTENDANCE_DIR, f)
                        df = pd.read_csv(filepath)
                        all_attendance.append(df)

                if not all_attendance:
                    messagebox.showinfo("No Data", f"No attendance data found for {selected_subject}.")
                    return

                combined_df = pd.concat(all_attendance, ignore_index=True)
                combined_df.drop_duplicates(subset=['Enrollment', 'Date'], keep='first', inplace=True)
                combined_df.sort_values(by=['Date', 'Time'], inplace=True)
                
                # Ask user for a save location
                filename = filedialog.asksaveasfilename(
                    defaultextension=".csv",
                    initialfile=f"{selected_subject}_Attendance_Report.csv",
                    filetypes=[("CSV files", "*.csv")]
                )
                if filename:
                    combined_df.to_csv(filename, index=False)
                    messagebox.showinfo("Success", f"Report saved to {filename}")
                    top.destroy()
            except Exception as e:
                messagebox.showerror("Report Error", f"Failed to generate report: {e}")

        ttk.Button(top, text="Generate & Export Report", command=process_report).pack(pady=20)

    def generate_semester_report(self):
        """
        Opens a new window to calculate attendance metrics for a specific student
        over a defined date range.
        """
        top = tk.Toplevel(self.controller.master)
        top.title("Generate Semester Report")
        top.geometry("550x450")
        top.configure(bg=BG_COLOR)

        report_frame = ttk.Frame(top, padding=20)
        report_frame.pack(fill="both", expand=True)

        ttk.Label(report_frame, text="Generate Semester Report", font=(FONT, 16, "bold")).pack(pady=10)
        
        input_frame = ttk.Frame(report_frame)
        input_frame.pack(pady=10)

        subjects = load_subjects()
        ttk.Label(input_frame, text="Subject:").grid(row=0, column=0, sticky="w", padx=5)
        self.subject_var_report = tk.StringVar()
        subject_combo = ttk.Combobox(input_frame, textvariable=self.subject_var_report, values=subjects, state="readonly")
        subject_combo.grid(row=0, column=1, sticky="ew", padx=5)
        if subjects:
            subject_combo.current(0)
        
        ttk.Label(input_frame, text="Student ID:").grid(row=1, column=0, sticky="w", padx=5)
        self.student_id_entry = ttk.Entry(input_frame, width=20)
        self.student_id_entry.grid(row=1, column=1, sticky="ew", padx=5)
        
        ttk.Label(input_frame, text="Start Date (YYYY-MM-DD):").grid(row=2, column=0, sticky="w", padx=5)
        self.start_date_entry = ttk.Entry(input_frame, width=20)
        self.start_date_entry.grid(row=2, column=1, sticky="ew", padx=5)
        
        ttk.Label(input_frame, text="End Date (YYYY-MM-DD):").grid(row=3, column=0, sticky="w", padx=5)
        self.end_date_entry = ttk.Entry(input_frame, width=20)
        self.end_date_entry.grid(row=3, column=1, sticky="ew", padx=5)

        self.result_label = ttk.Label(report_frame, text="", font=(FONT, 14, "bold"))
        self.result_label.pack(pady=20)

        ttk.Button(report_frame, text="Calculate Metrics", command=self.calculate_metrics).pack(pady=10)
        
        ttk.Button(report_frame, text="Close", command=top.destroy).pack(pady=10)

    def calculate_metrics(self):
        """
        Calculates a student's attendance percentage and percentile rank.
        This includes bug fixes for data type handling and the percentile formula.
        """
        subject = self.subject_var_report.get()
        student_id = self.student_id_entry.get().strip()
        start_date_str = self.start_date_entry.get().strip()
        end_date_str = self.end_date_entry.get().strip()

        if not all([subject, student_id, start_date_str, end_date_str]):
            messagebox.showerror("Input Error", "All fields are required.")
            return

        try:
            start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').date()
            end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').date()
            if start_date > end_date:
                 messagebox.showerror("Date Error", "Start date cannot be after end date.")
                 return
        except ValueError:
            messagebox.showerror("Date Format Error", "Invalid date format. Please use YYYY-MM-DD.")
            return

        try:
            all_attendance_records = []
            for f in os.listdir(ATTENDANCE_DIR):
                if f.startswith(subject) and f.endswith(".csv"):
                    file_path = os.path.join(ATTENDANCE_DIR, f)
                    
                    # Bug fix: Ensure 'Enrollment' and 'Date' columns are read as strings
                    df = pd.read_csv(file_path, dtype={'Enrollment': str, 'Date': str})
                    
                    # Filter by date range
                    df['Date'] = pd.to_datetime(df['Date']).dt.date
                    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
                    
                    all_attendance_records.append(df)

            if not all_attendance_records:
                self.result_label.config(text=f"No attendance data found for {subject}.")
                return

            combined_df = pd.concat(all_attendance_records, ignore_index=True)
            
            # Count the total number of classes held within the date range
            total_classes_df = combined_df.drop_duplicates(subset=['Date', 'Time'])
            total_classes = len(total_classes_df)

            # Count attended classes for the specific student
            student_attended_count = combined_df[combined_df['Enrollment'] == student_id].shape[0]
            
            if total_classes == 0:
                self.result_label.config(text="No classes held in the selected period.")
                return

            # --- Calculate Attendance Percentage ---
            attendance_percentage = (student_attended_count / total_classes) * 100
            
            # --- Bug Fix: Correct percentile calculation ---
            # Get attendance percentages for ALL students
            all_percentages = []
            unique_students = combined_df['Enrollment'].unique()
            for s_id in unique_students:
                attended_count = combined_df[combined_df['Enrollment'] == s_id].shape[0]
                all_percentages.append((attended_count / total_classes) * 100)

            # Sort the percentages to find the percentile rank
            sorted_percentages = sorted(all_percentages)
            
            # The percentile is the rank of the student's percentage divided by the total count
            percentile_rank = (np.searchsorted(sorted_percentages, attendance_percentage, side='right') / len(sorted_percentages)) * 100

            result_text = f"""
            Results for Student ID: {student_id}
            ------------------------------------------------
            Attendance Percentage: {attendance_percentage:.2f}%
            Attendance Percentile: {percentile_rank:.2f}%
            """
            self.result_label.config(text=result_text)

        except Exception as e:
            messagebox.showerror("Calculation Error", f"An error occurred: {e}")

    def add_row(self):
        """Adds a new row to the manual attendance Treeview."""
        subject = self.subject_var.get().strip()
        enr = self.enr_entry.get().strip()
        nm = self.nm_entry.get().strip()
        if not subject:
            messagebox.showerror("Missing Subject", "Please select a subject.")
            return
        if not enr or not enr.isdigit():
            messagebox.showerror("Invalid Enrollment", "Please enter numeric enrollment.")
            return
        if not nm:
            messagebox.showerror("Missing Name", "Please enter student name.")
            return
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        tms = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        self.rows.append([enr, nm, date, tms])
        self.tv.insert("", "end", values=[enr, nm, date, tms])
        self.enr_entry.delete(0, tk.END)
        self.nm_entry.delete(0, tk.END)

    def save_to_db_and_csv(self):
        """Saves the manual attendance data to a CSV and optionally to a database."""
        subject = self.subject_var.get().strip()
        if not subject:
            messagebox.showerror("Missing Subject", "Please select a subject.")
            return
        if not self.rows:
            messagebox.showerror("No Data", "Please add at least one row.")
            return
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        Hour, Minute, Second = timeStamp.split(":")
        csv_name = os.path.join(ATTENDANCE_DIR, f"{subject}_{date}_Time_{Hour}_{Minute}_{Second}_manual.csv")
        with open(csv_name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Enrollment", "Name", "Date", "Time"])
            w.writerows(self.rows)

        if pymysql is not None:
            try:
                connection = pymysql.connect(host='localhost', user='root', password='', db='manually_fill_attendance')
                cursor = connection.cursor()
                DB_table_name = f"{subject}_{date.replace('-', '_')}_Time_{Hour}_{Minute}_{Second}"
                cursor.execute(f"CREATE TABLE IF NOT EXISTS `{DB_table_name}`(ID INT NOT NULL AUTO_INCREMENT, ENROLLMENT VARCHAR(100) NOT NULL, NAME VARCHAR(50) NOT NULL, DATE VARCHAR(20) NOT NULL, TIME VARCHAR(20) NOT NULL, PRIMARY KEY (ID));")
                for r in self.rows:
                    cursor.execute(f"INSERT INTO `{DB_table_name}` (ENROLLMENT, NAME, DATE, TIME) VALUES (%s,%s,%s,%s)", (str(r[0]), str(r[1]), str(r[2]), str(r[3])))
                connection.commit()
                cursor.close()
                connection.close()
            except Exception as e:
                messagebox.showwarning("DB Warning", f"Could not write to database:\n{e}")

        messagebox.showinfo("Saved", f"Manual attendance saved:\n{csv_name}")
        self.controller.set_status(f"Manual attendance saved: {os.path.basename(csv_name)}")


class AdminLoginPage(ttk.Frame):
    """Login page for the administrator."""
    def __init__(self, parent, controller):
        super().__init__(parent, padding=50)
        self.controller = controller

        tk.Label(self, text="Admin Login", font=(FONT, 20, "bold"), bg="red").pack(pady=12)
        ttk.Label(self, text="Username:", font=(FONT, 12), background=BG_COLOR).pack(padx=28, anchor="w")
        self.un_entr = ttk.Entry(self, width=30)
        self.un_entr.pack(padx=28, pady=6)
        ttk.Label(self, text="Password:", font=(FONT, 12), background=BG_COLOR).pack(padx=28, anchor="w")
        self.pw_entr = ttk.Entry(self, width=30, show="•")
        self.pw_entr.pack(padx=28, pady=6)
        ttk.Button(self, text="Log In", command=self.do_login).pack(pady=14)
        
        # Back button now explicitly goes to the Home Page
        ttk.Button(self, text="⬅️ Back", command=lambda: self.controller.show_frame("HomePage")).pack(pady=20)

    def do_login(self):
        """Authenticates the administrator."""
        username = self.un_entr.get().strip()
        password = self.pw_entr.get().strip()
        self.pw_entr.delete(0, 'end') # Clear password field for UX
        
        # Hardcoded admin credentials
        if username == "admin" and password == "admin@123":
            self.controller.show_frame("AdminPage")
        else:
            messagebox.showerror("Login Failed", "Incorrect ID or Password.")


class AdminPage(ttk.Frame):
    """The main portal for administrators to manage system data."""
    def __init__(self, parent, controller):
        super().__init__(parent, padding=25)
        self.controller = controller
        
        self.viewer_frame = ttk.Frame(self)
        self.viewer_frame.pack(fill="both", expand=True, padx=8, pady=8)

        self.show_main_dashboard()
        
    def show_main_dashboard(self):
        """Displays the main admin options."""
        for widget in self.viewer_frame.winfo_children():
            widget.destroy()
        
        tk.Label(self.viewer_frame, text="Admin Portal", font=(FONT, 20, "bold"), bg=BG_COLOR).pack(pady=10)

        main_buttons_frame = ttk.Frame(self.viewer_frame)
        main_buttons_frame.pack(pady=10)
        
        ttk.Button(main_buttons_frame, text="Manage Students", command=self.manage_students).pack(side="left", padx=10)
        ttk.Button(main_buttons_frame, text="Manage Subjects", command=self.manage_subjects).pack(side="left", padx=10)
        ttk.Button(main_buttons_frame, text="Manage Teachers", command=lambda: self.controller.show_frame("ManageTeachersPage")).pack(side="left", padx=10)
        
        ttk.Button(self.viewer_frame, text="⬅️ Log Out", command=lambda: self.controller.show_frame("AdminLoginPage")).pack(pady=20)

    def manage_subjects(self):
        """Displays the UI for managing subjects."""
        for widget in self.viewer_frame.winfo_children():
            widget.destroy()

        tk.Label(self.viewer_frame, text="Manage Subjects", font=(FONT, 18, "bold"), bg=BG_COLOR).pack(pady=10)
        
        input_frame = ttk.Frame(self.viewer_frame)
        input_frame.pack(pady=10)
        
        ttk.Label(input_frame, text="Subject Name:", font=(FONT, 12)).pack(side="left", padx=5)
        self.subject_entry = ttk.Entry(input_frame, font=(FONT, 12), width=25)
        self.subject_entry.pack(side="left", padx=5)
        
        ttk.Button(input_frame, text="Add Subject", command=self.add_subject).pack(side="left", padx=5)
        
        self.subject_tv = ttk.Treeview(self.viewer_frame, columns=["Subject"], show="headings")
        self.subject_tv.heading("Subject", text="Subject")
        self.subject_tv.pack(fill="both", expand=True, padx=12, pady=12)
        
        ttk.Button(self.viewer_frame, text="Delete Selected Subject", command=self.delete_subject).pack(pady=10)
        ttk.Button(self.viewer_frame, text="⬅️ Back", command=self.show_main_dashboard).pack(pady=10)

        self.load_subjects()
        
    def load_subjects(self):
        """Loads and displays the list of subjects."""
        self.subject_tv.delete(*self.subject_tv.get_children())
        try:
            df = pd.read_csv(SUBJECT_CSV)
            for _, row in df.iterrows():
                self.subject_tv.insert("", "end", values=[row['Subject']])
        except Exception:
            pass

    def add_subject(self):
        """Adds a new subject to the subjects.csv file."""
        subject = self.subject_entry.get().strip()
        if not subject:
            messagebox.showerror("Error", "Subject name cannot be empty.")
            return
        
        new_file = not os.path.exists(SUBJECT_CSV)
        with open(SUBJECT_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["Subject"])
            writer.writerow([subject])
        
        self.subject_entry.delete(0, tk.END)
        self.load_subjects()
        messagebox.showinfo("Success", f"Subject '{subject}' added.")

    def delete_subject(self):
        """Deletes a selected subject from the subjects.csv file."""
        selected_item = self.subject_tv.selection()
        if not selected_item:
            messagebox.showerror("Error", "Please select a subject to delete.")
            return

        subject_to_delete = self.subject_tv.item(selected_item, 'values')[0]
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete '{subject_to_delete}'?"):
            df = pd.read_csv(SUBJECT_CSV)
            df = df[df['Subject'] != subject_to_delete]
            df.to_csv(SUBJECT_CSV, index=False)
            self.load_subjects()
            messagebox.showinfo("Success", f"Subject '{subject_to_delete}' deleted.")

    def manage_students(self):
        """Displays the UI for managing student details."""
        for widget in self.viewer_frame.winfo_children():
            widget.destroy()
        
        tk.Label(self.viewer_frame, text="Manage Student Details", font=(FONT, 18, "bold"), bg=BG_COLOR).pack(pady=10)
        
        btn_frame = ttk.Frame(self.viewer_frame)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Edit Student", command=self.edit_student).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Delete Student", command=self.delete_student).pack(side="left", padx=5)

        self.student_tv = ttk.Treeview(self.viewer_frame, columns=["Enrollment", "Name", "Date", "Time"], show="headings")
        for c in ["Enrollment", "Name", "Date", "Time"]:
            self.student_tv.heading(c, text=c)
            self.student_tv.column(c, width=160, anchor="center")
        self.student_tv.pack(fill="both", expand=True, padx=8, pady=8)
        
        ttk.Button(self.viewer_frame, text="⬅️ Back", command=self.show_main_dashboard).pack(pady=10)

        self.load_student_details()

    def load_student_details(self):
        """Loads and displays student data from the CSV file."""
        self.student_tv.delete(*self.student_tv.get_children())
        if not os.path.exists(STUDENT_CSV):
            messagebox.showerror("Not Found", f"{STUDENT_CSV} not found yet.")
            return
        
        try:
            # Bug fix: Read 'Enrollment' column as a string to avoid type errors
            df = pd.read_csv(STUDENT_CSV, dtype={'Enrollment': str})
            for _, r in df.iterrows():
                self.student_tv.insert("", "end", values=r.tolist())
        except Exception as e:
            messagebox.showerror("File Error", f"Could not read {STUDENT_CSV}:\n{e}")

    def edit_student(self):
        """Opens a dialog to edit a student's details."""
        selected_item = self.student_tv.selection()
        if not selected_item:
            messagebox.showerror("Error", "Please select a student to edit.")
            return
        
        values = self.student_tv.item(selected_item, 'values')
        current_enrollment = values[0]

        top = tk.Toplevel(self.controller.master)
        top.title("Edit Student")
        top.geometry("400x250")
        
        form_frame = ttk.Frame(top)
        form_frame.pack(pady=20, padx=20)

        ttk.Label(form_frame, text="New Enrollment:", font=(FONT, 12)).grid(row=0, column=0, pady=5)
        new_enr_entry = ttk.Entry(form_frame, font=(FONT, 12))
        new_enr_entry.insert(0, values[0])
        new_enr_entry.grid(row=0, column=1, pady=5)

        ttk.Label(form_frame, text="New Name:", font=(FONT, 12)).grid(row=1, column=0, pady=5)
        new_name_entry = ttk.Entry(form_frame, font=(FONT, 12))
        new_name_entry.insert(0, values[1])
        new_name_entry.grid(row=1, column=1, pady=5)

        def save_changes():
            """Saves the edited student details to the CSV file."""
            new_enr = new_enr_entry.get().strip()
            new_name = new_name_entry.get().strip()
            if not new_enr or not new_name:
                messagebox.showerror("Error", "All fields must be filled.")
                return

            try:
                # Bug fix: Ensure Enrollment is treated as string for comparison
                df = pd.read_csv(STUDENT_CSV, dtype={'Enrollment': str})
                df.loc[df['Enrollment'] == current_enrollment, 'Enrollment'] = new_enr
                df.loc[df['Enrollment'] == new_enr, 'Name'] = new_name
                df.to_csv(STUDENT_CSV, index=False)
                messagebox.showinfo("Success", "Student details updated.")
                self.load_student_details()
                top.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Could not save changes: {e}")

        ttk.Button(top, text="Save Changes", command=save_changes).pack(pady=10)


    def delete_student(self):
        """Deletes a selected student and their corresponding images."""
        selected_item = self.student_tv.selection()
        if not selected_item:
            messagebox.showerror("Error", "Please select a student to delete.")
            return

        values = self.student_tv.item(selected_item, 'values')
        enrollment_to_delete = values[0]
        name_to_delete = values[1]

        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete {name_to_delete} ({enrollment_to_delete})?"):
            try:
                # Bug fix: Ensure Enrollment is treated as string for comparison
                df = pd.read_csv(STUDENT_CSV, dtype={'Enrollment': str})
                df = df[df['Enrollment'] != enrollment_to_delete]
                df.to_csv(STUDENT_CSV, index=False)
                
                # Delete associated training images
                for filename in os.listdir(TRAINING_DIR):
                    if f".{enrollment_to_delete}." in filename:
                        os.remove(os.path.join(TRAINING_DIR, filename))

                messagebox.showinfo("Success", "Student and associated images deleted.")
                self.load_student_details()
            except Exception as e:
                messagebox.showerror("Error", f"Could not delete student: {e}")


class ManageTeachersPage(ttk.Frame):
    """Admin page for managing teacher accounts."""
    def __init__(self, parent, controller):
        super().__init__(parent, padding=25)
        self.controller = controller

        self.viewer_frame = ttk.Frame(self)
        self.viewer_frame.pack(fill="both", expand=True, padx=8, pady=8)
        self.show_teacher_management()

    def show_teacher_management(self):
        """Displays the UI for managing teachers."""
        for widget in self.viewer_frame.winfo_children():
            widget.destroy()

        tk.Label(self.viewer_frame, text="Manage Teachers", font=(FONT, 18, "bold"), bg=BG_COLOR).pack(pady=10)

        main_buttons_frame = ttk.Frame(self.viewer_frame)
        main_buttons_frame.pack(pady=10)
        
        ttk.Button(main_buttons_frame, text="Add New Teacher", command=self.add_teacher).pack(side="left", padx=5)
        ttk.Button(main_buttons_frame, text="Import from CSV", command=self.import_teachers_from_csv).pack(side="left", padx=5)
        ttk.Button(main_buttons_frame, text="Delete Selected", command=self.delete_teacher).pack(side="left", padx=5)

        self.teacher_tv = ttk.Treeview(self.viewer_frame, columns=["ID", "Password"], show="headings")
        self.teacher_tv.heading("ID", text="ID")
        self.teacher_tv.heading("Password", text="Password")
        self.teacher_tv.pack(fill="both", expand=True, padx=8, pady=8)

        ttk.Button(self.viewer_frame, text="⬅️ Back", command=self.controller.go_back).pack(pady=10)

        self.load_teachers()

    def load_teachers(self):
        """Loads and displays the list of teachers."""
        self.teacher_tv.delete(*self.teacher_tv.get_children())
        if not os.path.exists(TEACHER_CSV):
            messagebox.showerror("Error", "Teacher database file not found.")
            return
        
        try:
            # Bug fix: Read ID and Password as strings
            df = pd.read_csv(TEACHER_CSV, dtype={'ID': str, 'Password': str})
            for _, row in df.iterrows():
                self.teacher_tv.insert("", "end", values=[row['ID'], row['Password']])
        except Exception as e:
            messagebox.showerror("File Error", f"Could not read {TEACHER_CSV}:\n{e}")

    def add_teacher(self):
        """Opens a dialog to add a new teacher."""
        top = tk.Toplevel(self.controller.master)
        top.title("Add New Teacher")
        top.geometry("350x200")

        input_frame = ttk.Frame(top, padding=15)
        input_frame.pack(pady=10)

        ttk.Label(input_frame, text="ID:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        id_entry = ttk.Entry(input_frame)
        id_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        ttk.Label(input_frame, text="Password:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        pw_entry = ttk.Entry(input_frame)
        pw_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        def save_teacher():
            """Saves the new teacher to the CSV file."""
            new_id = id_entry.get().strip()
            new_pw = pw_entry.get().strip()
            if not new_id or not new_pw:
                messagebox.showerror("Error", "Both fields are required.")
                return

            new_file = not os.path.exists(TEACHER_CSV)
            with open(TEACHER_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                if new_file:
                    writer.writerow(["ID", "Password"])
                writer.writerow([new_id, new_pw])

            messagebox.showinfo("Success", "Teacher added successfully.")
            self.load_teachers()
            top.destroy()

        ttk.Button(top, text="Save", command=save_teacher).pack(pady=10)

    def delete_teacher(self):
        """Deletes a selected teacher from the CSV file."""
        selected_item = self.teacher_tv.selection()
        if not selected_item:
            messagebox.showerror("Error", "Please select a teacher to delete.")
            return
        
        teacher_id = self.teacher_tv.item(selected_item, 'values')[0]
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete teacher with ID '{teacher_id}'?"):
            try:
                # Bug fix: Read ID as string for comparison
                df = pd.read_csv(TEACHER_CSV, dtype={'ID': str})
                df = df[df['ID'] != teacher_id]
                df.to_csv(TEACHER_CSV, index=False)
                self.load_teachers()
                messagebox.showinfo("Success", "Teacher deleted.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete: {e}")

    def import_teachers_from_csv(self):
        """Imports teachers from a CSV file."""
        file_path = filedialog.askopenfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        try:
            # Bug fix: Read ID and Password as strings
            import_df = pd.read_csv(file_path, dtype={'ID': str, 'Password': str})
            if 'ID' not in import_df.columns or 'Password' not in import_df.columns:
                messagebox.showerror("File Error", "CSV must contain 'ID' and 'Password' columns.")
                return
            
            existing_df = pd.read_csv(TEACHER_CSV, dtype={'ID': str, 'Password': str})
            combined_df = pd.concat([existing_df, import_df], ignore_index=True)
            combined_df.drop_duplicates(subset=['ID'], keep='first', inplace=True)
            combined_df.to_csv(TEACHER_CSV, index=False)
            
            self.load_teachers()
            messagebox.showinfo("Import Successful", f"{len(import_df)} teachers imported.")
        except Exception as e:
            messagebox.showerror("Import Error", f"An error occurred during import: {e}")


# -----------------------------------------------------------
# Main Execution Block
# -----------------------------------------------------------
if __name__ == "__main__":
    app = App(window)
    window.protocol("WM_DELETE_WINDOW", lambda: app.master.destroy())
    window.mainloop()