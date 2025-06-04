from tkinter import messagebox
import sqlite3
import glob
import customtkinter 
import os
from PIL import ImageTk
import nibabel 
from PIL import Image
import numpy 
import pyvista
from tkinter import ttk

pd1 = None
pd2 = None
pd3 = None
pl = None

customtkinter.set_appearance_mode("light")
customtkinter.set_default_color_theme("blue")

def showPDPopup(ent):
    sltd = patientTBL.selection()
    if not sltd:
        return

    vals = patientTBL.item(sltd[0])['values']
    PopUP = customtkinter.CTkToplevel()
    PopUP.title("Patient Details")
    PopUP.geometry("500x400")

    customtkinter.CTkLabel(PopUP, text="Patient Details", font=("Segoe UI", 16, "bold")).pack(pady=10)

    scrollFrm = customtkinter.CTkScrollableFrame(PopUP, width=450, height=300)
    scrollFrm.pack(pady=10, padx=10)

    lbls = ["Patient ID", "Name", "Age", "Gender", "Comments"]
    for i, lbl in enumerate(lbls):
        text = f"{lbl}: {vals[i]}"
        customtkinter.CTkLabel(scrollFrm, text=text, anchor="w", justify="left", wraplength=400).pack(anchor="w", pady=5, padx=10)

def initialDB():
    cnct = sqlite3.connect("patients.db")
    csr = cnct.cursor()
    csr.execute("""CREATE TABLE IF NOT EXISTS patients (id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT UNIQUE NOT NULL, name TEXT, age INTEGER, gender TEXT, comments TEXT)""")
    cnct.commit()
    cnct.close()

def opn3drecon(pid):
    
    vtkFile = os.path.join(r"figures",
        f"{pid}.vtk")

    if os.path.exists(vtkFile):
        msh = pyvista.read(vtkFile)
        pltr = pyvista.Plotter()
        pltr.add_mesh(msh, color="red", opacity=0.8)
        pltr.add_axes()
        pltr.reset_camera()
        pltr.show(title=f"3D Reconstruction for patient ID {pid}")
    else:
        messagebox.showerror("File error", f"3D reconstruction file not found for patient ID {pid}.")





def createImgfrmnifti(niftiIMG):
    try:
        if niftiIMG is None:
            print("NIfTI image is None")
            return None

        dt = niftiIMG.get_fdata()

        if dt.size == 0:
            print("NIfTI image data is empty")
            return None

        if dt.ndim == 2:
            slc2d = dt
        elif dt.ndim == 3:
            slc2d = dt[:, :, dt.shape[2] // 2]
        else:
            print("Unsupported NIfTI dimensions:", dt.shape)
            return None

        slc2d = numpy.interp(slc2d, (slc2d.min(), slc2d.max()), (0, 255)).astype(numpy.uint8)
        hist, bins = numpy.histogram(slc2d.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdfmskd = numpy.ma.masked_equal(cdf, 0)
        cdfmskd = (cdfmskd - cdfmskd.min()) * 255 / (cdfmskd.max() - cdfmskd.min())
        cdffnl = numpy.ma.filled(cdfmskd, 0).astype('uint8')
        slc2d = cdffnl[slc2d]

        img = Image.fromarray(slc2d)
        img = img.resize((300, 300))
        imgtk = ImageTk.PhotoImage(img)
        return imgtk
    except Exception as e:
        print(f"Error converting image: {str(e)}")
        return None
def updPatientTable():
    try:
        cnct = sqlite3.connect("patients.db")
        csr = cnct.cursor()
        csr.execute("SELECT * FROM patients ORDER BY CAST(patient_id AS INTEGER) ASC") # here the results are ordered by the patient id in ascending order
        rec = csr.fetchall()
        cnct.close()

        for row in patientTBL.get_children():
            patientTBL.delete(row)

        for record in rec:
            patientTBL.insert("", "end", values=record[1:])

    
        if rec:
            firstpid = rec[0][1]
            pf = fetchPF(firstpid)
            if pf:
                pl.configure(text="\n".join([file[0] for file in pf]))

            else:
                pl.configure(text="No projections found for this patient.")
        else:
            pl.configure(text="No patients available.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to retrieve data: {str(e)}")
def savePD():
    """function for saving the patient details."""
    try:
        pid = pidEntry.get().strip()
        nm = " ".join(word.capitalize() for word in nmEntry.get().strip().split())
        age = ageEntry.get().strip()
        gndr = gndtEntry.get().strip()
        cmnts = cmntEntry.get("1.0", "end").strip()
        if cmnts == "Enter comments here...":
            cmnts = ""

        if not pid or not nm or not age or not cmnts:
            messagebox.showerror("Input Error", "Please fill all the details.")
            return

        cnct = sqlite3.connect("patients.db")
        csr = cnct.cursor()
        csr.execute("INSERT INTO patients (patient_id, name, age, gender, comments) VALUES (?, ?, ?, ?, ?)", 
                       (str(pid), str(nm), int(age), str(gndr), str(cmnts)))
        cnct.commit()
        messagebox.showinfo("Success", "Patient details saved successfully.")

        updPatientTable()
        
        pidEntry.delete(0, "end")
        nmEntry.delete(0, "end")
        ageEntry.delete(0, "end")
        gndtEntry.set("Male")
        cmntEntry.delete("1.0", "end")
        cmntEntry.insert("1.0", "Enter comments here...")
        cmntEntry.configure(text_color="gray")
        
    except sqlite3.IntegrityError:
        messagebox.showerror("Duplicate Error", "The Patient ID already exists. Please enter a unique ID.")
    except Exception as e:
        messagebox.showerror("Database Error", f"Failed to save details: {str(e)}")
    finally:
        if cnct:
            cnct.close()



def fetchPF(pid):
    try:
        pid = str(pid).strip()
        path = r"selected_Projection"
        if not os.path.exists(path):
            messagebox.showerror("File Error", f"Directory not found: {path}")
            return ["Error: Directory not found."]

        ptrn = os.path.join(path, f"{pid}.img.proj*.nii.gz")

        # the following is used to find all matching files
        fls = glob.glob(ptrn)
        if not fls:
            return ["No projection files found."]

        imgs = []
        for file in fls:
            try:
                img = nibabel.load(file)
                imgs.append((os.path.basename(file), img))
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                imgs.append((os.path.basename(file), None))

        return imgs
    except Exception as e:
        messagebox.showerror("File Error", f"Unable to load projections: {str(e)}")
        return ["Error fetching files."]
def onFocusOut(ent):
    if cmntEntry.get("1.0", "end-1c").strip() == "":
        cmntEntry.insert("1.0", "Enter comments here...")
        cmntEntry.configure(text_color="gray")

def onPatientSelect(ent):
    global pl, selectedPN, selectedpatientID, pd1, pd2

    try:
        sltd = patientTBL.focus()
        if not sltd:
            return
        pid = str(patientTBL.item(sltd)['values'][0])
        pname = patientTBL.item(sltd)['values'][1]
        selectedPN.configure(text=f"Patient Name: {pname}")
        selectedpatientID.configure(text=f"Patient ID: {pid}")
        pf = fetchPF(pid) # used for fetching and displaying all projection files

        for dply in [pd1, pd2, pd3]:
            if dply is not None:
                dply.configure(image=None)
                dply.image = None
                
                
        imgs = [createImgfrmnifti(img[1]) for img in pf[:3]] # it displays the projection in the visualization tab
        if imgs and len(imgs) > 0 and imgs[0]:
            pd1.configure(image=imgs[0])
            pd1.image = imgs[0]
        

        if len(imgs) > 1 and imgs[1]:
            pd2.configure(image=imgs[1])
            pd2.image = imgs[1]
        

        if len(imgs) > 2 and imgs[2]:
            pd3.configure(image=imgs[2])
            pd3.image = imgs[2]
        

        #the following is used in the patient details tab for updating the projections list
        if pf:
            pl.configure(text="\n".join([file[0] for file in pf]))
        else:
            pl.configure(text="No projections found for this patient.")
    except Exception as e:
        messagebox.showerror("Selection Error", f"Error fetching projection files: {str(e)}")

def onFocusIn(ent):
    if cmntEntry.get("1.0", "end-1c") == "Enter comments here...":
        cmntEntry.delete("1.0", "end")
        cmntEntry.configure(text_color="black")


def lgn():
    usrnm = usrnmEntry.get().strip()
    pswrd = pswrdEntry.get().strip()

    if usrnm in usrCred and pswrd == usrCred[usrnm]:
        appltn.withdraw()
        fmtNM = " ".join(part.capitalize() for part in usrnm.strip().split())
        openDashboard(fmtNM)
    else:
        messagebox.showerror("Access Denied", "Invalid credentials. Please try again.")
def openDashboard(usrnm):
    global pidEntry, nmEntry, ageEntry, gndtEntry, cmntEntry, patientTBL, pl, selectedPN, selectedpatientID, pd1, pd2, pd3
    dshbrd = customtkinter.CTkToplevel()
    dshbrd.title("Arterix 3D Dashboard")
    dshbrd.geometry("1200x800")
    dshbrd.configure(fg_color="#f6fafd")

    hdr = customtkinter.CTkLabel(dshbrd, text=f"Welcome, Dr. {usrnm}", font=("Segoe UI Semibold", 24))
    hdr.pack(pady=10)

    tbView = customtkinter.CTkTabview(dshbrd, width=1100, height=700)
    tbView.pack(pady=10)

    pdtab = tbView.add("Patient Details")
    vstab = tbView.add("Visualization")

    visscr = customtkinter.CTkScrollableFrame(vstab, width=1080, height=650, corner_radius=20)
    visscr.pack(padx=10, pady=10, fill="both", expand=True)

    hdrSec = customtkinter.CTkFrame(visscr, fg_color="white", corner_radius=15)
    hdrSec.pack(fill="x", padx=20, pady=(10, 20))

    hdrInner = customtkinter.CTkFrame(hdrSec, fg_color="transparent")
    hdrInner.pack(fill="x", padx=20, pady=10)

    selectedPN = customtkinter.CTkLabel(hdrInner, text="Patient Name: None", font=("Segoe UI", 16), text_color="#0f2c4c")
    selectedPN.pack(side="left", anchor="w")

    selectedpatientID = customtkinter.CTkLabel(hdrInner, text="Patient ID: None", font=("Segoe UI", 16), text_color="#0f2c4c")
    selectedpatientID.pack(side="right", anchor="e")

   
    projGrid = customtkinter.CTkFrame(visscr, fg_color="transparent")
    projGrid.pack(padx=20, pady=10)

    topRow = customtkinter.CTkFrame(projGrid, fg_color="transparent")
    topRow.pack()

    proj1Card = customtkinter.CTkFrame(topRow, fg_color="white", corner_radius=20)
    proj1Card.pack(side="left", padx=40, pady=10)
    customtkinter.CTkLabel(proj1Card, text="Projection View 1", font=("Segoe UI Semibold", 16), text_color="#0e3a59").pack(pady=(15, 5))
    pd1 = customtkinter.CTkLabel(proj1Card, text="", width=300, height=300, anchor="center",
                                fg_color="#e8f0f7", corner_radius=10)
    pd1.pack(padx=20, pady=(0, 20))

    proj2Card = customtkinter.CTkFrame(topRow, fg_color="white", corner_radius=20)
    proj2Card.pack(side="right", padx=40, pady=10)
    customtkinter.CTkLabel(proj2Card, text="Projection View 2", font=("Segoe UI Semibold", 16), text_color="#0e3a59").pack(pady=(15, 5))
    pd2 = customtkinter.CTkLabel(proj2Card, text="", width=300, height=300, anchor="center",
                                fg_color="#e8f0f7", corner_radius=10)
    pd2.pack(padx=20, pady=(0, 20))

    proj3Card = customtkinter.CTkFrame(projGrid, fg_color="white", corner_radius=20)
    proj3Card.pack(pady=10)
    customtkinter.CTkLabel(proj3Card, text="Projection View 3", font=("Segoe UI Semibold", 16), text_color="#0e3a59").pack(pady=(15, 5))
    pd3 = customtkinter.CTkLabel(proj3Card, text="", width=640, height=300, anchor="center",
                                fg_color="#e8f0f7", corner_radius=10)
    pd3.pack(padx=20, pady=(0, 20))
    vtkbtn = customtkinter.CTkButton(
        visscr,
        text="View 3D Reconstruction of Coronary Artery",
        command=lambda: onvtkbuttonclick(),
        width=300
    )
    vtkbtn.pack(pady=10)

    def onvtkbuttonclick():
        sltdItem = patientTBL.focus()
        if not sltdItem:
            messagebox.showwarning("No Patient Selected", "Please select a patient to view the 3D reconstruction.")
            return
        pid = str(patientTBL.item(sltdItem)['values'][0])
        opn3drecon(pid)

    # the help tab
    hlpTab = tbView.add("Help")
    hlpscrl = customtkinter.CTkScrollableFrame(hlpTab, width=1000, height=600, corner_radius=20)
    hlpscrl.pack(padx=30, pady=20, fill="both", expand=True)

    customtkinter.CTkLabel(hlpscrl, text="🩺 Help & User Guide", font=("Segoe UI Bold", 28), text_color="#123456").pack(anchor="center", pady=(10, 30))

    ppsFrame = customtkinter.CTkFrame(hlpscrl, fg_color="#ffffff", corner_radius=12)
    ppsFrame.pack(pady=15, padx=20, fill="x", expand=False)
    customtkinter.CTkLabel(ppsFrame, text="🌍 Purpose of the Application", font=("Segoe UI Semibold", 20), text_color="#0f2c4c").pack(anchor="w", padx=20, pady=(15, 5))
    customtkinter.CTkLabel(ppsFrame, text=(
        "This application is developed and designed to help the cardiologists or cardiac surgeons to plan "
        "the procedures that they are going to perform on the patients to treat their coronary artery disease."
        " By using just three 2D X-ray image projections, the AI model that runs in the backend of this application "
        "creates a 3D reconstruction of the coronary artery and display it through application. "
        "This interactive 3D reconstruction of the coronary artery helps the doctors or the cardiac surgeons "
        "to plan and execute the procedures such as percutaneous coronary intervention more safely, "
        "accurately and efficiently especially in the situations where the advanced imaging systems "
        "like CT scans are not available. Because of all this, the application is helpful in creating "
        "better outcomes and thereby reducing the risk of complications during the surgery. "
    ), font=("Segoe UI", 14), wraplength=900, justify="left", text_color="#333333").pack(padx=20, pady=(0, 15))

    htf = customtkinter.CTkFrame(hlpscrl, fg_color="#ffffff", corner_radius=12)
    htf.pack(pady=15, padx=20, fill="x", expand=False)
    customtkinter.CTkLabel(htf, text="🛠️ How to Use the Application", font=("Segoe UI Semibold", 20), text_color="#0f2c4c").pack(anchor="w", padx=20, pady=(15, 5))
    customtkinter.CTkLabel(htf, text="➊ Please do enter your username and password in the respective textbox and  then press the login button.", font=("Segoe UI", 14), text_color="#333333").pack(anchor="w", padx=20, pady=(5, 2))

    r1 = customtkinter.CTkFrame(htf, fg_color="transparent")
    r1.pack(anchor="w", fill="x", padx=20)
    customtkinter.CTkLabel(r1, text="➋ Make use of the ", font=("Segoe UI", 14), text_color="#333333").pack(side="left")
    customtkinter.CTkLabel(r1, text="Patient Details", font=("Segoe UI", 14, "bold"), text_color="#333333").pack(side="left")
    customtkinter.CTkLabel(r1, text=" tab to:", font=("Segoe UI", 14), text_color="#333333").pack(side="left")
    customtkinter.CTkLabel(htf, text="   - Enter patient information such as ID, Name, Age, Gender and Comments.", font=("Segoe UI", 14), text_color="#333333").pack(anchor="w", padx=40)
    customtkinter.CTkLabel(htf, text="   - Save the data to the database by pressing the 'Save' button.", font=("Segoe UI", 14), text_color="#333333").pack(anchor="w", padx=40)
    customtkinter.CTkLabel(htf, text="   - Select a patient from the table by a single click to view or delete their data (deleting the data is done by pressing the 'Delete' button).", font=("Segoe UI", 14), text_color="#333333").pack(anchor="w", padx=40)
    customtkinter.CTkLabel(htf, text="   - Double click to open a pop up window with full patient details to have a view of it easily.", font=("Segoe UI", 14), text_color="#333333").pack(anchor="w", padx=40)


    r2 = customtkinter.CTkFrame(htf, fg_color="transparent")
    r2.pack(anchor="w", fill="x", padx=20, pady=(10, 2))
    customtkinter.CTkLabel(r2, text="➌ In the ", font=("Segoe UI", 14), text_color="#333333").pack(side="left")
    customtkinter.CTkLabel(r2, text="Visualization", font=("Segoe UI", 14, "bold"), text_color="#333333").pack(side="left")
    customtkinter.CTkLabel(r2, text=" tab you can:", font=("Segoe UI", 14), text_color="#333333").pack(side="left")
    customtkinter.CTkLabel(htf, text="   - see the selected patient’s name and the patinet's ID at the top.", font=("Segoe UI", 14), text_color="#333333").pack(anchor="w", padx=40)
    customtkinter.CTkLabel(htf, text="   - have a view of the 2D X-ray projections of the selected patients.", font=("Segoe UI", 14), text_color="#333333").pack(anchor="w", padx=40)
    customtkinter.CTkLabel(htf, text="   - click the button below the third 3D X-ray projection to generate an interactive 3D model of the coronary artery.", font=("Segoe UI", 14), text_color="#333333").pack(anchor="w", padx=40)
    
    customtkinter.CTkLabel(hlpscrl, text="📧 For assistance or feedback, please contact via the email id: arterix3d@techteam.com", font=("Segoe UI Italic", 13), text_color="#777777").pack(anchor="center", pady=30)

    customtkinter.CTkLabel(pdtab, text="Patient Registration", font=("Segoe UI Bold", 20)).pack(pady=(20, 10))

    pidEntry = customtkinter.CTkEntry(pdtab, placeholder_text="Patient ID", width=400)
    nmEntry = customtkinter.CTkEntry(pdtab, placeholder_text="Patient Name", width=400)
    ageEntry = customtkinter.CTkEntry(pdtab, placeholder_text="Age", width=400)
    gndtEntry = customtkinter.CTkOptionMenu(pdtab, values=["Male", "Female", "Other"], width=400)
    cmntEntry = customtkinter.CTkTextbox(pdtab, width=400, height=100)

    cmntEntry.insert("1.0", "Enter comments here...")
    cmntEntry.configure(text_color="gray")
    cmntEntry.bind("<FocusIn>", onFocusIn)
    cmntEntry.bind("<FocusOut>", onFocusOut)

    for wdgt in [pidEntry, nmEntry, ageEntry, gndtEntry, cmntEntry]:
        wdgt.pack(pady=6)

    btnr = customtkinter.CTkFrame(pdtab, fg_color="transparent")
    btnr.pack(pady=10)
    customtkinter.CTkButton(btnr, text="💾 Save", command=savePD, width=180).pack(side="left", padx=10)
    customtkinter.CTkButton(btnr, text="🗑️ Delete", command=dltPatientDetails, width=180).pack(side="right", padx=10)

    projectionFrame = customtkinter.CTkFrame(pdtab, fg_color="white", corner_radius=15)
    projectionFrame.pack(pady=20, padx=20, fill="x")
    customtkinter.CTkLabel(projectionFrame, text="🫀 Projection Files", font=("Segoe UI Semibold", 16)).pack(pady=(10, 5))
    pl = customtkinter.CTkLabel(projectionFrame, text="", wraplength=500, justify="left")
    pl.pack(pady=5)

    tblFrame = customtkinter.CTkFrame(pdtab, fg_color="transparent")

    tblFrame.pack(pady=20, padx=20, fill="both", expand=True)

    clms = ("Patient ID", "Name", "Age", "Gender", "Comments")
    patientTBL = ttk.Treeview(tblFrame, columns=clms, show="headings", height=10)

    sby = ttk.Scrollbar(tblFrame, orient="vertical", command=patientTBL.yview)
    patientTBL.configure(yscrollcommand=sby.set)
    sby.pack(side="right", fill="y")
    patientTBL.pack(side="left", fill="both", expand=True)

    for cl in clms:
        patientTBL.heading(cl, text=cl)
        patientTBL.column(cl, anchor="center", minwidth=80, width=100)

    patientTBL.bind("<ButtonRelease-1>", onPatientSelect)
    patientTBL.bind("<Double-1>", showPDPopup)

    updPatientTable()

    for wdgt in vstab.pack_slaves():
        if isinstance(wdgt, customtkinter.CTkLabel) and not wdgt.cget("text"):
            wdgt.pack_forget()



def dltPatientDetails():
    try:
        sltd = patientTBL.selection()
        if not sltd:
            messagebox.showwarning("Selection error", "Please select a patient to delete.")
            return

        rspn = messagebox.askyesno("Confirm deletion", "Are you sure that you want to delete the selected patient?")
        if not rspn:
            return

        pid = patientTBL.item(sltd[0])['values'][0]

        cnct = sqlite3.connect("patients.db")
        csr = cnct.cursor()
        csr.execute("DELETE FROM patients WHERE patient_id = ?", (pid,))
        cnct.commit()

        # This resets the auto increment value properly
        csr.execute("VACUUM")
        csr.execute("SELECT MAX(id) FROM patients")
        maxID = csr.fetchone()[0]

        # This handles the case when no records exists
        if maxID is None:
            maxID = 0

        # this is used for updating autoincrement counter correctly
        csr.execute("UPDATE SQLITE_SEQUENCE SET seq = ? WHERE name = 'patients'", (maxID,))
        cnct.commit()
        cnct.close()

        updPatientTable()
        messagebox.showinfo("Success", "Patient details deleted successfully.")
    except Exception as e:
        messagebox.showerror("Database Error", f"Failed to delete details: {str(e)}")

usrCred = {
    "tom": "doctom",
    "john": "medoff",
    "ben": "bencardiac",
    "jack": "123med"
}

appltn = customtkinter.CTk()
appltn.title("Arterix 3D")
appltn.geometry("500x600")
appltn.configure(fg_color="#f0f4f9")

brndFrm = customtkinter.CTkFrame(appltn, fg_color="transparent")
brndFrm.pack(pady=30)


customtkinter.CTkLabel(brndFrm, text="🫀 Arterix 3D", font=("Segoe UI Bold", 32), text_color="#22577a").pack()
customtkinter.CTkLabel(brndFrm, text="See Beyond, Plan Beyond", font=("Segoe UI", 14), text_color="#5584ac").pack()
lgnCrd = customtkinter.CTkFrame(appltn, width=400, height=300, corner_radius=20)
lgnCrd.pack(pady=40)
lgnCrd.pack_propagate(False)

customtkinter.CTkLabel(lgnCrd, text="Login to Continue", font=("Segoe UI Semibold", 20)).pack(pady=(20, 10))

usrnmEntry = customtkinter.CTkEntry(lgnCrd, placeholder_text="Username", width=250)
usrnmEntry.pack(pady=(10, 10))

pswrdEntry = customtkinter.CTkEntry(lgnCrd, placeholder_text="Password", show="*", width=250)
pswrdEntry.pack(pady=(10, 10))

lgnbtn = customtkinter.CTkButton(lgnCrd, text="Login", command=lgn, width=200, height=40)
lgnbtn.pack(pady=20)

initialDB()
customtkinter.CTkLabel(appltn, text="© Arterix 3D 2025", font=("Segoe UI", 12), text_color="#a3bfc9").pack(side="bottom", pady=15)
appltn.mainloop()