import os
import json
import argparse
import tkinter as tk
from tkinter import Tk, Label, Button, Frame, Text, messagebox, ttk
from PIL import Image, ImageTk

class CorrectionApp:
    def __init__(self, root, input_json, output_json, label_map):
        self.root = root
        self.root.title("APT Caption Correction Tool")
        # Increase window size to accommodate larger elements
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b') 
        
        self.input_json = input_json
        self.output_json = output_json
        self.label_map = label_map
        
        # Load data
        with open(self.input_json, 'r') as f:
            self.data = json.load(f)
            
        self.current_index = 0
        self.total_images = len(self.data)
        self.edit_mode = False
        
        if self.total_images == 0:
            messagebox.showinfo("Info", "No images to process.")
            self.root.destroy()
            return

        self.setup_ui()
        self.display_current()

    def setup_ui(self):
        # Main container with padding (Req 10)
        self.main_frame = Frame(self.root, padx=20, pady=20, bg='#2b2b2b')
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top info bar
        self.info_frame = Frame(self.main_frame, bg='#2b2b2b')
        self.info_frame.pack(fill=tk.X, pady=10)
        
        # Reduced font size (Req: 0.6x of 28 -> ~18)
        self.index_label = Label(self.info_frame, text="", font=('Helvetica', 18, 'bold'), bg='#2b2b2b', fg='#ffffff')
        self.index_label.pack(side=tk.LEFT)
        
        # Req 9: Image name hidden (removed filename_label)

        # Image area
        self.image_frame = Frame(self.main_frame, bg='#2b2b2b')
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.image_label = Label(self.image_frame, bg='#2b2b2b', fg='#ffffff')
        self.image_label.pack()

        # Controls area
        self.controls_frame = Frame(self.main_frame, bg='#2b2b2b')
        self.controls_frame.pack(fill=tk.X, pady=10)

        # Label selection
        Label(self.controls_frame, text="Class Label:", font=('Helvetica', 18), bg='#2b2b2b', fg='#ffffff').pack(side=tk.LEFT)
        self.label_var = tk.StringVar()
        
        # Req: Replace OptionMenu with Radiobuttons
        # Container for radio buttons
        self.radio_frame = Frame(self.controls_frame, bg='#2b2b2b')
        self.radio_frame.pack(side=tk.LEFT, padx=20)
        
        if self.label_map:
            self.label_var.set(self.label_map[0])
            
        for label in self.label_map:
            rb = tk.Radiobutton(
                self.radio_frame, 
                text=label, 
                variable=self.label_var, 
                value=label,
                command=lambda: self.on_label_change(None),
                font=('Helvetica', 18),
                bg='#2b2b2b',
                fg='#ffffff',
                selectcolor='#4b4b4b', # Darker background when selected
                activebackground='#2b2b2b',
                activeforeground='#ffffff'
            )
            rb.pack(side=tk.LEFT, padx=10)

        # Rationale Area
        Label(self.main_frame, text="Rationale (Cmd+E to Edit):", font=('Helvetica', 18), bg='#2b2b2b', fg='#ffffff').pack(anchor=tk.W, pady=(10, 5))
        
        # Container for rationale to switch between Read/Edit
        self.rationale_frame = Frame(self.main_frame, bg='#2b2b2b')
        self.rationale_frame.pack(fill=tk.BOTH, expand=True)
        
        # Read-only Text widget (better for copying than Label) - Req 5, 6
        self.rationale_read = Text(
            self.rationale_frame, 
            height=8, 
            font=('Helvetica', 18), 
            wrap=tk.WORD,
            bg="#3c3f41",
            fg="#ffffff",
            insertbackground="#ffffff",
            relief=tk.FLAT
        )
        self.rationale_read.pack(fill=tk.BOTH, expand=True)
        self.rationale_read.config(state=tk.DISABLED) # Read-only
        
        # Editable Text widget - Req 7
        self.rationale_edit = Text(
            self.rationale_frame, 
            height=8, 
            font=('Helvetica', 18), 
            wrap=tk.WORD,
            bg="#3c3f41",
            fg="#ffffff",
            insertbackground="#ffffff",
            relief=tk.SUNKEN,
            undo=True # Req 7: Undo support
        )
        # Initially hidden
        
        # Bind shortcuts
        self.root.bind('<Control-e>', self.toggle_edit_mode)
        self.root.bind('<Command-e>', self.toggle_edit_mode)
        self.root.bind('<Right>', self.next_image)
        self.root.bind('<Left>', self.prev_image)
        
        # Save shortcuts
        self.root.bind('<Control-s>', self.finish)
        self.root.bind('<Command-s>', self.finish)
        
        # Ensure standard shortcuts work (Mac specific sometimes needed)
        self.enable_text_shortcuts(self.rationale_edit)
        self.enable_text_shortcuts(self.rationale_read) # For copy

        # Navigation Buttons
        self.nav_frame = Frame(self.main_frame, bg='#2b2b2b')
        self.nav_frame.pack(fill=tk.X, pady=20)
        
        # Req 2: Fix buttons (ensure commands are bound correctly)
        # Req 4: Reduced size (0.6x of 24 -> ~14)
        Button(self.nav_frame, text="< Previous", command=self.prev_image, font=('Helvetica', 14)).pack(side=tk.LEFT)
        
        # Save hint label instead of button
        Label(self.nav_frame, text="Press Cmd+S to Save & Finish", font=('Helvetica', 14), bg='#2b2b2b', fg='#aaaaaa').pack(side=tk.RIGHT, padx=20)
        
        Button(self.nav_frame, text="Next >", command=self.next_image, font=('Helvetica', 14)).pack(side=tk.RIGHT, padx=20)

    def enable_text_shortcuts(self, widget):
        # Basic bindings for Mac if not present
        # Cmd+C, Cmd+V, Cmd+X, Cmd+A, Cmd+Z
        def select_all(event):
            widget.tag_add("sel", "1.0", "end")
            return "break"
            
        widget.bind("<Command-a>", select_all)
        # Copy/Paste/Cut/Undo are usually handled by OS on Mac for Text widgets if focus is right.
        # But we can enforce if needed. Usually Tkinter on Mac handles Cmd+C/V/X/Z natively in Text widgets.

    def toggle_edit_mode(self, event=None):
        if self.edit_mode:
            # Switch to Read Mode
            # Save text from edit widget to data
            new_text = self.rationale_edit.get("1.0", "end-1c").strip()
            self.data[self.current_index]['rationale'] = new_text
            
            # Update Read widget
            self.rationale_read.config(state=tk.NORMAL)
            self.rationale_read.delete("1.0", tk.END)
            self.rationale_read.insert("1.0", new_text)
            self.rationale_read.config(state=tk.DISABLED)
            
            # Swap widgets
            self.rationale_edit.pack_forget()
            self.rationale_read.pack(fill=tk.BOTH, expand=True)
            
            self.edit_mode = False
            # Return focus to root so arrows work
            self.root.focus_set()
        else:
            # Switch to Edit Mode
            # Get text from data
            current_text = self.data[self.current_index].get('rationale', '')
            
            # Update Edit widget
            self.rationale_edit.delete("1.0", tk.END)
            self.rationale_edit.insert("1.0", current_text)
            
            # Swap widgets
            self.rationale_read.pack_forget()
            self.rationale_edit.pack(fill=tk.BOTH, expand=True)
            self.rationale_edit.focus_set()
            
            self.edit_mode = True

    def display_current(self):
        # Ensure we are in read mode when switching images
        if self.edit_mode:
            self.toggle_edit_mode() # Save and switch back
            
        item = self.data[self.current_index]
        
        # Update Info
        self.index_label.config(text=f"Image {self.current_index + 1} / {self.total_images}")
        
        # Update Image - Req 1: Reduced size (0.75x of 800 -> 600)
        try:
            img = Image.open(item['image_path'])
            # Resize to fit reasonable area, e.g., max height 450
            base_height = 620
            h_percent = (base_height / float(img.size[1]))
            w_size = int((float(img.size[0]) * float(h_percent)))
            img = img.resize((w_size, base_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo
        except Exception as e:
            self.image_label.config(image='', text=f"Error loading image: {e}")

        # Update Label
        current_label = item.get('label', '')
        if current_label in self.label_map:
            self.label_var.set(current_label)
        else:
            if self.label_map:
                self.label_var.set(self.label_map[0]) 
            
        # Update Rationale (Read Mode)
        self.rationale_read.config(state=tk.NORMAL)
        self.rationale_read.delete('1.0', tk.END)
        self.rationale_read.insert('1.0', item.get('rationale', ''))
        self.rationale_read.config(state=tk.DISABLED)

    def save_current_label(self):
        # Save label selection (rationale is saved on toggle or navigation if we add that logic)
        # Actually, let's ensure rationale is saved if user navigates while in edit mode
        if self.edit_mode:
             new_text = self.rationale_edit.get("1.0", "end-1c").strip()
             self.data[self.current_index]['rationale'] = new_text
             
        self.data[self.current_index]['label'] = self.label_var.get()

    def next_image(self, event=None):
        if self.edit_mode:
            return # Block navigation in edit mode
            
        self.save_current_label()
        if self.current_index < self.total_images - 1:
            self.current_index += 1
            self.display_current()
        else:
            # Optional: Loop or stop
            pass

    def prev_image(self, event=None):
        if self.edit_mode:
            return # Block navigation in edit mode
            
        self.save_current_label()
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current()

    def on_label_change(self, event):
        self.save_current_label()

    def finish(self, event=None):
        self.save_current_label()
        # Write to output file
        with open(self.output_json, 'w') as f:
            json.dump(self.data, f, indent=4)
        print(f"Saved corrections to {self.output_json}")
        self.root.destroy()

def parse_args():
    parser = argparse.ArgumentParser(description="APT Caption Correction Tool")
    parser.add_argument("--input_json", required=True, help="Path to input JSON file")
    parser.add_argument("--output_json", required=True, help="Path to output JSON file")
    parser.add_argument("--label_map", nargs='+', required=True, help="List of valid labels")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    root = Tk()
    app = CorrectionApp(root, args.input_json, args.output_json, args.label_map)
    root.mainloop()
