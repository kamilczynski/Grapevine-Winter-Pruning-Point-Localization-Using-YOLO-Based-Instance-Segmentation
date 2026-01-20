import os
import tkinter as tk
from openpyxl import Workbook
from tkinter import filedialog, messagebox, ttk


class YoloLabelCounterApp(tk.Tk):
    

    def __init__(self) -> None:
        super().__init__()

        
        self.title("YOLO Label Counter")
        self.geometry("420x320")
        self.resizable(False, False)

        
        self.dir_path = tk.StringVar()

     
        container = ttk.Frame(self, padding=10)
        container.pack(fill="both", expand=True)

        ttk.Label(container, text="Selected folder:").pack(anchor="w")
        ttk.Entry(container, textvariable=self.dir_path, state="readonly").pack(
            fill="x", pady=(0, 6)
        )

        btn_frame = ttk.Frame(container)
        btn_frame.pack(fill="x", pady=(0, 8))
        ttk.Button(btn_frame, text="Selected folder", command=self.choose_dir).pack(
            side="left", expand=True, fill="x"
        )
        ttk.Button(btn_frame, text="Count labels", command=self.count_labels).pack(
            side="right", expand=True, fill="x"
        )

      
        self.results = tk.Text(
            container,
            height=10,
            state="disabled",
            wrap="none",
            font=("Courier New", 10),
        )
        self.results.pack(fill="both", expand=True)

    

    def choose_dir(self) -> None:
        
        path = filedialog.askdirectory(title="Select the folder with YOLO labels")
        if path:
            self.dir_path.set(path)

    def count_labels(self) -> None:
        directory = self.dir_path.get()
        if not directory:
            messagebox.showwarning("Selected folder", "First, select a folder.")
            return

        counts_total: dict[str, int] = {}
        per_file_data = []

        for root, _dirs, files in os.walk(directory):
            for fname in files:
                if not fname.lower().endswith(".txt"):
                    continue

                file_path = os.path.join(root, fname)

                class0 = 0
                class1 = 0
                total = 0

                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        label = line.split()[0]

                        counts_total[label] = counts_total.get(label, 0) + 1

                        if label == "0":
                            class0 += 1
                        elif label == "1":
                            class1 += 1

                        total += 1

                per_file_data.append((fname, class0, class1, total))

        
        wb = Workbook()
        ws = wb.active
        ws.title = "YOLO summary"
        ws.append(["filename", "class_0", "class_1", "total"])

        for row in per_file_data:
            ws.append(row)

        excel_path = os.path.join(directory, "yolo_label_report.xlsx")
        wb.save(excel_path)

       
        self.results.config(state="normal")
        self.results.delete("1.0", tk.END)

        if counts_total:
            for label_id, qty in sorted(counts_total.items(), key=lambda t: int(t[0])):
                self.results.insert(tk.END, f"Label {label_id}: {qty}\n")

            self.results.insert(
                tk.END,
                f"\nReport saved Excel:\n{excel_path}"
            )
        else:
            self.results.insert(
                tk.END,
                "No YOLO labels found in the selected folder.",
            )

        self.results.config(state="disabled")



total_lines = 0
...


if __name__ == "__main__":
    app = YoloLabelCounterApp()
    app.mainloop()
