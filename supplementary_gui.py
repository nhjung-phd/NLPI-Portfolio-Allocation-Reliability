"""macOS-friendly Tk GUI for supplementary experiments."""
import subprocess, sys, tkinter as tk
from tkinter import ttk, messagebox

class App(tk.Tk):
 def __init__(self):
  super().__init__(); self.title("NLPI Supplementary Experiments"); self.geometry("820x650")
  self.models=tk.StringVar(value="llama3.1:8b"); self.seed=tk.StringVar(value="42"); self.temp=tk.StringVar(value="0.0"); self.top_p=tk.StringVar(value="0.9"); self.dates=tk.StringVar(value="7"); self.repeats=tk.StringVar(value="5"); self.eid=tk.StringVar(value="supplementary_v1"); self.dry=tk.BooleanVar(value=False); self.retry=tk.BooleanVar(value=False)
  f=ttk.Frame(self,padding=14); f.pack(fill="both",expand=True)
  for r,(lab,var) in enumerate((("Ollama models (space-separated)",self.models),("Seed",self.seed),("Temperature",self.temp),("Top-p",self.top_p),("Decision dates",self.dates),("Repeats",self.repeats),("Experiment ID",self.eid))): ttk.Label(f,text=lab).grid(row=r,column=0,sticky="w",pady=5); ttk.Entry(f,textvariable=var,width=55).grid(row=r,column=1,sticky="ew")
  self.checks={}; base=8
  for j,x in enumerate(("bridge","baseline","repeatability","projection_ablation","p5_audit")):
   v=tk.BooleanVar(value=x in ("bridge","baseline","repeatability")); self.checks[x]=v; ttk.Checkbutton(f,text=x,variable=v).grid(row=base+j,column=0,columnspan=2,sticky="w")
  ttk.Checkbutton(f,text="Dry run / synthetic data (smoke test)",variable=self.dry).grid(row=14,column=0,columnspan=2,sticky="w",pady=8)
  ttk.Checkbutton(f,text="Retry previously failed tasks (completed tasks are always skipped)",variable=self.retry).grid(row=15,column=0,columnspan=2,sticky="w")
  ttk.Button(f,text="Run / Resume",command=self.run).grid(row=16,column=0,sticky="w"); self.log=tk.Text(f,height=15); self.log.grid(row=17,column=0,columnspan=2,sticky="nsew",pady=8); f.columnconfigure(1,weight=1); f.rowconfigure(17,weight=1)
 def run(self):
  ex=[k for k,v in self.checks.items() if v.get()]
  if not ex: return messagebox.showerror("Selection","Select at least one experiment")
  cmd=[sys.executable,"-m","supplementary_experiments.runner","--experiments",*ex,"--models",*self.models.get().split(),"--seed",self.seed.get(),"--temperature",self.temp.get(),"--top-p",self.top_p.get(),"--max-dates",self.dates.get(),"--repeats",self.repeats.get(),"--experiment-id",self.eid.get()]
  if self.dry.get(): cmd += ["--dry-run","--synthetic-data"]
  if self.retry.get(): cmd += ["--retry-failed"]
  self.log.insert("end"," ".join(cmd)+"\n"); self.update()
  p=subprocess.run(cmd,text=True,capture_output=True); self.log.insert("end",p.stdout+p.stderr+f"\nexit={p.returncode}\n"); self.log.see("end")
if __name__=="__main__": App().mainloop()
