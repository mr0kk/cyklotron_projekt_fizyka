import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches
import sys

class CyclotronApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Symulator Cyklotronu - FOGT 25Z")
        self.root.geometry("1200x900")

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.particles = {
            "Proton": {"m": 1.67e-27, "q": 1.60e-19, "color": "#1f77b4", "desc": "Lekki, ładunek +1"},
            "Cząstka Alfa": {"m": 6.64e-27, "q": 3.20e-19, "color": "#d62728", "desc": "Ciężka, ładunek +2"},
            "Jon Węgla (C12)": {"m": 1.99e-26, "q": 9.61e-19, "color": "#2ca02c", "desc": "Bardzo ciężki, jon dodatni"}
        }

        # --- Zmienne stanu ---
        self.is_running = False
        self.pos = np.array([0.0, 0.0])
        self.vel = np.array([0.0, 0.0])
        self.traj_x, self.traj_y = [], []
        self.history = []
        self.phys_widgets = []

        # --- Zmienne UI ---
        self.selected_particle = tk.StringVar(value="Proton")
        self.b_field = tk.DoubleVar(value=1.5)
        self.voltage = tk.DoubleVar(value=15000)
        self.sim_speed = tk.IntVar(value=3)
        self.start_vel_factor = tk.DoubleVar(value=1.5)
        self.playback_var = tk.IntVar(value=0)

        self.setup_ui()
        self.init_plot()

    def create_slider_row(self, parent, label_text, variable, from_, to, unit="", is_int=False, is_phys=True):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)
        ttk.Label(frame, text=label_text, font=('Arial', 9, 'bold')).pack(anchor=tk.W)
        row = ttk.Frame(frame)
        row.pack(fill=tk.X)
        scale = ttk.Scale(row, from_=from_, to=to, variable=variable, orient=tk.HORIZONTAL)
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        val_label = ttk.Label(row, text="", width=12, font=('Consolas', 10, 'bold'), foreground="#2980b9")
        val_label.pack(side=tk.RIGHT)
        if is_phys:
            self.phys_widgets.append(scale)
        def update_label(*args):
            val = variable.get()
            val_label.config(text=f"{int(val) if is_int else val:.2f} {unit}")
        variable.trace_add("write", update_label)
        update_label()
        return frame

    def setup_ui(self):
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.left_side = ttk.Frame(self.main_frame)
        self.left_side.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.plot_frame = ttk.Frame(self.left_side)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        self.pb_frame = ttk.LabelFrame(self.left_side, text=" Analiza klatka po klatce ", padding="10")
        self.pb_frame.pack(fill=tk.X, pady=(10, 0))
        self.playback_slider = ttk.Scale(self.pb_frame, from_=0, to=100, variable=self.playback_var, orient=tk.HORIZONTAL, command=self.manual_step)
        self.playback_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.playback_label = ttk.Label(self.pb_frame, text="Klatka: 0", width=15, font=('Consolas', 9))
        self.playback_label.pack(side=tk.RIGHT)

        self.ctrl_frame = ttk.LabelFrame(self.main_frame, text=" Parametry Symulacji ", padding="15")
        self.ctrl_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        ttk.Label(self.ctrl_frame, text="RODZAJ CZĄSTKI:", font=('Arial', 9, 'bold')).pack(anchor=tk.W)
        self.part_cb = ttk.Combobox(self.ctrl_frame, textvariable=self.selected_particle, values=list(self.particles.keys()), state="readonly")
        self.part_cb.pack(fill=tk.X, pady=(5, 15))
        self.part_cb.bind("<<ComboboxSelected>>", lambda e: self.reset_sim())
        self.phys_widgets.append(self.part_cb)

        self.create_slider_row(self.ctrl_frame, "PRĘDKOŚĆ STARTOWA [V0]:", self.start_vel_factor, 0.1, 10.0, unit="j.", is_phys=True)
        self.create_slider_row(self.ctrl_frame, "TEMPO ANIMACJI (KROKI):", self.sim_speed, 1, 40, unit="st.", is_int=True, is_phys=False)
        ttk.Separator(self.ctrl_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        self.create_slider_row(self.ctrl_frame, "POLE MAGNETYCZNE B:", self.b_field, 0.5, 3.0, unit="T", is_phys=True)
        self.create_slider_row(self.ctrl_frame, "NAPIĘCIE ELEKTROD V:", self.voltage, 5000, 50000, unit="V", is_int=True, is_phys=True)

        self.status_box = tk.Label(self.ctrl_frame, text="GOTOWY", font=('Courier', 10, 'bold'), bg='#34495e', fg='white', height=4, width=40, justify=tk.CENTER, relief=tk.RIDGE, borderwidth=2)
        self.status_box.pack(pady=25)

        ttk.Button(self.ctrl_frame, text="▶ START", command=self.start_sim).pack(fill=tk.X, pady=2)
        ttk.Button(self.ctrl_frame, text="■ STOP", command=self.stop_sim).pack(fill=tk.X, pady=2)
        ttk.Button(self.ctrl_frame, text="↺ RESETUJ", command=self.reset_sim).pack(fill=tk.X, pady=2)

    def init_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.draw_cyclotron()

    def set_phys_controls_state(self, state):
        for w in self.phys_widgets:
            if isinstance(w, ttk.Combobox):
                w.config(state="readonly" if state == "normal" else "disabled")
            else:
                w.config(state=state)

    def draw_cyclotron(self):
        self.ax.clear()
        r_dee, self.gap_w = 0.5, 0.06
        t = np.linspace(-np.pi / 2, np.pi / 2, 100)
        self.ax.set_aspect('equal', adjustable='box')
        self.dee_l = self.ax.fill_betweenx(r_dee * np.sin(t), -self.gap_w / 2 - r_dee * np.cos(t), -self.gap_w / 2, color='#bdc3c7', alpha=0.4)
        self.dee_r = self.ax.fill_betweenx(r_dee * np.sin(t), self.gap_w / 2, self.gap_w / 2 + r_dee * np.cos(t), color='#bdc3c7', alpha=0.4)
        self.ax.add_patch(mpatches.Circle((0, 0), 0.015, color='#f39c12', ec='black', zorder=5))
        self.field_arrow = self.ax.quiver(0, 0, 1, 0, color='#f1c40f', scale=5, width=0.02, pivot='mid', alpha=0)
        self.ax.set_xlim(-0.6, 0.6); self.ax.set_ylim(-0.6, 0.6); self.ax.axis('off')
        p = self.particles[self.selected_particle.get()]
        self.line, = self.ax.plot([], [], color=p["color"], lw=2)
        self.point, = self.ax.plot([0], [0], 'o', color='black', markersize=7, zorder=10)
        self.canvas.draw()

    def start_sim(self):
        if not self.is_running:
            self.set_phys_controls_state("disabled")
            self.playback_slider.config(state="disabled")
            if not self.history:
                v0 = self.start_vel_factor.get() * 1.5e5
                self.vel = np.array([v0, 0.0])
            self.is_running = True
            self.animate()

    def stop_sim(self):
        self.is_running = False
        self.set_phys_controls_state("normal")
        self.playback_slider.config(state="normal")

    def reset_sim(self):
        self.stop_sim()
        self.pos, self.traj_x, self.traj_y, self.history = np.array([0.0, 0.0]), [], [], []
        self.playback_slider.config(to=0); self.playback_var.set(0)
        self.draw_cyclotron()

    def animate(self):
        if not self.is_running: return
        p = self.particles[self.selected_particle.get()]
        m, q, B, V = p['m'], p['q'], self.b_field.get(), self.voltage.get()
        dt = 6e-10
        for _ in range(self.sim_speed.get()):
            ax = (q * B / m) * self.vel[1]
            ay = -(q * B / m) * self.vel[0]
            in_gap = abs(self.pos[0]) < self.gap_w / 2
            dir_e = 0
            if in_gap:
                dir_e = 1 if self.vel[0] > 0 else -1
                ax += (q * V / (m * self.gap_w)) * dir_e
            self.vel += np.array([ax, ay]) * dt
            self.pos += self.vel * dt
            self.traj_x.append(self.pos[0]); self.traj_y.append(self.pos[1])
            self.history.append({'tx': list(self.traj_x), 'ty': list(self.traj_y), 'px': self.pos[0], 'py': self.pos[1], 'dir': dir_e, 'in_gap': in_gap})
        self.apply_state(self.history[-1])
        self.playback_slider.config(to=len(self.history) - 1); self.playback_var.set(len(self.history) - 1)
        if np.linalg.norm(self.pos) < 0.51:
            self.root.after(20, self.animate)
        else:
            self.stop_sim()

    def manual_step(self, val):
        if not self.history or self.is_running: return
        idx = int(float(val))
        self.apply_state(self.history[idx])
        self.playback_label.config(text=f"Klatka: {idx}")

    def apply_state(self, s):
        self.line.set_data(s['tx'], s['ty'])
        self.point.set_data([s['px']], [s['py']])
        if s['in_gap']:
            self.field_arrow.set_alpha(1); self.field_arrow.set_UVC(s['dir'], 0)
            c = '#c0392b' if s['dir'] > 0 else '#2980b9'
            self.dee_l.set_facecolor('#e74c3c' if s['dir'] > 0 else '#3498db')
            self.dee_r.set_facecolor('#3498db' if s['dir'] > 0 else '#e74c3c')
            self.status_box.config(text="PRZYSPIESZANIE W SZCZELINIE", bg=c)
        else:
            self.field_arrow.set_alpha(0)
            self.dee_l.set_facecolor('#bdc3c7'); self.dee_r.set_facecolor('#bdc3c7')
            self.status_box.config(text="ZAKRZYWIANIE (POLE B)", bg='#2c3e50')
        self.canvas.draw_idle()

    # NOWA METODA: Czyści pamięć i zamyka wszystkie okna
    def on_closing(self):
        self.is_running = False  # Zatrzymuje animate()
        plt.close('all')        # Zamyka okna wykresów Matplotlib
        self.root.destroy()     # Zamyka okno Tkinter
        sys.exit()              # Kończy proces Pythona

if __name__ == "__main__":
    root = tk.Tk()
    app = CyclotronApp(root)
    root.mainloop()