import tkinter as tk
from tkinter import simpledialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import os
import numpy as np
import func

def create_gui_app():
    root = tk.Tk()
    root.title("Mortality and LAR Visualization Tool")

    try:
        year = simpledialog.askinteger("Input", "Enter the year (1970 ~ 2023):", parent=root)
        if year is None: return
        sex = simpledialog.askstring("Input", "Enter the sex (남자/여자):", parent=root)
        if sex is None: return

        # Use standardized loader from func.py
        year, sex, Dx, Ex, age, observed_mu = func.load_life_table(year, sex)
    except Exception as e:
        messagebox.showerror("Error", str(e))
        root.destroy()
        return

    # Try to load initial params from 측정 결과.xlsx
    init_params = {'a': 1e-4, 'b': 0.1, 'gamma': 0.1, 'c': 1e-4}
    result_file = "측정 결과.xlsx"
    if os.path.exists(result_file):
        row = func.get_data_from_file(result_file, year, sex)
        if row and all(k in row and row[k] is not None for k in ['a', 'b', 'gamma', 'c']):
            init_params = {
                'a': row['a'],
                'b': row['b'],
                'gamma': row['gamma'],
                'c': row['c']
            }

    # Main container frame (split left/right)
    main_container = tk.Frame(root)
    main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Left panel: parameter controls + metrics panel
    control_panel = tk.Frame(main_container, width = 800)
    control_panel.pack(side=tk.LEFT, fill=tk.Y)
    control_panel.pack_propagate(False)

    # Metrics panel (top)
    metrics_frame = tk.LabelFrame(control_panel, text="Fit Metrics", padx=5, pady=5)
    metrics_frame.pack(fill=tk.X, padx=5, pady=10)

    metrics_text = tk.StringVar()
    metrics_label = tk.Label(metrics_frame, textvariable=metrics_text, justify=tk.LEFT, anchor="w")
    metrics_label.pack(fill=tk.X)

    # x* panel (below metrics)
    xstar_frame = tk.LabelFrame(control_panel, text="x* Value", padx=5, pady=5)
    xstar_frame.pack(fill=tk.X, padx=5, pady=10)

    xstar_text = tk.StringVar()
    xstar_label = tk.Label(xstar_frame, textvariable=xstar_text, justify=tk.LEFT, anchor="w")
    xstar_label.pack(fill=tk.X)

    # Scrollable canvas and frame for parameter controls
    canvas_container_left = tk.Canvas(control_panel, width = 200)
    canvas_container_left.pack(side="left", fill="both", expand=True)
    scrollbar_left = tk.Scrollbar(control_panel, orient="vertical", command=canvas_container_left.yview)
    scrollable_frame_left = tk.Frame(canvas_container_left)

    scrollable_frame_left.bind(
        "<Configure>",
        lambda e: canvas_container_left.configure(
            scrollregion=canvas_container_left.bbox("all")
        )
    )

    canvas_container_left.create_window((0, 0), window=scrollable_frame_left, anchor="nw")
    canvas_container_left.configure(yscrollcommand=scrollbar_left.set)
    
    canvas_container_left.pack(side="left", fill="both", expand=True)
    scrollbar_left.pack(side="right", fill="y")
    
    param_labels = ['a', 'b', 'gamma', 'c']
    param_scales = {}
    param_entries = {}
    param_values = {
        'a': tk.DoubleVar(value=init_params['a']),
        'b': tk.DoubleVar(value=init_params['b']),
        'gamma': tk.DoubleVar(value=init_params['gamma']),
        'c': tk.DoubleVar(value=init_params['c'])
    }
    
    bounds = {
        'a': (1e-7, 1e-4),
        'b': (0.05, 0.15),
        'gamma': (0.01, 0.3),
        'c': (1e-6, 1e-3)
    }

    def update_from_entry(p):
        try:
            val = float(param_entries[p].get())
            param_values[p].set(val)
            update_plots()
        except ValueError:
            messagebox.showwarning("Invalid input", f"Parameter {p} must be a number.")

    def reset_param(p):
        param_values[p].set(init_params[p])
        update_plots()

    def adjust_param(p, delta):
        current = param_values[p].get()
        min_val, max_val = bounds[p]
        step = (max_val - min_val) / 1000.0
        new_val = min(max_val, max(min_val, current + delta * step))
        param_values[p].set(new_val)
        update_plots()

    current_focus_param = {"name": None}

    for i, p in enumerate(param_labels):
        row_frame = tk.Frame(scrollable_frame_left)
        row_frame.pack(fill=tk.X, padx=5, pady=4)

        label = tk.Label(row_frame, text=f'Parameter {p}:', font=("Arial", 11))
        label.pack(side=tk.LEFT, padx=5)

        scale = tk.Scale(
            row_frame,
            from_=bounds[p][0],
            to=bounds[p][1],
            orient=tk.HORIZONTAL,
            resolution=(bounds[p][1] - bounds[p][0]) / 1000,
            variable=param_values[p],
            length=300
        )
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        param_scales[p] = scale

        entry = tk.Entry(row_frame, width=12, font=("Arial", 11))
        entry.insert(0, str(param_values[p].get()))
        entry.pack(side=tk.LEFT, padx=5)
        param_entries[p] = entry

        btn = tk.Button(row_frame, text="Set", width=6, font=("Arial", 10), command=lambda p=p: update_from_entry(p))
        btn.pack(side=tk.LEFT, padx=2)

        reset_btn = tk.Button(row_frame, text="Reset", width=6, font=("Arial", 10), command=lambda p=p: reset_param(p))
        reset_btn.pack(side=tk.LEFT, padx=2)

        entry.bind("<FocusIn>", lambda e, p=p: current_focus_param.update({"name": p}))

    def handle_left(event):
        if current_focus_param["name"]:
            adjust_param(current_focus_param["name"], -1)

    def handle_right(event):
        if current_focus_param["name"]:
            adjust_param(current_focus_param["name"], 1)

    root.bind_all("<Left>", handle_left)
    root.bind_all("<Right>", handle_right)

    # Right panel: graphs
    graph_panel = tk.Frame(main_container)
    graph_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    sex_title = 'Male' if sex == '남자' else 'Female'
    fig.suptitle(f'{year} {sex_title} Mortality Comparison', fontsize=16)

    canvas = FigureCanvasTkAgg(fig, master=graph_panel)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas, graph_panel)
    toolbar.update()
    canvas_widget.pack(fill=tk.BOTH, expand=True)

    def update_plots(event=None):
        a = param_values['a'].get()
        b = param_values['b'].get()
        gamma = param_values['gamma'].get()
        c = param_values['c'].get()

        # keep entries in sync with sliders
        for p in param_labels:
            param_entries[p].delete(0, tk.END)
            param_entries[p].insert(0, f"{param_values[p].get():.6g}")

        ax1.clear()
        ax1.plot(age, observed_mu, 'o-', label=f'Observed data')
        mu_ggm, _ = func.calc_ggm([a, b, gamma, c], age)
        ax1.plot(age, mu_ggm, 'r-', label=f'GGM Model')
        ax1.set_xlabel('Age')
        ax1.set_ylabel('Mortality ($m_x$)')
        ax1.grid(True)
        ax1.set_title(f'Observed vs. GGM Model\n(a={a:.2e}, b={b:.3f}, γ={gamma:.3f}, c={c:.2e})')
        ax1.legend()

        ax2.clear()
        lar = func.calc_lar([a, b, gamma, c], age)
        ax2.plot(age, lar, 'b-', label='GGM Model LAR')
        ax2.set_xlabel('Age')
        ax2.set_ylabel('LAR')
        ax2.grid(True)
        ax2.set_title('LAR (Life-table Aging Rate)')
        ax2.legend()

        # Update metrics
        metrics = func.evaluate_fit_metrics(observed_mu, mu_ggm, notice = False)
        metrics_text.set(f"RMSE: {metrics['rmse']:.4f}\nMAE: {metrics['mae']:.4f}\nMAPE: {metrics['mape']:.2f}%")

        # Update x*
        try:
            num = (b + c * gamma) * c
            denom = 2 * a * b
            root_numer = (b + c * gamma) * c * gamma * ((b + c * gamma) * c - 4 * b * (a * gamma - b))
            root_denom = 2 * a * b * gamma
            log_argument = (num / denom) + (np.sqrt(root_numer) / root_denom)
        
            x_star = (1 / b) * np.log(log_argument)
            xstar_text.set(f"x* = {x_star:.2f}")
        except Exception:
            xstar_text.set("x* = N/A")
        
        fig.tight_layout()
        canvas.draw()

    for p in param_labels:
        param_scales[p].config(command=update_plots)

    update_plots()
    
    root.mainloop()

if __name__ == "__main__":
    create_gui_app()
