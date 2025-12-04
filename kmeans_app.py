
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # Necesario para 3D

class ClusteringApp:
    def __init__(self, master):
        self.master = master
        master.title("Grupo 5 — K-Means Clustering de Clientes")

        self.df = None
        self.features = []
        self.cluster_labels = None
        self.k_results = {}

        # Variables de control
        self.selected_vars = []
        self.k_var = tk.StringVar(value="3")
        self.random_state_var = tk.StringVar(value="42")
        self.scale_var = tk.BooleanVar(value=True)
        self.plot_canvas = None

        self.setup_ui()

    def setup_ui(self):
        # Colores
        UTP_RED = "#C8102E"
        UTP_RED_PASTEL = "#F4D6DB"
        UTP_WHITE = "#FFFFFF"
        EXCEL_GREEN = "#1D6F42"

        self.master.configure(bg=UTP_WHITE)
        self.master.state('zoomed')

        # --- Estilos ttk ---
        style = ttk.Style()
        style.configure("Title.TLabelframe.Label", foreground=UTP_RED, font=("Segoe UI", 12, "bold"))
        style.configure("TLabel", background=UTP_WHITE, font=("Segoe UI", 10))
        style.configure("TFrame", background=UTP_WHITE)
        style.configure("TCheckbutton", background=UTP_WHITE)

        # --- PANEL IZQUIERDO ---
        control_frame = tk.Frame(
            self.master,
            bg=UTP_RED_PASTEL,
            padx=14,
            pady=14,
            bd=0
        )
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Título del panel usando ttk
        lab = ttk.Label(control_frame, text="Configuración y Ejecución", font=("Segoe UI", 12, "bold"))
        lab.pack(anchor="w", pady=(0, 10))

        # BOTÓN CARGAR CSV
        tk.Button(
            control_frame,
            text="📁  Cargar CSV",
            bg=EXCEL_GREEN,
            fg="white",
            font=("Segoe UI", 10, "bold"),
            height=1,
            command=self.load_csv
        ).pack(fill=tk.X, pady=6)

        # Selección de variables
        ttk.Label(control_frame, text="2. Seleccionar 2 o 3 Variables:", background=UTP_RED_PASTEL,
                  font=("Segoe UI", 10, "bold")).pack(anchor='w', pady=(10, 0))

        vars_container = tk.Frame(control_frame, bg=UTP_RED_PASTEL)
        vars_container.pack(fill=tk.X)
        self.vars_frame = vars_container

        # Parámetros
        param_frame = ttk.LabelFrame(control_frame, text="3. Parámetros K-Means", padding=10)
        param_frame.pack(fill=tk.X, pady=10)

        ttk.Label(param_frame, text="Valor de K (≥ 2):").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(param_frame, textvariable=self.k_var, width=10).grid(row=0, column=1)

        ttk.Label(param_frame, text="Random State:").grid(row=1, column=0, sticky='w', pady=5)
        ttk.Entry(param_frame, textvariable=self.random_state_var, width=10).grid(row=1, column=1)

        ttk.Checkbutton(param_frame, text="Aplicar StandardScaler", variable=self.scale_var).grid(
            row=2, column=0, columnspan=2, sticky='w', pady=5
        )

        # BOTÓN EJECUTAR K-MEANS (Rojo)
        tk.Button(
            control_frame,
            text="▶ Ejecutar K-Means",
            bg=UTP_RED,
            fg="white",
            font=("Segoe UI", 10, "bold"),
            height=1,
            command=self.run_kmeans
        ).pack(fill=tk.X, pady=8)

        # BOTÓN GUARDAR RESULTADOS
        tk.Button(
            control_frame,
            text="💾 Guardar Resultados",
            bg="#444444",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            height=1,
            command=self.save_results
        ).pack(fill=tk.X, pady=8)

        # --- PANEL DERECHO ---
        results_frame = ttk.Frame(self.master, padding=10)
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Pestañas
        data_tab = ttk.Frame(self.notebook)
        plot_tab = ttk.Frame(self.notebook)
        self.notebook.add(data_tab, text="📄 Datos y Estadísticas")
        self.notebook.add(plot_tab, text="📊 Visualización 2D/3D")

        self.setup_data_tab(data_tab)
        self.setup_plot_tab(plot_tab)

    def setup_data_tab(self, tab):
        ttk.Label(tab, text="Primeros 10 Registros del CSV:").pack(anchor='w', pady=(0, 5))
        self.tree = ttk.Treeview(tab, height=5)
        self.tree.pack(fill=tk.X, pady=5)

        ttk.Label(tab, text="\nEstadísticas de Clústeres (Tamaño y Medias):").pack(anchor='w', pady=(10, 5))
        self.stats_tree = ttk.Treeview(tab, height=6)
        self.stats_tree.pack(fill=tk.X, pady=5)

        self.metrics_label = ttk.Label(tab, text="Métricas: Inercia = N/A | Silhouette = N/A")
        self.metrics_label.pack(anchor='w', pady=8)

    def setup_plot_tab(self, tab):
        self.plot_container = ttk.Frame(tab)
        self.plot_container.pack(fill=tk.BOTH, expand=True)

    def load_csv(self):
        file_path = filedialog.askopenfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="clientes_riesgoB.csv"
        )

        if not file_path:
            return

        try:
            self.df = pd.read_csv(file_path)

            # Solo numéricas
            self.features = [
                col for col in self.df.columns
                if pd.api.types.is_numeric_dtype(self.df[col])
            ]

            # Imputación con la media
            for col in self.features:
                if self.df[col].isnull().any():
                    self.df[col].fillna(self.df[col].mean(), inplace=True)

            self.display_data_preview()
            self.update_variable_selection()

            messagebox.showinfo(
                "Carga Exitosa",
                f"CSV '{file_path.split('/')[-1]}' cargado correctamente.\n"
                f"Variables numéricas disponibles: {len(self.features)}"
            )

        except Exception as e:
            messagebox.showerror("Error al cargar CSV", str(e))
            self.df = None

    def display_data_preview(self):
        # Limpiar Treeview
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Configurar columnas del Treeview
        if self.df is not None:
            self.tree["columns"] = list(self.df.columns)
            self.tree["show"] = "headings"
            for col in self.df.columns:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=80)

            # Insertar los primeros 10 registros
            for i, row in self.df.head(10).iterrows():
                self.tree.insert("", "end", values=list(row))

    def update_variable_selection(self):
        # Limpiar el frame de selección de variables
        for widget in self.vars_frame.winfo_children():
            widget.destroy()

        self.selected_vars_tk = {}
        row, col = 0, 0
        for feature in self.features:
            var = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(self.vars_frame, text=feature, variable=var, command=self.on_var_select)
            cb.grid(row=row, column=col, sticky='w', padx=5, pady=2)
            self.selected_vars_tk[feature] = var
            col += 1
            if col > 1:
                col = 0
                row += 1

    def on_var_select(self):
        # Actualiza la lista de variables seleccionadas y verifica la restricción 2-3
        self.selected_vars = [name for name, var in self.selected_vars_tk.items() if var.get()]
        
        if len(self.selected_vars) > 3:
            messagebox.showwarning("Advertencia de Variables", "Solo se permiten 2 o 3 variables para la visualización.")
            # Desmarca la última variable seleccionada si excede el límite
            last_selected = self.selected_vars.pop()
            self.selected_vars_tk[last_selected].set(False)

    def run_kmeans(self):
        if self.df is None or not self.selected_vars:
            messagebox.showerror("Error de Ejecución", "Primero cargue el CSV y seleccione 2 o 3 variables.")
            return

        n_vars = len(self.selected_vars)
        if n_vars not in [2, 3]:
            messagebox.showerror("Error de Ejecución", f"Debe seleccionar 2 o 3 variables. Actualmente hay {n_vars} seleccionadas.")
            return

        try:
            k = int(self.k_var.get())
            if k < 2:
                raise ValueError("K debe ser un entero mayor o igual a 2.")
            
            random_state = int(self.random_state_var.get()) if self.random_state_var.get().isdigit() else None

            # Preparar los datos
            X = self.df[self.selected_vars].copy()

            # Escalado opcional
            if self.scale_var.get():
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_np = X_scaled
            else:
                X_np = X.values

            # Ejecutar K-Means
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
            self.cluster_labels = kmeans.fit_predict(X_np)
            self.cluster_centers = kmeans.cluster_centers_

            # 1. Calcular Métricas
            inertia = kmeans.inertia_
            silhouette = silhouette_score(X_np, self.cluster_labels) if k > 1 else np.nan
            
            self.k_results = {
                'labels': self.cluster_labels,
                'centers': self.cluster_centers,
                'inertia': inertia,
                'silhouette': silhouette,
                'df_scaled': X_np if self.scale_var.get() else X.values,
                'scaler': scaler if self.scale_var.get() else None,
            }
            
            # 2. Mostrar Estadísticas
            self.display_stats(X, self.cluster_labels, k)

            # 3. Visualizar
            self.plot_results(X_np, self.cluster_labels, self.cluster_centers, n_vars, self.scale_var.get())
            
            messagebox.showinfo("K-Means Exitoso", f"Clustering completado con K={k}. Revise las pestañas 'Datos y Estadísticas' y 'Visualización'.")

        except ValueError as e:
            messagebox.showerror("Error de Input", f"Parámetro no válido: {e}")
        except Exception as e:
            messagebox.showerror("Error de K-Means", f"Ocurrió un error durante el clustering: {e}")


    def display_stats(self, X_original, labels, k):
        # 1. Calcular el DataFrame de estadísticas
        X_with_cluster = X_original.copy()
        X_with_cluster['cluster_kmeans'] = labels
        
        # Calcular el tamaño del clúster
        cluster_counts = X_with_cluster['cluster_kmeans'].value_counts().sort_index().rename('Tamaño')

        # Calcular las medias
        cluster_means = X_with_cluster.groupby('cluster_kmeans')[self.selected_vars].mean()
        
        # Combinar y resetear índice para la visualización
        stats_df = pd.concat([cluster_counts, cluster_means], axis=1)
        stats_df.index.name = 'Clúster'
        stats_df = stats_df.reset_index()

        # 2. Actualizar Treeview de Estadísticas
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)

        cols = list(stats_df.columns)
        self.stats_tree["columns"] = cols
        self.stats_tree["show"] = "headings"
        for col in cols:
            self.stats_tree.heading(col, text=col)
            self.stats_tree.column(col, width=100)

        for i, row in stats_df.iterrows():
            self.stats_tree.insert("", "end", values=[f"Clúster {int(row['Clúster'])}"] + [f"{val:.2f}" for val in row.drop('Clúster')])
        
        # 3. Actualizar Métricas
        inertia = self.k_results['inertia']
        silhouette = self.k_results['silhouette']
        self.metrics_label.config(text=f"Métricas: Inercia = {inertia:.2f} | Silhouette = {silhouette:.4f}")


    def plot_results(self, X_data, labels, centers, n_vars, is_scaled):
        # Limpiar el contenedor de gráficos
        for widget in self.plot_container.winfo_children():
            widget.destroy()

        fig = plt.figure(figsize=(8, 6))

        if n_vars == 2:
            ax = fig.add_subplot(111)
            # Puntos de datos
            scatter = ax.scatter(X_data[:, 0], X_data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
            # Centroides
            ax.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, c='red', edgecolor='black', label='Centroides')
            
            # Etiquetas y Título
            ax.set_xlabel(self.selected_vars[0] + (' (Escalado)' if is_scaled else ''))
            ax.set_ylabel(self.selected_vars[1] + (' (Escalado)' if is_scaled else ''))
            ax.set_title(f"K-Means 2D (K={len(centers)})")
            ax.legend(*scatter.legend_elements(), title="Clústeres")
            ax.grid(True)

        elif n_vars == 3:
            ax = fig.add_subplot(111, projection='3d')
            # Puntos de datos
            scatter = ax.scatter(X_data[:, 0], X_data[:, 1], X_data[:, 2], c=labels, cmap='viridis', s=50, alpha=0.7)
            # Centroides
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='X', s=200, c='red', edgecolor='black', label='Centroides')

            # Etiquetas y Título
            ax.set_xlabel(self.selected_vars[0] + (' (Escalado)' if is_scaled else ''))
            ax.set_ylabel(self.selected_vars[1] + (' (Escalado)' if is_scaled else ''))
            ax.set_zlabel(self.selected_vars[2] + (' (Escalado)' if is_scaled else ''))
            ax.set_title(f"K-Means 3D (K={len(centers)})")
            ax.legend(*scatter.legend_elements(), title="Clústeres")

        # Integrar el gráfico en Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()
        
        # Guardar referencia para evitar que sea recolectado por el garbage collector
        self.plot_canvas = canvas 

    def save_results(self):
        if self.df is None or self.cluster_labels is None:
            messagebox.showerror("Error al Guardar", "No hay resultados de K-Means para guardar. Ejecute el clustering primero.")
            return

        try:
            # Crear una copia del DataFrame original y añadir la columna de etiquetas
            df_export = self.df.copy()
            df_export['cluster_kmeans'] = self.cluster_labels
            
            # Pedir al usuario la ubicación para guardar
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                initialfile="clientes_con_clusters.csv"
            )
            
            if file_path:
                df_export.to_csv(file_path, index=False)
                messagebox.showinfo("Guardado Exitoso", f"Resultados guardados en: {file_path}")

        except Exception as e:
            messagebox.showerror("Error al Guardar", f"No se pudo guardar el archivo CSV: {e}")

if __name__ == '__main__':
    root = tk.Tk()

    # Forzamos fondo blanco
    root.configure(bg="#FFFFFF")

    # Cargar tema Azure si está disponible
    try:
        root.tk.call("source", "azure.tcl")
        root.tk.call("set_theme", "light")
    except:
        pass

    app = ClusteringApp(root)
    root.mainloop()
